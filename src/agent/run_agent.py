"""
Runs the LangGraph agent over cases from Phase 3/4 and writes:
    data/processed/case_results.json   -> full per-case state (narrative, citations, status)
    data/processed/audit_log.jsonl     -> one line per case, append-only, for FR-10 auditability

Incrementally saves after every batch of completed cases (not just at the end) and skips
cases already present in case_results.json, so a network blip or crash loses at most the
in-flight cases and the run can simply be re-invoked to resume.

Cases are processed CONCURRENTLY via a thread pool (--workers, default 5). This is safe
because agent.invoke() per case is independent -- each call gets its own fresh state -- and
the work here is I/O-bound (waiting on LLM API responses), so threads give a near-linear
speedup even under Python's GIL. A lock protects the shared results dict and both output
files so concurrent writes never interleave or corrupt data.

Run:
    python -m src.agent.run_agent                    # all cases (resumes automatically)
    python -m src.agent.run_agent --limit 10          # first 10 (for a quick/cheap test run)
    python -m src.agent.run_agent --fresh             # ignore existing results, start over
    python -m src.agent.run_agent --workers 8         # more/less parallel LLM calls
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from src.agent.graph import build_agent

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROC = Path("data/processed")
RESULTS_PATH = PROC / "case_results.json"
AUDIT_PATH = PROC / "audit_log.jsonl"
MAX_RETRIES = 3
RETRY_DELAY_SECS = 10
DEFAULT_WORKERS = 5          # concurrent LLM-calling threads; raise/lower based on your API rate limit
CHECKPOINT_EVERY = 10        # save case_results.json after every N newly-completed cases

# One compiled agent is reused across all threads. LangGraph's invoke() builds a fresh
# execution state per call, so concurrent invoke() calls on the same compiled graph are
# safe *as long as* the underlying node code (src/agent/nodes.py, llm.py) doesn't rely on
# one shared mutable client/session across threads. If you see odd/garbled responses or
# connection errors only under concurrency (not with --workers 1), that's the place to look
# -- e.g. a module-level httpx/OpenAI client reused across threads may need to be created
# fresh per call or made explicitly thread-safe.
_agent = build_agent()

# Guards: results_by_id dict mutation, case_results.json writes, audit_log.jsonl appends,
# and the periodic-checkpoint counter -- all from multiple worker threads at once.
_lock = threading.Lock()


def load_existing() -> dict[int, dict]:
    if not RESULTS_PATH.exists():
        return {}
    try:
        existing = json.load(open(RESULTS_PATH))
        return {r["case_id"]: r for r in existing}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_results(results_by_id: dict[int, dict]) -> None:
    ordered = [results_by_id[k] for k in sorted(results_by_id)]
    with open(RESULTS_PATH, "w") as f:
        json.dump(ordered, f, indent=2, default=str)


def append_audit(result: dict) -> None:
    line = json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": result["case_id"],
        "decision": result["decision"],
        "decision_rationale": result["decision_rationale"],
        "status": result["status"],
        "citation_accuracy": result.get("citation_accuracy"),
        "faithfulness_score": result.get("faithfulness_score"),
        "narrative_attempts": result.get("narrative_attempts", 0),
    }, default=str)
    with open(AUDIT_PATH, "a") as f:
        f.write(line + "\n")


def process_case(case: dict) -> dict:
    """Runs the agent on one case, with retries. Pure function -- no shared state touched,
    so this is what each worker thread calls; safe to run many of these concurrently."""
    cid = case["case_id"]
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _agent.invoke({"case_id": cid, "case": case})
            break
        except Exception as e:  # noqa: BLE001 -- network blips, API throttling, etc.
            print(f"  case_id={cid} attempt {attempt}/{MAX_RETRIES} failed: "
                  f"{type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                result = {
                    "case_id": cid, "case": case, "decision": case.get("decision", "UNKNOWN"),
                    "decision_rationale": "N/A", "status": "AGENT_ERROR",
                    "error": f"{type(e).__name__}: {e}",
                }
            else:
                time.sleep(RETRY_DELAY_SECS)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fresh", action="store_true", help="ignore existing case_results.json, start over")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="number of cases to process concurrently (default: %(default)s)")
    args = p.parse_args()

    cases = json.load(open(PROC / "case_decisions.json"))
    if args.limit:
        cases = cases[: args.limit]

    results_by_id = {} if args.fresh else load_existing()
    if results_by_id:
        print(f"Resuming: {len(results_by_id)} case(s) already done, will be skipped.")

    pending = [c for c in cases if c["case_id"] not in results_by_id]
    total = len(cases)
    print(f"Processing {len(pending)} pending case(s) out of {total} total, "
          f"with {args.workers} concurrent worker(s).")

    done_this_run = 0
    processed_total = len(results_by_id)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, case): case["case_id"] for case in pending}

        for future in as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001 -- should be rare, process_case already retries
                result = {
                    "case_id": cid, "decision": "UNKNOWN", "decision_rationale": "N/A",
                    "status": "AGENT_ERROR", "error": f"unhandled: {type(e).__name__}: {e}",
                }

            with _lock:
                results_by_id[cid] = result
                append_audit(result)
                done_this_run += 1
                processed_total += 1

                if done_this_run % CHECKPOINT_EVERY == 0:
                    save_results(results_by_id)  # periodic checkpoint

                if processed_total % 25 == 0 or processed_total == total:
                    print(f"  processed {processed_total}/{total} (this run: {done_this_run})")

    with _lock:
        save_results(results_by_id)  # final save

    all_results = list(results_by_id.values())
    print("\n===== RUN SUMMARY (all cases so far, including resumed) =====")
    for status in sorted(set(r["status"] for r in all_results)):
        n = sum(1 for r in all_results if r["status"] == status)
        print(f"  {status}: {n}")

    errored = [r for r in all_results if r["status"] == "AGENT_ERROR"]
    if errored:
        print(f"\n  {len(errored)} case(s) failed after {MAX_RETRIES} retries -- re-run this "
              f"script to retry them (they ARE marked done with status AGENT_ERROR, so remove "
              f"them from case_results.json first if you want them retried automatically).")

    citation_failed = [r for r in all_results if r["status"] == "ESCALATED_NARRATIVE_FAILED_CITATION_CHECK"]
    if citation_failed:
        print(f"\n  {len(citation_failed)} case(s) failed citation check after retry.")

    faith_failed = [r for r in all_results if r["status"] == "ESCALATED_NARRATIVE_FAILED_FAITHFULNESS_CHECK"]
    if faith_failed:
        print(f"\n  {len(faith_failed)} case(s) failed faithfulness check after retry.")

    verified = [r for r in all_results if r["status"] == "ESCALATED_WITH_VERIFIED_NARRATIVE"]
    if verified:
        avg_faith = sum(r["faithfulness_score"] for r in verified) / len(verified)
        print(f"\n  {len(verified)} verified narrative(s), average faithfulness_score={avg_faith:.3f}")


if __name__ == "__main__":
    main()