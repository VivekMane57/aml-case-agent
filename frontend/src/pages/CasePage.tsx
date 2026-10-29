import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { readDecisions, saveDecision, useCases } from "../data/api";
import { Card, Chip, SeverityBadge, StatusBadge } from "../components/Badge";

const DECISIONS = ["Escalate for senior review", "Recommend filing a report", "Close as false positive"] as const;
const SCORED_RULES = ["Structuring", "Amount anomaly", "Fan pattern"];

function Bar({ value }: { value: number }) {
  return (
    <div className="mt-2 h-2 rounded-full bg-slate-200">
      <div className="h-2 rounded-full bg-[#0f2447]" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

function Fact({ label, name, from }: { label: string; name: string; from: string }) {
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 font-semibold">{name}</div>
      <div className="text-xs text-slate-500">From: {from}</div>
    </div>
  );
}

// The agent cites transactions as [1234567]. Show them as chips instead of raw brackets.
function Narrative({ text }: { text: string }) {
  const sentences = text.split(/(?<=[.!?])\s+(?=[A-Z])/);
  return (
    <div className="space-y-3 text-sm">
      {sentences.map((s, i) => {
        const ids = [...s.matchAll(/\[(\d+)\]/g)].map((m) => m[1]);
        const clean = s.replace(/\s*\[\d+\]/g, "");
        return (
          <p key={i}>
            {clean}{" "}
            <span className="ml-1 inline-flex flex-wrap gap-1">{ids.map((id) => <Chip key={id}>TX-{id}</Chip>)}</span>
          </p>
        );
      })}
    </div>
  );
}

const money = (n: number) => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export default function CasePage() {
  const { alertId = "" } = useParams();
  const { data, error } = useCases();
  const [choice, setChoice] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const [, bump] = useState(0);

  if (error) return <p className="text-red-700">Could not load case data. ({error})</p>;
  if (!data) return <p className="text-slate-500">Loading case…</p>;

  const c = data.find((x) => x.alertId === alertId);
  if (!c) {
    return (
      <div>
        <Link to="/" className="text-sm underline">Alerts</Link>
        <p className="mt-4 text-slate-600">No case named {alertId}. Pick one from the alert queue.</p>
      </div>
    );
  }

  const saved = readDecisions()[c.alertId];
  const canRecord = choice !== null && reason.trim().length > 0 && !saved;
  const unverified = c.status.startsWith("ESCALATED_NARRATIVE_FAILED");
  const suggestion = c.decision === "ESCALATE" ? "Escalate for senior review" : "Request more evidence";
  const rulesN = c.rulesFired.length;

  return (
    <div>
      <div className="text-sm text-slate-500">
        <Link to="/" className="underline">Alerts</Link> / {c.alertId}
      </div>

      <Card className="mt-4 flex flex-wrap items-center gap-x-10 gap-y-3">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="font-mono text-2xl font-bold">{c.alertId}</h1>
            <SeverityBadge value={c.severity} />
            <StatusBadge value={saved ? "Decided" : "Awaiting decision"} />
            <StatusBadge value={c.statusLabel} />
          </div>
          <div className="mt-1 text-xs text-slate-500">Account ACC-{c.account} · name masked · {c.numAccounts} accounts in case</div>
        </div>
        <div className="text-xs text-slate-500">Window reviewed<div className="text-sm font-semibold text-slate-900">{c.timeStart} → {c.timeEnd}</div></div>
        <div className="text-xs text-slate-500">Transactions<div className="text-sm font-semibold text-slate-900">{c.numTransactions}</div></div>
        <div className="text-xs text-slate-500">Total amount<div className="text-sm font-semibold text-slate-900">{money(c.totalAmount)}</div></div>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          <Card className="border-t-4 border-t-[#0f2447]">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-lg font-bold">Case summary from the agent</h2>
              <span className="rounded-full bg-blue-100 px-3 py-1 text-xs font-semibold text-blue-800">Draft for your review</span>
            </div>

            {c.narrative ? (
              <div className="mt-4">
                {unverified && (
                  <div className="mb-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900">
                    This draft failed the agent's own verification after {c.attempts} attempts
                    ({c.status.includes("CITATION") ? "a cited transaction was not in the evidence" : `faithfulness ${c.faithfulness?.toFixed(2) ?? "n/a"}, below the 0.85 threshold`}).
                    Treat it as unverified and check it against the transactions below.
                  </div>
                )}
                <Narrative text={c.narrative} />
              </div>
            ) : (
              <p className="mt-4 rounded-lg bg-slate-100 p-3 text-sm text-slate-700">
                No summary was written. The risk step asked for more evidence before drafting ({c.decisionRationale}).
              </p>
            )}

            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <Fact label="Pattern" name={c.pattern} from="rule engine" />
              <Fact label="Risk level" name={c.severity} from={`score percentile ${Math.round(c.percentile * 100)}`} />
              <Fact label="Not checked" name="Customer profile, source of funds" from="Not in the dataset" />
            </div>
            <div className="mt-3"><Fact label="Suggested next step" name={suggestion} from="the agent" /></div>
            <p className="mt-4 text-xs text-slate-500">
              {c.narrative
                ? `Cited transactions were checked to exist (accuracy ${((c.citationAccuracy ?? 0) * 100).toFixed(0)}%) and the claims were scored for faithfulness to the evidence (${c.faithfulness?.toFixed(2) ?? "n/a"}). `
                : ""}
              The agent suggests; you decide.
            </p>
          </Card>

          <Card>
            <h2 className="text-lg font-bold">Flow of funds</h2>
            <p className="text-sm text-slate-500">Counterparties of ACC-{c.account} across the transactions in this case.</p>
            <div className="mt-4 grid items-center gap-4 md:grid-cols-[1fr_auto_200px_auto_1fr]">
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-600">Money in (from)</div>
                {c.moneyIn.length === 0 && <div className="text-xs text-slate-400">None in this case</div>}
                {c.moneyIn.slice(0, 6).map((m) => (
                  <div key={m.account} className="flex justify-between rounded-md border border-slate-200 px-3 py-2 text-sm">
                    <span className="font-mono">ACC-{m.account}</span><span className="text-slate-500">{m.count} {m.count === 1 ? "transfer" : "transfers"}</span>
                  </div>
                ))}
                {c.moneyIn.length > 6 && <div className="text-xs text-slate-500">+ {c.moneyIn.length - 6} more</div>}
              </div>
              <span className="hidden text-center text-slate-400 md:block">→</span>
              <div className="rounded-xl border-2 border-[#0f2447] bg-blue-50 p-5 text-center">
                <div className="font-mono font-semibold">ACC-{c.account}</div>
                <div className="text-xs text-slate-500">Account under review</div>
              </div>
              <span className="hidden text-center text-slate-400 md:block">→</span>
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-600">Money out (to)</div>
                {c.moneyOut.length === 0 && <div className="text-xs text-slate-400">None in this case</div>}
                {c.moneyOut.slice(0, 6).map((m) => (
                  <div key={m.account} className="flex justify-between rounded-md border border-slate-200 px-3 py-2 text-sm">
                    <span className="font-mono">ACC-{m.account}</span><span className="text-slate-500">{m.count} {m.count === 1 ? "transfer" : "transfers"}</span>
                  </div>
                ))}
                {c.moneyOut.length > 6 && <div className="text-xs text-slate-500">+ {c.moneyOut.length - 6} more</div>}
              </div>
            </div>
          </Card>

          <Card className="!p-0 overflow-hidden">
            <h2 className="px-6 pt-5 text-lg font-bold">Transactions in the window</h2>
            <div className="overflow-x-auto">
              <table className="mt-4 w-full text-left text-sm">
                <thead className="bg-slate-100 text-xs font-semibold text-slate-600">
                  <tr><th className="px-6 py-3">ID</th><th>Date</th><th>Direction</th><th>Counterparty</th><th>Type</th><th className="text-right">Amount</th><th className="px-6">Flagged by</th></tr>
                </thead>
                <tbody>
                  {c.transactions.map((t) => (
                    <tr key={t.id} className="border-t border-slate-100">
                      <td className="px-6 py-3 font-mono">TX-{t.id}</td>
                      <td>{t.date}</td>
                      <td>{t.direction}</td>
                      <td className="font-mono">ACC-{t.counterparty}</td>
                      <td>{t.paymentType}</td>
                      <td className="text-right">{money(t.amount)}</td>
                      <td className="px-6">{t.flaggedBy}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="px-6 py-4 text-xs text-slate-500">Amounts are as recorded in the dataset (currency not specified in this view).</p>
          </Card>

          <Card>
            <h2 className="text-lg font-bold">Typology notes retrieved</h2>
            <p className="mt-1 text-sm text-slate-500">
              Typology retrieval (the Qdrant step in the PRD) is not built yet, so no guidance notes are shown. The agent
              above works only from the transaction evidence.
            </p>
          </Card>
        </div>

        <aside className="space-y-6">
          <Card className="border-t-4 border-t-[#0f2447]">
            <h2 className="text-lg font-bold">Your decision</h2>
            <p className="text-sm text-slate-500">Nothing is filed or closed until you choose. The agent has no access to this step.</p>
            <div className="mt-4 space-y-2">
              {DECISIONS.map((d) => (
                <button
                  key={d}
                  disabled={!!saved}
                  onClick={() => setChoice(d)}
                  className={`w-full rounded-lg border-2 px-4 py-2.5 text-left text-sm font-semibold ${
                    (saved?.choice ?? choice) === d ? "border-[#0f2447] bg-[#0f2447] text-white" : "border-[#0f2447] text-[#0f2447] hover:bg-slate-50"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
            <label className="mt-4 block text-xs font-semibold text-slate-700">
              Reason (required for every decision)
              <textarea
                value={saved?.reason ?? reason}
                onChange={(e) => setReason(e.target.value)}
                disabled={!!saved}
                rows={4}
                className="mt-1 w-full rounded-md border border-slate-300 p-2 text-sm font-normal"
              />
            </label>
            <button
              disabled={!canRecord}
              onClick={() => { saveDecision({ alertId: c.alertId, choice: choice!, reason: reason.trim(), at: new Date().toISOString() }); bump((n) => n + 1); }}
              className="mt-3 w-full rounded-lg bg-[#0f2447] py-2.5 text-sm font-semibold text-white disabled:bg-slate-200 disabled:text-slate-500"
            >
              Record decision
            </button>
            {saved && <p className="mt-3 text-xs text-green-700">Recorded in this browser on {saved.at.slice(0, 10)}. There is no backend yet, so it is not shared.</p>}
          </Card>

          <Card>
            <h2 className="text-lg font-bold">Signals</h2>
            <p className="text-sm text-slate-500">Checks shown side by side so you can see where they agree.</p>
            <div className="mt-4 space-y-3">
              <div className="rounded-lg border border-slate-200 p-3">
                <div className="flex justify-between text-sm font-semibold"><span>Rule engine</span><span className="font-mono">{rulesN} of 3 rules</span></div>
                <Bar value={rulesN / 3} />
                <p className="mt-2 text-xs text-slate-500">{rulesN ? `${c.rulesFired.join(", ")} fired.` : "No scoring rule fired; the case was raised by the model score."}</p>
              </div>
              <div className="rounded-lg border border-slate-200 p-3">
                <div className="flex justify-between text-sm font-semibold"><span>Model score</span><span className="font-mono">{c.score.toFixed(4)}</span></div>
                <Bar value={c.score} />
                <p className="mt-2 text-xs text-slate-500">Highest score in the case, at the {Math.round(c.percentile * 100)}th percentile of alerted cases.</p>
              </div>
            </div>
            <h3 className="mt-5 text-sm font-bold">Rule hits</h3>
            <div className="mt-2 space-y-2">
              {SCORED_RULES.map((r) => {
                const fired = c.rulesFired.includes(r);
                return (
                  <div key={r} className={`flex justify-between rounded-md px-3 py-2 text-sm ${fired ? "bg-red-50" : "bg-slate-50 text-slate-500"}`}>
                    <span>{r}</span><span className={`font-semibold ${fired ? "text-red-700" : ""}`}>{fired ? "Fired" : "Not fired"}</span>
                  </div>
                );
              })}
              <div className="flex justify-between rounded-md bg-slate-100 px-3 py-2 text-sm text-slate-600">
                <span>Velocity</span><span className="font-semibold">Not used in score</span>
              </div>
            </div>
            <p className="mt-4 text-xs text-slate-500">The model score orders the alerts. It is not proof of laundering.</p>
          </Card>
        </aside>
      </div>
    </div>
  );
}
