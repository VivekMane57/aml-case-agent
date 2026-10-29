import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { Severity } from "../data/types";
import { readDecisions, useCases } from "../data/api";
import { Card, SeverityBadge, StatusBadge } from "../components/Badge";

const PAGE = 25;
const RULE_FILTERS = ["All rules", "Structuring", "Fan pattern", "Amount anomaly", "No rule fired"];

function Stat({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="text-sm text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-bold">{value}</div>
    </div>
  );
}

const pct = (p: number) => `${Math.round(p * 100)}th`;

export default function AlertsPage() {
  const { data, error } = useCases();
  const [query, setQuery] = useState("");
  const [rule, setRule] = useState("All rules");
  const [severity, setSeverity] = useState<"All" | Severity>("All");
  const [shown, setShown] = useState(PAGE);
  const decided = Object.keys(readDecisions());

  const rows = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    return data.filter((a) => {
      if (q && !a.alertId.toLowerCase().includes(q) && !a.account.includes(q.replace("acc-", ""))) return false;
      if (rule === "No rule fired" && a.rulesFired.length > 0) return false;
      if (rule !== "All rules" && rule !== "No rule fired" && !a.rulesFired.includes(rule)) return false;
      if (severity !== "All" && a.severity !== severity) return false;
      return true;
    });
  }, [data, query, rule, severity]);

  if (error) return <p className="text-red-700">Could not load /data/cases.json. Run python src/api/export_frontend_data.py first. ({error})</p>;
  if (!data) return <p className="text-slate-500">Loading cases…</p>;

  const critical = data.filter((a) => a.severity === "Critical").length;
  const ready = data.filter((a) => a.status === "ESCALATED_WITH_VERIFIED_NARRATIVE").length;

  return (
    <div>
      <h1 className="text-2xl font-bold">Alert queue</h1>
      <p className="mt-2 max-w-xl text-sm text-slate-500">
        Cases built from transactions flagged by the rule engine and the models, ranked by risk. The agent prepares a
        case summary; an analyst makes every decision.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Stat label="Open alerts" value={data.length - decided.length} />
        <Stat label="Critical" value={critical} />
        <Stat label="Summary ready" value={ready} />
        <Stat label="Decided (this browser)" value={decided.length} />
      </div>

      <div className="mt-6 flex flex-wrap items-end gap-4 text-xs font-semibold text-slate-700">
        <label className="flex w-72 flex-col gap-1">
          Search alert or account
          <input
            value={query}
            onChange={(e) => { setQuery(e.target.value); setShown(PAGE); }}
            placeholder="AL-... or ACC-..."
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-normal"
          />
        </label>
        <label className="flex flex-col gap-1">
          Rule triggered
          <select value={rule} onChange={(e) => { setRule(e.target.value); setShown(PAGE); }} className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-normal">
            {RULE_FILTERS.map((t) => <option key={t}>{t}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Severity
          <select value={severity} onChange={(e) => { setSeverity(e.target.value as "All" | Severity); setShown(PAGE); }} className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-normal">
            <option>All</option><option>Critical</option><option>High</option><option>Medium</option>
          </select>
        </label>
      </div>

      <Card className="mt-6 !p-0 overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4 text-sm">
          <span className="font-semibold">Ranked by risk percentile, highest first</span>
          <span className="text-slate-500">{rows.length} of {data.length} cases</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-100 text-xs font-semibold text-slate-600">
              <tr>
                {["Alert", "Account", "Pattern seen", "Rules", "Txns", "Score", "Percentile", "Severity", "Status", ""].map((h) => (
                  <th key={h} className="px-4 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, shown).map((a) => (
                <tr key={a.alertId} className="border-t border-slate-100">
                  <td className="px-4 py-3 font-mono font-semibold">{a.alertId}</td>
                  <td className="px-4 py-3 text-slate-600">ACC-{a.account}</td>
                  <td className="max-w-[280px] px-4 py-3">{a.pattern}</td>
                  <td className="px-4 py-3 text-slate-600">{a.rulesFired.length} of 3 rules</td>
                  <td className="px-4 py-3 text-slate-600">{a.numTransactions}</td>
                  <td className="px-4 py-3 font-mono text-slate-600">{a.score.toFixed(4)}</td>
                  <td className="px-4 py-3 text-slate-600">{pct(a.percentile)}</td>
                  <td className="px-4 py-3"><SeverityBadge value={a.severity} /></td>
                  <td className="px-4 py-3"><StatusBadge value={a.statusLabel} /></td>
                  <td className="px-4 py-3 text-right">
                    <Link to={`/case/${a.alertId}`} className="rounded-md border border-[#1e3a6e] px-4 py-2 text-sm font-semibold text-[#0f2447] hover:bg-slate-50">
                      Investigate
                    </Link>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr><td colSpan={10} className="px-4 py-8 text-center text-slate-500">No alerts match these filters.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        {shown < rows.length && (
          <div className="border-t border-slate-100 p-4 text-center">
            <button onClick={() => setShown((n) => n + PAGE)} className="rounded-md border border-slate-300 px-4 py-2 text-sm font-semibold hover:bg-slate-50">
              Show {Math.min(PAGE, rows.length - shown)} more
            </button>
          </div>
        )}
      </Card>

      <p className="mt-4 max-w-2xl text-xs text-slate-500">
        Synthetic SAML-D data. Severity comes from the case's risk percentile (raw scores all sit between 0.985 and 1.0,
        so they barely separate cases). Scores order the queue; they never close or file a case.
      </p>
    </div>
  );
}
