import { LAYERS, LIMITATIONS, RULES } from "../data/evalData";
import { useSummary } from "../data/api";
import { Card } from "../components/Badge";

// A value that has not been produced by an eval script yet is shown as a placeholder, never invented.
const PENDING = "[from eval script]";

function Val({ v }: { v: string | null }) {
  return <span className="font-mono text-sm">{v ?? PENDING}</span>;
}

export default function EvaluationPage() {
  const { data: sm } = useSummary();
  const checks = [
    { name: "Typology retrieval, Precision@3", sub: "On a hand-labelled question set", value: null as string | null },
    {
      name: "Citation faithfulness",
      sub: "Every cited reference exists and supports the sentence",
      value: sm ? `${((sm.verified / sm.escalated) * 100).toFixed(1)}% passed both gates (${sm.verified}/${sm.escalated}); mean faithfulness ${sm.meanFaithfulnessVerified}` : null,
    },
    { name: "Refusal when no note matches", sub: 'Says "not found" instead of guessing', value: null as string | null },
    {
      name: "Cases closed by the agent",
      sub: "Must be zero",
      value: sm ? `${sm.casesClosedByAgent} of ${sm.totalCases} (decisions seen: ${sm.decisionsSeen.join(", ")})` : null,
    },
  ];
  return (
    <div>
      <h1 className="text-2xl font-bold">Model evaluation</h1>
      <p className="mt-2 max-w-xl text-sm text-slate-500">
        Rules, tabular model and graph model compared on the same chronological test split. Values in brackets are
        filled from the evaluation script; none are typed by hand.
      </p>

      <div className="mt-6 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
        <b>Read this first:</b> SAML-D is known to be easy to separate, so near-perfect scores are a property of the
        dataset, not proof the system would do this well on real bank data. An ablation without rule features gave the
        same scores, so rule leakage is not the cause.
      </div>

      <Card className="mt-6">
        <h2 className="text-lg font-bold">How the alert score is built</h2>
        <p className="mt-2 text-sm text-slate-700">
          The score is fixed in advance, not tuned. XGBoost is the main scorer. When both accounts of a transaction
          were seen in the training period, the score is 50% XGBoost rank plus 50% graph model rank. Otherwise it is
          the XGBoost rank alone. Ranks are computed within each split, so the score orders alerts in a batch; it is
          not a calibrated probability.
        </p>
      </Card>

      <Card className="mt-6 !p-0 overflow-hidden">
        <h2 className="px-6 pt-6 text-lg font-bold">Detection layers</h2>
        <table className="mt-4 w-full text-left text-sm">
          <thead className="bg-slate-100 text-xs font-semibold text-slate-600">
            <tr><th className="px-6 py-3">Layer</th><th>What it is</th><th>Precision@K</th><th>ROC-AUC</th><th className="px-6">Note</th></tr>
          </thead>
          <tbody>
            {LAYERS.map((l) => (
              <tr key={l.layer} className="border-t border-slate-100">
                <td className="px-6 py-3 font-semibold">{l.layer}</td>
                <td>{l.what}</td>
                <td><Val v={l.pAtK} /></td>
                <td className="font-mono">{l.auc}</td>
                <td className="px-6 text-slate-600">{l.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card className="!p-0 overflow-hidden">
          <h2 className="px-6 pt-6 text-lg font-bold">Rule engine</h2>
          <table className="mt-4 w-full text-left text-sm">
            <thead className="bg-slate-100 text-xs font-semibold text-slate-600">
              <tr><th className="px-6 py-3">Rule</th><th>Lift on test split</th><th className="px-6">In alert score</th></tr>
            </thead>
            <tbody>
              {RULES.map((r) => (
                <tr key={r.rule} className="border-t border-slate-100">
                  <td className="px-6 py-3 font-semibold">{r.rule}</td>
                  <td className="font-mono">{r.lift}</td>
                  <td className="px-6">{r.inScore}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-6 py-4 text-xs text-slate-500">
            The velocity rule was redesigned three times and still showed negative lift, so it is excluded from the
            score. Rule thresholds were tuned with the test split in view, which is stated as a limitation.
          </p>
        </Card>

        <Card>
          <h2 className="text-lg font-bold">Agent and retrieval checks</h2>
          <div className="mt-4 space-y-3">
            {checks.map((a) => (
              <div key={a.name} className="flex items-start justify-between gap-4 rounded-lg border border-slate-200 p-3">
                <div>
                  <div className="text-sm font-medium">{a.name}</div>
                  <div className="text-xs text-slate-500">{a.sub}</div>
                </div>
                <div className="max-w-[45%] text-right"><Val v={a.value} /></div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="mt-6">
        <h2 className="text-lg font-bold">Limitations to state with the results</h2>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm text-slate-700">
          {LIMITATIONS.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </Card>
    </div>
  );
}
