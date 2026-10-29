import type { Severity } from "../data/types";

const SEV: Record<Severity, string> = {
  Critical: "bg-red-100 text-red-700",
  High: "bg-amber-100 text-amber-800",
  Medium: "bg-slate-200 text-slate-700",
};

const STATUS: Record<string, string> = {
  "Summary ready": "bg-blue-100 text-blue-800",
  "Needs more evidence": "bg-slate-200 text-slate-700",
  "Summary unverified": "bg-amber-100 text-amber-800",
  "Awaiting decision": "bg-blue-100 text-blue-800",
  Decided: "bg-green-100 text-green-800",
};

export function SeverityBadge({ value }: { value: Severity }) {
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${SEV[value]}`}>{value}</span>;
}

export function StatusBadge({ value }: { value: string }) {
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS[value] ?? "bg-slate-200 text-slate-700"}`}>{value}</span>;
}

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`rounded-xl border border-slate-200 bg-white p-5 ${className}`}>{children}</section>;
}

export function Chip({ children }: { children: React.ReactNode }) {
  return <span className="rounded bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-700">{children}</span>;
}
