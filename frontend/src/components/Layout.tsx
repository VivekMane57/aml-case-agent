import { NavLink, Outlet, useLocation } from "react-router-dom";

function NavItem({ to, label, active }: { to: string; label: string; active?: boolean }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
          isActive || active ? "bg-[#1e3a6e] text-white" : "text-slate-300 hover:text-white"
        }`
      }
    >
      {label}
    </NavLink>
  );
}

export default function Layout() {
  const { pathname } = useLocation();
  const isEval = pathname.startsWith("/evaluation");
  return (
    <div className="min-h-screen bg-[#f3f5f9] text-slate-900">
      <header className="flex items-center justify-between bg-[#0f2447] px-6 py-3">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-white">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7fb0ff" strokeWidth="2" strokeLinecap="round">
              <circle cx="5" cy="12" r="2.5" /><circle cx="18" cy="5" r="2.5" /><circle cx="18" cy="19" r="2.5" />
              <path d="M7.2 11l8.6-4.7M7.2 13l8.6 4.7" />
            </svg>
            <span className="font-semibold">AML Case Agent</span>
          </div>
          <nav className="flex gap-1">
            <NavItem to="/" label="Alerts" />
            <NavItem to="/case/AL-0071" label="Case" />
            <NavItem to="/evaluation" label="Evaluation" />
          </nav>
        </div>
        <div className="flex items-center gap-2 text-xs font-medium">
          <span className="rounded-full bg-amber-100 px-3 py-1 text-amber-900">Synthetic data (SAML-D)</span>
          <span className="rounded-full bg-blue-100 px-3 py-1 text-[#0f2447]">
            {isEval ? "Admin" : "Analyst"} · V. Mane
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-[1100px] px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
