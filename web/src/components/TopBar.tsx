import { Link } from "../router";
import type { ModelInfo } from "../types";

export function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function TopBar(props: { page: "landing" | "assess"; info: ModelInfo | null; offline: boolean }) {
  const { page, info, offline } = props;
  return (
    <header className="topbar">
      <Link to="/" className="brand">
        <Logo />
        <span className="brand-name">Glassbox</span>
      </Link>

      <nav className="topnav">
        {page === "landing" ? (
          <>
            <button type="button" className="topnav-link" onClick={() => scrollToSection("how")}>
              How it works
            </button>
            <button type="button" className="topnav-link" onClick={() => scrollToSection("landscape")}>
              Risk landscape
            </button>
            <Link to="/assess" className="topnav-cta">
              <span>
                Assess<span className="cta-long"> an applicant</span>
              </span>
              <span aria-hidden="true">→</span>
            </Link>
          </>
        ) : (
          <>
            <Link to="/" className="topnav-link">
              <span aria-hidden="true">←</span> Overview
            </Link>
            <span className={offline ? "status is-down" : "status"}>
              <i />
              {offline ? "API offline" : info ? "Model online" : "Connecting"}
              {info && <span className="status-version">· {info.model_version}</span>}
            </span>
          </>
        )}
      </nav>
    </header>
  );
}

export function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="3" y="3" width="20" height="20" rx="2.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
      <rect x="10" y="10" width="19" height="19" rx="2.5" className="logo-glass" strokeWidth="2.2" />
    </svg>
  );
}
