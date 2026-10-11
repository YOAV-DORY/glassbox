import { useEffect, useState } from "react";

import { fetchModelInfo } from "./api";
import { ApplicantForm } from "./components/ApplicantForm";
import { stagger } from "./components/controls";
import { DecisionPanel } from "./components/DecisionPanel";
import { EXAMPLES } from "./examples";
import { formatPercent } from "./format";
import type { ApplicantDraft, ModelInfo } from "./types";
import { usePrediction } from "./usePrediction";

const START = EXAMPLES[1]; // open on the borderline case: the most interesting one

export function App() {
  const [draft, setDraft] = useState<ApplicantDraft>(START.applicant);
  const [activeExample, setActiveExample] = useState<string | null>(START.id);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState(false);
  const prediction = usePrediction(draft);

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(() => setInfoError(true));
  }, []);

  // Typed so each key only accepts its own value type (e.g. own_car only takes a boolean).
  const setField = <K extends keyof ApplicantDraft>(key: K, value: ApplicantDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }));
    setActiveExample(null); // edited by hand: no longer one of the examples
  };

  const loadExample = (id: string) => {
    const example = EXAMPLES.find((e) => e.id === id);
    if (!example) return;
    setDraft(example.applicant);
    setActiveExample(id);
  };

  const result = prediction.result;

  return (
    <div className="app">
      <header className="topbar reveal" style={stagger(0)}>
        <a className="brand" href="/">
          <Logo />
          <span className="brand-name">Glassbox</span>
        </a>
        <span className="brand-tag">Credit decisions you can see inside</span>
        <span className={infoError ? "status is-down" : "status"}>
          <i />
          {infoError ? "API offline" : info ? "Model online" : "Connecting"}
          {info && <span className="status-version">· {info.model_version}</span>}
        </span>
      </header>

      <main className="layout">
        <section className="intro reveal" style={stagger(1)}>
          <p className="eyebrow">Decision desk</p>
          <h1>
            Assess an <em>applicant</em>.
          </h1>
          <p className="lede">
            Fill in the application and watch the decision update live. The model estimates the chance of default; the
            lending policy turns it into approve or decline.
          </p>
        </section>

        <div className="workspace">
          <ApplicantForm
            draft={draft}
            setField={setField}
            activeExample={activeExample}
            onExample={loadExample}
          />
          <div className="panel-wrap reveal" style={stagger(2)}>
            <DecisionPanel prediction={prediction} info={info} />
          </div>
        </div>
      </main>

      {/* Small screens: the panel scrolls away while editing, so keep the verdict pinned. */}
      {result && (
        <div className={`mini-result mini-result--${result.decision}`}>
          <span className="mini-figure">{formatPercent(result.probability_of_default)}</span>
          <span className="mini-label">chance of default</span>
          <span className="mini-decision">{result.decision === "approve" ? "Approve" : "Decline"}</span>
        </div>
      )}

      <footer className="colophon">
        Trained on the Home Credit Default Risk dataset (Kaggle). A portfolio project, not a lending product.
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="3" y="3" width="20" height="20" rx="2.5" fill="none" stroke="currentColor" strokeWidth="2.2" />
      <rect x="10" y="10" width="19" height="19" rx="2.5" className="logo-glass" strokeWidth="2.2" />
    </svg>
  );
}
