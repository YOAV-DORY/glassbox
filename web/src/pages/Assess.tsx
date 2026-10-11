import { useRef, useState } from "react";

import { predict, validateDraft } from "../api";
import { ApplicantForm } from "../components/ApplicantForm";
import { stagger } from "../components/controls";
import { DecisionPanel } from "../components/DecisionPanel";
import { KineticText } from "../components/kinetic";
import { EXAMPLES } from "../examples";
import { formatPercent } from "../format";
import { useOnScreen } from "../hooks";
import type { ApplicantDraft, ModelInfo, Policy, PredictResponse } from "../types";
import { useLiveRequest } from "../useLiveRequest";

const START = EXAMPLES[1]; // open on the borderline case: the most interesting one

const HOW_TO = [
  "Pick an example, or type over any value.",
  "Change one thing at a time to see what moves the risk.",
];

export function Assess({ info, policy }: { info: ModelInfo | null; policy: Policy }) {
  const [draft, setDraft] = useState<ApplicantDraft>(START.applicant);
  const [activeExample, setActiveExample] = useState<string | null>(START.id);
  // Re-score whenever the applicant or the lending policy changes.
  const prediction = useLiveRequest<PredictResponse>((signal) => {
    const applicant = validateDraft(draft);
    return typeof applicant === "string" ? applicant : predict(applicant, policy, signal);
  }, [draft, policy]);

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
  const panelRef = useRef<HTMLDivElement>(null);
  const panelVisible = useOnScreen(panelRef);

  return (
    <main className="assess">
      <div className="workspace">
        {/* The intro sits in the form column, so the decision panel starts at the top right. */}
        <div className="desk-main">
          <section className="intro">
            <p className="eyebrow load-in">Decision desk</p>
            <h1>
              <KineticText text="Assess an" delayMs={100} />{" "}
              <KineticText text="applicant." delayMs={400} className="intro-accent" />
            </h1>
            <p className="lede load-in" style={{ animationDelay: "500ms" }}>
              Fill in the application and the decision panel updates as you go: the model estimates the chance of
              default, and the lending policy turns it into approve or decline.
            </p>
            <ol className="how-to load-in" style={{ animationDelay: "650ms" }}>
              {HOW_TO.map((tip, i) => (
                <li key={tip}>
                  <span>{i + 1}</span>
                  {tip}
                </li>
              ))}
            </ol>
          </section>
          <ApplicantForm draft={draft} setField={setField} activeExample={activeExample} onExample={loadExample} />
        </div>
        <div ref={panelRef} className="panel-wrap load-in" style={stagger(4)}>
          <DecisionPanel prediction={prediction} info={info} policy={policy} />
        </div>
      </div>

      {/* Small screens: the panel scrolls away while editing, so pin the verdict, but only
          while the panel itself is off screen (otherwise the bar would cover it). */}
      {result && !panelVisible && (
        <div className={`mini-result mini-result--${result.decision}`}>
          <span className="mini-figure">{formatPercent(result.probability_of_default)}</span>
          <span className="mini-label">chance of default</span>
          <span className="mini-decision">{result.decision === "approve" ? "Approve" : "Decline"}</span>
        </div>
      )}
    </main>
  );
}
