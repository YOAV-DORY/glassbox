import { useEffect, useRef, useState } from "react";

import { fetchRiskSurface } from "../api";
import { CountUp, KineticOnView, KineticText, Marquee, Reveal, WordCycler } from "../components/kinetic";
import { RiskLandscape } from "../components/RiskLandscape";
import { scrollToSection } from "../components/TopBar";
import { WireCube } from "../components/WireCube";
import { formatPercent } from "../format";
import { Link } from "../router";
import type { ModelInfo, RiskSurface } from "../types";

const CYCLE = [
  { text: "decision.", className: "word-decision" },
  { text: "approval.", className: "word-approve" },
  { text: "decline.", className: "word-decline" },
];

export function Landing({ info }: { info: ModelInfo | null }) {
  const [surface, setSurface] = useState<RiskSurface | null>(null);
  useEffect(() => {
    fetchRiskSurface().then(setSurface).catch(() => setSurface(null));
    // Deep links like /#landscape: the section only exists after React renders, so the
    // browser's own jump-to-anchor misses it. Do it ourselves.
    const target = window.location.hash && document.getElementById(window.location.hash.slice(1));
    if (target) target.scrollIntoView({ block: "start" });
  }, []);

  return (
    <main className="landing">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow load-in" style={{ animationDelay: "0ms" }}>
            Explainable credit risk
          </p>
          <h1 className="hero-title">
            <span className="hero-line">
              <KineticText text="See inside" delayMs={150} />
            </span>
            <span className="hero-line">
              <KineticText text="every credit" delayMs={450} />
            </span>
            <span className="hero-line">
              <WordCycler words={CYCLE} delayMs={850} />
            </span>
          </h1>
          <p className="hero-lede load-in" style={{ animationDelay: "1200ms" }}>
            Glassbox estimates how likely a borrower is to default, turns that into an approve-or-decline call
            with a clear lending policy, and shows its working.
          </p>
          <div className="hero-actions load-in" style={{ animationDelay: "1400ms" }}>
            <Link to="/assess" className="button button--primary">
              Assess an applicant <span className="button-arrow">→</span>
            </Link>
            <button type="button" className="button button--ghost" onClick={() => scrollToSection("how")}>
              How it works
            </button>
          </div>
        </div>

        <figure className="hero-visual load-in" style={{ animationDelay: "300ms" }}>
          <WireCube />
          <figcaption>
            <span>Fig. 1</span> The glass box. Each dot is an applicant; red ones defaulted. The blue plane is the
            model, scoring them.
          </figcaption>
        </figure>
      </section>

      <div className="marquees">
        <Marquee items={["Probability", "Policy", "Decision", "Reasons", "Fairness"]} />
        <Marquee items={["Calibrated", "Monotone", "Explainable", "Audited"]} direction={-1} />
      </div>

      <HowItWorks info={info} />

      {info && <Numbers info={info} />}

      <section id="landscape" className="landscape">
        <div className="landscape-copy">
          <Reveal>
            <p className="eyebrow">Fig. 2 · The risk landscape</p>
          </Reveal>
          <h2 className="section-heading">
            <KineticOnView text="The real model," stepMs={22} />{" "}
            <KineticOnView text="in three dimensions." className="heading-accent" stepMs={22} />
          </h2>
          <Reveal delayMs={160}>
            <p className="section-lede">
              Not an illustration. Each line is the model's predicted default rate as two credit-bureau scores
              change, with every other input held at a typical applicant. Where the surface rises above the dashed
              cut-off plane, it turns red: the policy declines.
            </p>
          </Reveal>
          <Reveal delayMs={240}>
            <ul className="landscape-notes">
              <li>
                <strong>Steps, not curves.</strong> A tree model learns thresholds, so risk changes in small jumps.
              </li>
              <li>
                <strong>Always downhill</strong> towards higher scores: the model is constrained so a better score
                can never raise risk.
              </li>
              <li>
                <strong>Drag to turn it.</strong>
              </li>
            </ul>
          </Reveal>
        </div>
        <Reveal className="landscape-stage" delayMs={120}>
          {surface ? <RiskLandscape surface={surface} /> : <p className="landscape-empty">Start the API to see the model.</p>}
        </Reveal>
      </section>

      <section className="finale">
        <Reveal>
          <p className="eyebrow">Ready?</p>
        </Reveal>
        <h2 className="finale-title">
          <KineticOnView text="Your turn." stepMs={60} />
        </h2>
        <Reveal delayMs={160}>
          <p className="section-lede">
            Pick an example applicant or type your own, and watch the decision change as you go.
          </p>
        </Reveal>
        <Reveal delayMs={240}>
          <Link to="/assess" className="button button--primary button--large">
            Assess an applicant <span className="button-arrow">→</span>
          </Link>
        </Reveal>
      </section>
    </main>
  );
}

const STEPS = [
  {
    title: "Score the risk",
    body: "A LightGBM model reads 15 fields from the application: loan size, income, employment, credit-bureau scores. It returns the probability that this borrower defaults.",
    note: "Gender and age are never inputs.",
  },
  {
    title: "Apply a lending policy",
    body: "Not a magic 50%. A loan that defaults loses about 60% of its value; a repaid one earns about 10%. Approve only when the expected profit is positive.",
    formula: true,
  },
  {
    title: "Show the reasons",
    body: "Every decision lists the factors that pushed the risk up or down, measured with SHAP: reason-code-style explanations of what the model did, not a compliant adverse-action notice.",
    note: "Contributions add up exactly to the score: nothing is hidden.",
  },
  {
    title: "Check fairness",
    body: "Gender and age stay out of the model, but other inputs can stand in for them. Approval rates, default rates and calibration are compared across groups.",
    note: "The audit flags under-30s: approved far less often, partly because the model overstates their risk.",
  },
];

/** Sticky big number on the left; steps scroll past on the right and light up in turn. */
function HowItWorks({ info }: { info: ModelInfo | null }) {
  const [active, setActive] = useState(0);
  const stepRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    // A step becomes active when it crosses a thin band in the middle of the screen.
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(Number((entry.target as HTMLElement).dataset.index));
        }
      },
      { rootMargin: "-45% 0px -45% 0px" },
    );
    stepRefs.current.forEach((step) => step && observer.observe(step));
    return () => observer.disconnect();
  }, []);

  const policy = info?.policy ?? { lgd: 0.6, margin: 0.1, threshold: 0.1 / 0.7 };

  return (
    <section id="how" className="how">
      <div className="how-sticky">
        <p className="eyebrow">How it works</p>
        <div className="how-number" aria-hidden="true">
          <KineticText key={active} text={String(active + 1).padStart(2, "0")} stepMs={70} />
        </div>
        <p className="how-current" key={`title-${active}`}>
          {STEPS[active].title}
        </p>
        <div className="how-progress" aria-hidden="true">
          {STEPS.map((_, i) => (
            <i key={i} className={i <= active ? "is-on" : undefined} />
          ))}
        </div>
      </div>

      <ol className="how-steps">
        {STEPS.map((step, i) => (
          <li
            key={step.title}
            ref={(element) => {
              stepRefs.current[i] = element;
            }}
            data-index={i}
            className={i === active ? "step is-active" : "step"}
          >
            <span className="step-index">{String(i + 1).padStart(2, "0")}</span>
            <h3 className="step-title">{step.title}</h3>
            <p className="step-body">{step.body}</p>
            {step.formula && (
              <p className="step-formula">
                approve if <var>p</var> &lt; {formatPercent(policy.margin, 0)} ÷ ({formatPercent(policy.margin, 0)} +{" "}
                {formatPercent(policy.lgd, 0)}) = <strong>{formatPercent(policy.threshold)}</strong>
              </p>
            )}
            {step.formula && (
              <Link to="/portfolio" className="step-link">
                Try other numbers in the portfolio simulator →
              </Link>
            )}
            {step.note && <p className="step-note">{step.note}</p>}
          </li>
        ))}
      </ol>
    </section>
  );
}

function Numbers({ info }: { info: ModelInfo }) {
  const items = [
    { value: info.train_rows, format: (v: number) => Math.round(v).toLocaleString("en-US"), label: "past loans the model learned from" },
    {
      value: info.test_roc_auc ?? info.validation_roc_auc,
      format: (v: number) => v.toFixed(3),
      label: info.test_roc_auc_ci
        ? `ROC-AUC on applicants it never saw (95% CI ${info.test_roc_auc_ci[0].toFixed(3)}–${info.test_roc_auc_ci[1].toFixed(3)})`
        : "ROC-AUC on applicants it never saw",
    },
    { value: info.policy.threshold, format: (v: number) => formatPercent(v), label: "policy cut-off, from loss vs. margin" },
    { value: info.average_default_rate, format: (v: number) => formatPercent(v), label: "of applicants default on average" },
  ];
  return (
    <section className="numbers">
      <Reveal className="numbers-head">
        <p className="eyebrow">The model in four numbers</p>
      </Reveal>
      {items.map((item, i) => (
        <Reveal key={item.label} className="number" delayMs={i * 90}>
          <CountUp value={item.value} format={item.format} />
          <span className="number-label">{item.label}</span>
        </Reveal>
      ))}
    </section>
  );
}
