import type { Reason } from "../types";

/** SHAP reason codes: the factors that pushed this applicant's risk up or down. */
export function Reasons(props: { up: Reason[]; down: Reason[]; decision: "approve" | "decline" }) {
  const { up, down, decision } = props;
  if (!up.length && !down.length) return null; // e.g. a model type without explanations

  // All bars share one scale, so lengths compare across both lists.
  const largest = Math.max(...[...up, ...down].map((r) => Math.abs(r.impact)));
  return (
    <section className="reasons">
      <h3 className="reasons-title">{decision === "decline" ? "Main factors behind this decline" : "Why this score"}</h3>
      <ReasonGroup title="Raising risk" reasons={up} largest={largest} direction="up" />
      <ReasonGroup title="Lowering risk" reasons={down} largest={largest} direction="down" />
      <p className="reasons-note">
        Bar length is each factor's SHAP contribution. Together, all contributions add up exactly to this score.
        A reason-code-style explanation, not a compliant adverse-action notice.
      </p>
    </section>
  );
}

function ReasonGroup(props: { title: string; reasons: Reason[]; largest: number; direction: "up" | "down" }) {
  if (!props.reasons.length) return null;
  return (
    <div className={`reason-group reason-group--${props.direction}`}>
      <p className="reason-group-title">{props.title}</p>
      <ul>
        {props.reasons.map((reason) => (
          // key={feature}: a factor that stays in the list keeps its row, so its bar animates.
          <li key={reason.feature} className="reason" title={`${reason.impact > 0 ? "+" : ""}${reason.impact.toFixed(2)} log-odds`}>
            <span className="reason-label">{reason.label}</span>
            <span className="reason-value">{reason.value}</span>
            <span className="reason-bar" aria-hidden="true">
              <i style={{ width: `${(Math.abs(reason.impact) / props.largest) * 100}%` }} />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
