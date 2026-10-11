import { formatAmount, formatPercent } from "../format";
import { activeThreshold } from "../policy";
import { Link } from "../router";
import type { ModelInfo, Policy, PredictResponse } from "../types";
import type { LiveState } from "../useLiveRequest";
import { useTweenedNumber } from "../useTweenedNumber";
import { Reasons } from "./Reasons";
import { RiskScale } from "./RiskScale";

const MODEL_NAMES: Record<string, string> = { logreg: "Logistic regression", lgbm: "LightGBM" };

export function DecisionPanel(props: { prediction: LiveState<PredictResponse>; info: ModelInfo | null; policy: Policy }) {
  const { result, loading, error } = props.prediction;
  const { info, policy } = props;
  const probability = result?.probability_of_default ?? null;
  const threshold = result?.threshold ?? activeThreshold(policy); // same formula as the API
  const shown = useTweenedNumber(probability ?? 0);
  const [whole, decimals] = (shown * 100).toFixed(1).split(".");

  return (
    <aside className="panel" aria-live="polite">
      <header className="panel-head">
        <span className="eyebrow">Probability of default</span>
        <span className={loading ? "live is-busy" : "live"}>{loading ? "Scoring" : "Live"}</span>
      </header>

      <p className={loading || error ? "figure is-dim" : "figure"}>
        {probability === null ? (
          <span className="figure-whole">—</span>
        ) : (
          <>
            <span className="figure-whole">{whole}</span>
            <span className="figure-decimals">.{decimals}</span>
            <span className="figure-unit">%</span>
          </>
        )}
      </p>

      {loading && !result && !error && (
        <p className="panel-wait">Scoring… The first request can take up to a minute while the server wakes up.</p>
      )}
      {error ? (
        <p className="panel-error">{error}</p>
      ) : (
        result && (
          // key={decision}: React remounts the stamp when the decision flips, replaying its animation.
          <div key={result.decision} className={`stamp stamp--${result.decision}`}>
            <span className="stamp-word">{result.decision === "approve" ? "Approve" : "Decline"}</span>
            <span className="stamp-note">
              {result.decision === "approve" ? "Under" : "Over"} the {formatPercent(threshold)} cut-off
            </span>
          </div>
        )
      )}

      <RiskScale probability={probability} threshold={threshold} average={info?.average_default_rate ?? null} />

      {probability !== null && info && result && (
        <dl className="panel-facts">
          <div>
            <dt>Versus the average applicant</dt>
            <dd>{(probability / info.average_default_rate).toFixed(1)}× the risk</dd>
          </div>
          <div>
            <dt>Expected profit if approved</dt>
            <dd className={result.expected_profit >= 0 ? "is-gain" : "is-loss"}>
              {result.expected_profit >= 0 ? "+" : "−"}
              {formatAmount(Math.abs(result.expected_profit))}
            </dd>
          </div>
        </dl>
      )}

      {result && !error && <Reasons up={result.reasons_up} down={result.reasons_down} decision={result.decision} />}

      {info && (
        <footer className="panel-foot">
          <span>{MODEL_NAMES[info.model_type] ?? info.model_type}</span>
          <span>Test ROC-AUC {(info.test_roc_auc ?? info.validation_roc_auc).toFixed(3)}</span>
          <Link to="/portfolio" className="panel-policy">
            {policy.threshold_override !== null
              ? `Manual cut-off ${formatPercent(policy.threshold_override)}`
              : `Policy: ${formatPercent(policy.lgd, 0)} loss, ${formatPercent(policy.margin, 0)} margin`}{" "}
            · change →
          </Link>
        </footer>
      )}
    </aside>
  );
}
