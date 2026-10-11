import { formatPercent } from "../format";
import type { ModelInfo } from "../types";
import type { PredictionState } from "../usePrediction";
import { useTweenedNumber } from "../useTweenedNumber";
import { RiskScale } from "./RiskScale";

const MODEL_NAMES: Record<string, string> = { logreg: "Logistic regression", lgbm: "LightGBM" };

export function DecisionPanel(props: { prediction: PredictionState; info: ModelInfo | null }) {
  const { result, loading, error } = props.prediction;
  const { info } = props;
  const probability = result?.probability_of_default ?? null;
  const threshold = result?.threshold ?? info?.policy.threshold ?? 0.15;
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

      <dl className="facts">
        {probability !== null && info && (
          <div>
            <dt>Compared with average</dt>
            <dd>{(probability / info.average_default_rate).toFixed(1)}× the typical risk</dd>
          </div>
        )}
        {info && (
          <div>
            <dt>Lending policy</dt>
            <dd>
              Approve when expected profit is positive: {formatPercent(info.policy.lgd, 0)} lost on a default,{" "}
              {formatPercent(info.policy.margin, 0)} earned on a repaid loan.
            </dd>
          </div>
        )}
      </dl>

      {info && (
        <footer className="panel-foot">
          <span>{MODEL_NAMES[info.model_type] ?? info.model_type}</span>
          <span>Test ROC-AUC {(info.test_roc_auc ?? info.validation_roc_auc).toFixed(3)}</span>
          <span>Trained {info.trained_at.slice(0, 10)}</span>
        </footer>
      )}
    </aside>
  );
}
