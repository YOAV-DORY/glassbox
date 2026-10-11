import type { CSSProperties } from "react";

import { formatPercent } from "../format";
import { DEFAULT_POLICY, formulaThreshold } from "../policy";
import type { Policy } from "../types";

function PercentSlider(props: {
  id: string;
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  const { value, min, max } = props;
  return (
    <div className="policy-field">
      <div className="policy-field-head">
        <label htmlFor={props.id} className="field-label">
          {props.label}
        </label>
        <output htmlFor={props.id} className="policy-value">
          {formatPercent(value, value < 0.1 ? 1 : 0)}
        </output>
      </div>
      <input
        id={props.id}
        type="range"
        min={min}
        max={max}
        step={props.step}
        value={value}
        style={{ "--fill": `${((value - min) / (max - min)) * 100}%` } as CSSProperties}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
      <p className="field-hint">{props.hint}</p>
    </div>
  );
}

/** The two business inputs, the cut-off they imply, and an optional manual cut-off. */
export function PolicyControls({ policy, onChange }: { policy: Policy; onChange: (policy: Policy) => void }) {
  const formula = formulaThreshold(policy);
  const manual = policy.threshold_override !== null;

  return (
    <div className="policy-card">
      <p className="eyebrow">Lending policy</p>

      <PercentSlider
        id="lgd"
        label="Loss when a borrower defaults"
        hint="Share of the loan never recovered: the cost of a wrong approval."
        value={policy.lgd}
        min={0.1}
        max={1}
        step={0.01}
        onChange={(lgd) => onChange({ ...policy, lgd })}
      />
      <PercentSlider
        id="margin"
        label="Margin on a repaid loan"
        hint="Interest minus funding and servicing costs: what a wrong rejection gives up."
        value={policy.margin}
        min={0.01}
        max={0.4}
        step={0.005}
        onChange={(margin) => onChange({ ...policy, margin })}
      />

      <div className={manual ? "policy-formula is-off" : "policy-formula"}>
        <span className="policy-formula-label">Cut-off from the formula</span>
        <span className="policy-formula-value">{formatPercent(formula)}</span>
        <span className="policy-formula-math">
          approve if risk &lt; {formatPercent(policy.margin, 1)} ÷ ({formatPercent(policy.margin, 1)} +{" "}
          {formatPercent(policy.lgd, 0)})
        </span>
      </div>

      <label className="check policy-manual">
        <input
          type="checkbox"
          checked={manual}
          onChange={() => onChange({ ...policy, threshold_override: manual ? null : Math.round(formula * 100) / 100 })}
        />
        <span>Set the cut-off by hand (or click the chart)</span>
      </label>
      {manual && (
        <PercentSlider
          id="override"
          label="Manual cut-off"
          hint="Explore what happens away from the formula."
          value={policy.threshold_override!}
          min={0.01}
          max={0.6}
          step={0.01}
          onChange={(threshold_override) => onChange({ ...policy, threshold_override })}
        />
      )}

      <button type="button" className="policy-reset" onClick={() => onChange(DEFAULT_POLICY)}>
        Reset to 60% loss · 10% margin
      </button>
    </div>
  );
}
