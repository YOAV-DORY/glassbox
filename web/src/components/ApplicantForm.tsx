import type { ReactNode } from "react";

import { EXAMPLES } from "../examples";
import { formatPercent } from "../format";
import { EDUCATION_TYPES, INCOME_TYPES, type ApplicantDraft } from "../types";
import { AmountInput, Field, NullableSlider, Segmented, Select, stagger, Stepper } from "./controls";

type SetField = <K extends keyof ApplicantDraft>(key: K, value: ApplicantDraft[K]) => void;

const YES_NO = [
  { value: true, label: "Yes" },
  { value: false, label: "No" },
];
const formatScore = (value: number) => value.toFixed(2);
const formatYears = (value: number) => `${value.toFixed(1)} yrs`;

function Section(props: { index: string; title: string; note?: string; children: ReactNode }) {
  return (
    <fieldset className="section">
      <legend className="section-head">
        <span className="section-index">{props.index}</span>
        <span className="section-title">{props.title}</span>
      </legend>
      {props.note && <p className="section-note">{props.note}</p>}
      <div className="section-grid">{props.children}</div>
    </fieldset>
  );
}

export function ApplicantForm(props: {
  draft: ApplicantDraft;
  setField: SetField;
  activeExample: string | null;
  onExample: (id: string) => void;
}) {
  const { draft, setField } = props;

  return (
    <form className="form" onSubmit={(event) => event.preventDefault()}>
      <div className="examples reveal" style={stagger(2)}>
        <span className="examples-label">Start from an example</span>
        <div className="examples-row">
          {EXAMPLES.map((example, i) => (
            <button
              key={example.id}
              type="button"
              className={example.id === props.activeExample ? "example is-on" : "example"}
              onClick={() => props.onExample(example.id)}
            >
              <span className="example-index">{String.fromCharCode(65 + i)}</span>
              <span className="example-label">{example.label}</span>
              <span className="example-desc">{example.description}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="reveal" style={stagger(3)}>
        <Section index="01" title="Loan">
          <Field label="Loan amount" htmlFor="amt_credit">
            <AmountInput id="amt_credit" value={draft.amt_credit} onChange={(v) => setField("amt_credit", v)} />
          </Field>
          <Field label="Annuity" htmlFor="amt_annuity" hint="The regular repayment">
            <AmountInput id="amt_annuity" value={draft.amt_annuity} onChange={(v) => setField("amt_annuity", v)} />
          </Field>
          <Field label="Goods price" htmlFor="amt_goods_price" hint="What the loan pays for · optional">
            <AmountInput
              id="amt_goods_price"
              value={draft.amt_goods_price}
              placeholder="Not given"
              onChange={(v) => setField("amt_goods_price", v)}
            />
          </Field>
          <Field label="Contract">
            <Segmented
              name="contract_type"
              value={draft.contract_type}
              onChange={(v) => setField("contract_type", v)}
              options={[
                { value: "Cash loans", label: "Cash" },
                { value: "Revolving loans", label: "Revolving" },
              ]}
            />
          </Field>
        </Section>
      </div>

      <div className="reveal" style={stagger(4)}>
        <Section index="02" title="Income & work">
          <Field label="Income" htmlFor="amt_income_total">
            <AmountInput
              id="amt_income_total"
              value={draft.amt_income_total}
              onChange={(v) => setField("amt_income_total", v)}
            />
          </Field>
          <Field label="Income type" htmlFor="income_type">
            <Select
              id="income_type"
              options={INCOME_TYPES}
              value={draft.income_type}
              onChange={(v) => setField("income_type", v)}
            />
          </Field>
          <Field label="Education" htmlFor="education_type">
            <Select
              id="education_type"
              options={EDUCATION_TYPES}
              value={draft.education_type}
              onChange={(v) => setField("education_type", v)}
            />
          </Field>
          <Field label="Years employed" htmlFor="years_employed" wide>
            <NullableSlider
              id="years_employed"
              value={draft.years_employed}
              onChange={(v) => setField("years_employed", v)}
              min={0}
              max={40}
              step={0.5}
              format={formatYears}
              nullLabel="Not employed"
            />
          </Field>
        </Section>
      </div>

      <div className="reveal" style={stagger(5)}>
        <Section
          index="03"
          title="Credit scores"
          note="Scores from external credit bureaus, 0 = worst, 1 = best. Many applicants have no score."
        >
          {(["ext_source_1", "ext_source_2", "ext_source_3"] as const).map((key, i) => (
            <Field key={key} label={`External score ${i + 1}`} htmlFor={key} wide>
              <NullableSlider
                id={key}
                value={draft[key]}
                onChange={(v) => setField(key, v)}
                min={0}
                max={1}
                step={0.01}
                format={formatScore}
                nullLabel="No score"
              />
            </Field>
          ))}
        </Section>
      </div>

      <div className="reveal" style={stagger(6)}>
        <Section index="04" title="Household">
          <Field label="Owns a car">
            <Segmented name="own_car" options={YES_NO} value={draft.own_car} onChange={(v) => setField("own_car", v)} />
          </Field>
          <Field label="Owns a home">
            <Segmented
              name="own_realty"
              options={YES_NO}
              value={draft.own_realty}
              onChange={(v) => setField("own_realty", v)}
            />
          </Field>
          <Field label="Children" htmlFor="cnt_children">
            <Stepper
              id="cnt_children"
              value={draft.cnt_children}
              onChange={(v) => setField("cnt_children", v)}
              min={0}
              max={20}
            />
          </Field>
          <Field label="Region rating" hint="1 = best-rated region">
            <Segmented
              name="region_rating"
              value={draft.region_rating}
              onChange={(v) => setField("region_rating", v)}
              options={[
                { value: 1, label: "1" },
                { value: 2, label: "2" },
                { value: 3, label: "3" },
              ]}
            />
          </Field>
        </Section>
      </div>

      <DerivedRatios draft={draft} />
    </form>
  );
}

/**
 * The ratios the model computes from these inputs (see credit_risk/features.py).
 * Shown for transparency only; the API computes the real values.
 */
function DerivedRatios({ draft }: { draft: ApplicantDraft }) {
  const { amt_annuity, amt_credit, amt_income_total, amt_goods_price } = draft;
  const ratio = (a: number | null, b: number | null) => (a && b ? a / b : null);
  const items = [
    { label: "Payment to income", value: ratio(amt_annuity, amt_income_total), format: formatPercent },
    { label: "Credit to income", value: ratio(amt_credit, amt_income_total), format: (v: number) => `${v.toFixed(1)}×` },
    { label: "Loan to value", value: ratio(amt_credit, amt_goods_price), format: formatPercent },
  ];
  return (
    <div className="derived reveal" style={stagger(7)}>
      <span className="derived-label">What the model also sees</span>
      <dl className="derived-list">
        {items.map((item) => (
          <div key={item.label}>
            <dt>{item.label}</dt>
            <dd>{item.value === null ? "—" : item.format(item.value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
