import { useRef, type CSSProperties, type ReactNode } from "react";

import { formatAmount } from "../format";

/** Order of an element in the page-load animation (styles.css reads --i to delay it). */
export const stagger = (i: number) => ({ "--i": i }) as CSSProperties;

export function Field(props: { label: string; htmlFor?: string; hint?: string; wide?: boolean; children: ReactNode }) {
  return (
    <div className={props.wide ? "field field--wide" : "field"}>
      <label className="field-label" htmlFor={props.htmlFor}>
        {props.label}
      </label>
      {props.children}
      {props.hint && <p className="field-hint">{props.hint}</p>}
    </div>
  );
}

/** Text input that shows thousands separators (450,000) but reports a plain number. */
export function AmountInput(props: {
  id: string;
  value: number | null;
  onChange: (value: number | null) => void;
  placeholder?: string;
}) {
  return (
    <input
      id={props.id}
      className="input input--amount"
      inputMode="numeric"
      autoComplete="off"
      placeholder={props.placeholder}
      value={props.value === null ? "" : formatAmount(props.value)}
      onChange={(event) => {
        const digits = event.target.value.replace(/\D/g, "").slice(0, 10);
        props.onChange(digits ? Number(digits) : null);
      }}
    />
  );
}

/** Slider with a checkbox for "no value" (e.g. no credit score, not employed). */
export function NullableSlider(props: {
  id: string;
  value: number | null;
  onChange: (value: number | null) => void;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  nullLabel: string;
}) {
  const { value, min, max } = props;
  // Remember the last real value so un-ticking the checkbox restores it.
  const lastValue = useRef(value ?? (min + max) / 2);
  const shown = value ?? lastValue.current;
  const fill = `${((shown - min) / (max - min)) * 100}%`;

  const setValue = (next: number) => {
    lastValue.current = next;
    props.onChange(next);
  };

  return (
    <div className={value === null ? "slider is-null" : "slider"}>
      <input
        id={props.id}
        type="range"
        min={min}
        max={max}
        step={props.step}
        value={shown}
        disabled={value === null}
        style={{ "--fill": fill } as CSSProperties}
        onChange={(event) => setValue(Number(event.target.value))}
      />
      <output className="slider-value" htmlFor={props.id}>
        {value === null ? "—" : props.format(value)}
      </output>
      <label className="check">
        <input
          type="checkbox"
          checked={value === null}
          onChange={() => props.onChange(value === null ? lastValue.current : null)}
        />
        <span>{props.nullLabel}</span>
      </label>
    </div>
  );
}

/** A row of mutually exclusive options (styled radio buttons, so keyboard and screen readers work). */
export function Segmented<T extends string | number | boolean>(props: {
  name: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {props.options.map((option) => (
        <label key={String(option.value)} className={option.value === props.value ? "is-on" : undefined}>
          <input
            type="radio"
            name={props.name}
            checked={option.value === props.value}
            onChange={() => props.onChange(option.value)}
          />
          <span>{option.label}</span>
        </label>
      ))}
    </div>
  );
}

export function Select<T extends string>(props: {
  id: string;
  options: readonly T[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <select
      id={props.id}
      className="input input--select"
      value={props.value}
      onChange={(event) => props.onChange(event.target.value as T)}
    >
      {props.options.map((option) => (
        <option key={option}>{option}</option>
      ))}
    </select>
  );
}

export function Stepper(props: { id: string; value: number; onChange: (value: number) => void; min: number; max: number }) {
  const { value, min, max, onChange } = props;
  return (
    <div className="stepper">
      <button type="button" aria-label="Decrease" disabled={value <= min} onClick={() => onChange(value - 1)}>
        −
      </button>
      <output id={props.id}>{value}</output>
      <button type="button" aria-label="Increase" disabled={value >= max} onClick={() => onChange(value + 1)}>
        +
      </button>
    </div>
  );
}
