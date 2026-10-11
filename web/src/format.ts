export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatAmount(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumSignificantDigits: 3 });

/** Large money amounts: 2,070,000,000 -> "2.07B", 983,670,000 -> "984M". Optionally signed. */
export function formatMoney(value: number, signed = false): string {
  const text = compact.format(Math.abs(value));
  if (!signed) return value < 0 ? `−${text}` : text;
  return `${value < 0 ? "−" : "+"}${text}`;
}
