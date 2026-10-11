import { useEffect, useRef, useState, type MouseEvent } from "react";

import { formatMoney, formatPercent } from "../format";
import type { PortfolioResponse } from "../types";

// The SVG is drawn at its real on-screen width (1 unit = 1 pixel), so labels stay readable
// on a phone instead of shrinking with the whole picture.
const PAD = { left: 56, right: 16, top: 24, bottom: 40 };
const X_MAX = 0.6;

/**
 * Profit on the validation applicants for every cut-off from 1% to 60%: realized (what
 * happened, solid) and expected (from the model's probabilities, dashed). Hand-drawn SVG:
 * a chart this simple doesn't need a charting library.
 */
export function ProfitCurve(props: { data: PortfolioResponse; onPick: (threshold: number) => void }) {
  const { data } = props;
  const [hover, setHover] = useState<number | null>(null); // index into data.curve
  const boxRef = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(760);
  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const observer = new ResizeObserver(([entry]) => setW(Math.max(300, Math.round(entry.contentRect.width))));
    observer.observe(box);
    return () => observer.disconnect();
  }, []);
  const narrow = W < 560;
  const H = narrow ? 260 : 320;
  const flipAt = narrow ? 0.3 : 0.42; // past this cut-off, labels go on the left so they stay inside

  const profits = data.curve.flatMap((p) => [p.realized_profit, p.expected_profit]);
  const low = Math.min(...profits, data.approve_all.realized_profit, 0);
  const high = Math.max(...profits, data.approve_all.realized_profit);
  const step = niceStep((high - low) / 5);
  const yMin = Math.floor(low / step) * step;
  const yMax = Math.ceil((high * 1.05) / step) * step;
  const x = (threshold: number) => PAD.left + (threshold / X_MAX) * (W - PAD.left - PAD.right);
  const y = (profit: number) => PAD.top + ((yMax - profit) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);

  const path = (value: (p: (typeof data.curve)[number]) => number) =>
    data.curve.map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)},${y(value(p)).toFixed(1)}`).join(" ");
  const line = path((p) => p.realized_profit);
  const expectedLine = path((p) => p.expected_profit);
  const area = `${line} L${x(data.curve.at(-1)!.threshold)},${y(yMin)} L${x(data.curve[0].threshold)},${y(yMin)} Z`;
  const yTicks = Array.from({ length: Math.round((yMax - yMin) / step) + 1 }, (_, i) => yMin + i * step);
  const policyX = x(Math.min(data.policy.threshold, X_MAX));

  // Pointer position (in screen pixels) -> nearest cut-off on the curve.
  const nearest = (event: MouseEvent<SVGSVGElement>) => { // pointer and click events are both mouse events
    const box = event.currentTarget.getBoundingClientRect();
    const svgX = ((event.clientX - box.left) / box.width) * W;
    const threshold = ((svgX - PAD.left) / (W - PAD.left - PAD.right)) * X_MAX;
    const index = Math.round(threshold * 100) - 1; // curve points are 1%, 2%, ... 60%
    return Math.min(Math.max(index, 0), data.curve.length - 1);
  };
  const hovered = hover === null ? null : data.curve[hover];

  return (
    <div ref={boxRef} className="profit-curve-box">
    <svg
      className="profit-curve"
      viewBox={`0 0 ${W} ${H}`}
      width={W}
      height={H}
      role="img"
      aria-label={`Profit by cut-off. Most profitable at ${formatPercent(data.best_threshold, 0)}.`}
      onPointerMove={(event) => setHover(nearest(event))}
      onPointerLeave={() => setHover(null)}
      onClick={(event) => props.onPick(data.curve[nearest(event)].threshold)}
    >
      {yTicks.map((tick) => (
        <g key={tick}>
          <line className="grid-line" x1={PAD.left} x2={W - PAD.right} y1={y(tick)} y2={y(tick)} />
          <text className="axis-label" x={PAD.left - 10} y={y(tick) + 4} textAnchor="end">
            {formatMoney(tick)}
          </text>
        </g>
      ))}
      {(narrow ? [0, 0.2, 0.4, 0.6] : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).map((t) => (
        <text key={t} className="axis-label" x={x(t)} y={H - 14} textAnchor="middle">
          {formatPercent(t, 0)}
        </text>
      ))}

      {/* Reference: profit if everyone were approved */}
      <line className="ref-line" x1={PAD.left} x2={W - PAD.right} y1={y(data.approve_all.realized_profit)} y2={y(data.approve_all.realized_profit)} />
      {/* Below the line, at the right end: there the curve runs just above it, so the label stays clear */}
      <text className="ref-label" x={W - PAD.right} y={y(data.approve_all.realized_profit) + 16} textAnchor="end">
        approve everyone {formatMoney(data.approve_all.realized_profit)}
      </text>

      <path className="curve-area" d={area} />
      <path className="curve-expected" d={expectedLine} />
      <path className="curve-line" d={line} />

      {/* Formula cut-off vs. the best cut-off in hindsight: close together means the probabilities are honest */}
      <line className="best-line" x1={x(data.best_threshold)} x2={x(data.best_threshold)} y1={PAD.top} y2={H - PAD.bottom} />
      {/* Near the right edge the label goes on the left of its line, so it never runs off the chart */}
      <text
        className="best-label"
        x={x(data.best_threshold) + (data.best_threshold > flipAt ? -6 : 6)}
        y={PAD.top + 10}
        textAnchor={data.best_threshold > flipAt ? "end" : "start"}
      >
        most profitable here {formatPercent(data.best_threshold, 0)}
      </text>

      <g className="policy-marker" style={{ transform: `translateX(${policyX}px)` }}>
        <line x1={0} x2={0} y1={PAD.top} y2={H - PAD.bottom} />
        <circle cx={0} cy={y(data.policy.realized_profit)} r={6} />
      </g>

      {hovered && (
        <g className="hover">
          <line x1={x(hovered.threshold)} x2={x(hovered.threshold)} y1={PAD.top} y2={H - PAD.bottom} />
          <circle cx={x(hovered.threshold)} cy={y(hovered.realized_profit)} r={4} />
          <text x={x(hovered.threshold) + (hovered.threshold > flipAt ? -10 : 10)} y={y(hovered.realized_profit) - 12}
            textAnchor={hovered.threshold > flipAt ? "end" : "start"}>
            {formatPercent(hovered.threshold, 0)}: {formatMoney(hovered.realized_profit, true)}
            {!narrow && ` (expected ${formatMoney(hovered.expected_profit, true)}) · approve ${formatPercent(hovered.approval_rate, 0)}`}
          </text>
        </g>
      )}
    </svg>
    </div>
  );
}

/** A round tick step (1, 2, 2.5 or 5 times a power of ten) at least as big as `rough`. */
function niceStep(rough: number): number {
  const power = 10 ** Math.floor(Math.log10(rough));
  return [1, 2, 2.5, 5, 10].map((m) => m * power).find((step) => step >= rough)!;
}
