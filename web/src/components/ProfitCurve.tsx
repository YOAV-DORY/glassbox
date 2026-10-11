import { useState, type MouseEvent } from "react";

import { formatMoney, formatPercent } from "../format";
import type { PortfolioResponse } from "../types";

// Drawing area inside the SVG's coordinate system (it scales to the container's width).
const W = 760;
const H = 320;
const PAD = { left: 64, right: 24, top: 24, bottom: 40 };
const X_MAX = 0.6;

/**
 * Profit on the test applicants for every cut-off from 1% to 60%. Hand-drawn SVG:
 * a chart this simple doesn't need a charting library.
 */
export function ProfitCurve(props: { data: PortfolioResponse; onPick: (threshold: number) => void }) {
  const { data } = props;
  const [hover, setHover] = useState<number | null>(null); // index into data.curve

  const profits = data.curve.map((p) => p.profit);
  const low = Math.min(...profits, data.approve_all.profit, 0);
  const high = Math.max(...profits, data.approve_all.profit);
  const step = niceStep((high - low) / 5);
  const yMin = Math.floor(low / step) * step;
  const yMax = Math.ceil((high * 1.05) / step) * step;
  const x = (threshold: number) => PAD.left + (threshold / X_MAX) * (W - PAD.left - PAD.right);
  const y = (profit: number) => PAD.top + ((yMax - profit) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);

  const line = data.curve.map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)},${y(p.profit).toFixed(1)}`).join(" ");
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
    <svg
      className="profit-curve"
      viewBox={`0 0 ${W} ${H}`}
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
      {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((t) => (
        <text key={t} className="axis-label" x={x(t)} y={H - 14} textAnchor="middle">
          {formatPercent(t, 0)}
        </text>
      ))}

      {/* Reference: profit if everyone were approved */}
      <line className="ref-line" x1={PAD.left} x2={W - PAD.right} y1={y(data.approve_all.profit)} y2={y(data.approve_all.profit)} />
      {/* Below the line, at the right end: there the curve runs just above it, so the label stays clear */}
      <text className="ref-label" x={W - PAD.right} y={y(data.approve_all.profit) + 16} textAnchor="end">
        approve everyone {formatMoney(data.approve_all.profit)}
      </text>

      <path className="curve-area" d={area} />
      <path className="curve-line" d={line} />

      {/* Formula cut-off vs. the best cut-off in hindsight: close together means the probabilities are honest */}
      <line className="best-line" x1={x(data.best_threshold)} x2={x(data.best_threshold)} y1={PAD.top} y2={H - PAD.bottom} />
      <text className="best-label" x={x(data.best_threshold) + 6} y={PAD.top + 10}>
        best in hindsight {formatPercent(data.best_threshold, 0)}
      </text>

      <g className="policy-marker" style={{ transform: `translateX(${policyX}px)` }}>
        <line x1={0} x2={0} y1={PAD.top} y2={H - PAD.bottom} />
        <circle cx={0} cy={y(data.policy.profit)} r={6} />
      </g>

      {hovered && (
        <g className="hover">
          <line x1={x(hovered.threshold)} x2={x(hovered.threshold)} y1={PAD.top} y2={H - PAD.bottom} />
          <circle cx={x(hovered.threshold)} cy={y(hovered.profit)} r={4} />
          <text x={x(hovered.threshold) + (hovered.threshold > 0.42 ? -10 : 10)} y={y(hovered.profit) - 12}
            textAnchor={hovered.threshold > 0.42 ? "end" : "start"}>
            {formatPercent(hovered.threshold, 0)}: {formatMoney(hovered.profit, true)} · approve {formatPercent(hovered.approval_rate, 0)}
          </text>
        </g>
      )}
    </svg>
  );
}

/** A round tick step (1, 2, 2.5 or 5 times a power of ten) at least as big as `rough`. */
function niceStep(rough: number): number {
  const power = 10 ** Math.floor(Math.log10(rough));
  return [1, 2, 2.5, 5, 10].map((m) => m * power).find((step) => step >= rough)!;
}
