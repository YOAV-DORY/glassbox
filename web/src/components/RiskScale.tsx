import { formatPercent } from "../format";

// Most applicants score well under 40%, so the ruler stops there; higher values pin to the end.
const SCALE_MAX = 0.4;
const TICKS = Array.from({ length: 41 }, (_, i) => i); // one tick per percentage point

const position = (p: number) => `${(Math.min(Math.max(p, 0), SCALE_MAX) / SCALE_MAX) * 100}%`;

/** A ruler from 0% to 40%: approve zone, policy cut-off, portfolio average and this applicant. */
export function RiskScale(props: { probability: number | null; threshold: number; average: number | null }) {
  const { probability, threshold, average } = props;
  return (
    <div className="scale">
      <div
        className="scale-ruler"
        role="img"
        aria-label={
          probability === null
            ? "Risk scale"
            : `Risk ${formatPercent(probability)} against a cut-off of ${formatPercent(threshold)}`
        }
      >
        <div className="scale-zone scale-zone--approve" style={{ width: position(threshold) }} />
        <div className="scale-zone scale-zone--decline" style={{ left: position(threshold) }} />
        {TICKS.map((i) => (
          <span
            key={i}
            className={i % 10 === 0 ? "tick tick--major" : i % 5 === 0 ? "tick tick--mid" : "tick"}
            style={{ left: `${(i / 40) * 100}%` }}
          />
        ))}
        <div className="scale-cutoff" style={{ left: position(threshold) }}>
          <span>cut-off {formatPercent(threshold)}</span>
        </div>
        {average !== null && <div className="scale-average" style={{ left: position(average) }} />}
        {probability !== null && <div className="scale-needle" style={{ left: position(probability) }} />}
      </div>
      <div className="scale-axis">
        {[0, 10, 20, 30, 40].map((v) => (
          <span key={v} style={{ left: `${(v / 40) * 100}%` }}>
            {v === 40 ? "40%+" : `${v}%`}
          </span>
        ))}
      </div>
      <div className="scale-legend">
        <span>
          <i className="legend-needle" /> this applicant
        </span>
        {average !== null && (
          <span>
            <i className="legend-average" /> average applicant {formatPercent(average)}
          </span>
        )}
      </div>
    </div>
  );
}
