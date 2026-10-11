import { simulatePortfolio } from "../api";
import { KineticText } from "../components/kinetic";
import { PolicyControls } from "../components/PolicyControls";
import { ProfitCurve } from "../components/ProfitCurve";
import { formatMoney, formatPercent } from "../format";
import { Link } from "../router";
import type { CalibrationBin, FairnessGroup, Policy, PortfolioOutcome, PortfolioResponse } from "../types";
import { useLiveRequest } from "../useLiveRequest";

const ASSUMPTIONS = [
  "One period: no interest timing, prepayment or partial recoveries.",
  "The same loss and margin for every loan; the whole amount is at risk.",
  "Amounts are in the dataset's currency, which Kaggle does not name.",
];

export function Portfolio({ policy, setPolicy }: { policy: Policy; setPolicy: (policy: Policy) => void }) {
  const portfolio = useLiveRequest<PortfolioResponse>((signal) => simulatePortfolio(policy, signal), [policy]);
  const data = portfolio.result;

  return (
    <main className="portfolio">
      <section className="intro">
        <p className="eyebrow load-in">Portfolio simulator</p>
        <h1>
          <KineticText text="Set the" delayMs={100} /> <KineticText text="policy." delayMs={350} className="intro-accent" />
        </h1>
        <p className="lede load-in" style={{ animationDelay: "500ms" }}>
          The cut-off isn't a modelling choice, it's a business one. Say what a default costs and what a good loan
          earns, and see what that policy would have done to{" "}
          {data ? data.applicants.toLocaleString("en-US") : "the"} validation applicants, using what really
          happened to them. The test set is kept for the headline model metrics and never used here.
        </p>
      </section>

      <div className="sim">
        <aside className="sim-controls load-in" style={{ animationDelay: "300ms" }}>
          <PolicyControls policy={policy} onChange={setPolicy} />
          <Link to="/assess" className="sim-to-desk">
            Try this policy on one applicant →
          </Link>
        </aside>

        <div className={portfolio.loading ? "sim-results is-busy" : "sim-results"}>
          {portfolio.error && <p className="panel-error">{portfolio.error}</p>}
          {!data && !portfolio.error && (
            <p className="panel-wait">Replaying the policy… The first request can take up to a minute while the server wakes up.</p>
          )}
          <p className="bias-note load-in" style={{ animationDelay: "350ms" }}>
            <strong>Read with care:</strong> outcomes exist only for loans the lender approved in the past. Declining
            one of them here is a what-if on an approved loan, so everything on the decline side (losses avoided, good
            loans given up) is an assumption, not a measurement of how rejected applicants would have behaved.
          </p>
          {data && (
            <>
              <Kpis data={data} />
              <section className="card chart-card load-in" style={{ animationDelay: "450ms" }}>
                <header className="card-head">
                  <h2 className="card-title">Profit by cut-off</h2>
                  <p className="card-note">
                    Solid: realized profit (what happened). Dashed: expected profit (from the model's probabilities
                    alone). Blue: your cut-off. Red: the most profitable cut-off on these applicants. Hover to read
                    values, click to set a manual cut-off.
                  </p>
                </header>
                <ProfitCurve data={data} onPick={(t) => setPolicy({ ...policy, threshold_override: t })} />
              </section>
              <Comparison data={data} />
              <Calibration bins={data.calibration} threshold={data.policy.threshold} />
              <Fairness data={data} />
            </>
          )}
          <section className="card assumptions load-in" style={{ animationDelay: "600ms" }}>
            <h2 className="card-title">Simplifications</h2>
            <ul>
              {ASSUMPTIONS.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </main>
  );
}

function Kpis({ data }: { data: PortfolioResponse }) {
  const { policy, approve_all: all } = data;
  const uplift = (policy.realized_profit - all.realized_profit) / Math.abs(all.realized_profit);
  const tiles = [
    {
      label: "Approved",
      value: formatPercent(policy.approval_rate),
      note: `${policy.approved.toLocaleString("en-US")} of ${data.applicants.toLocaleString("en-US")} applicants`,
    },
    {
      label: "Default among approved",
      value: formatPercent(policy.bad_rate),
      note: `vs ${formatPercent(all.bad_rate)} if everyone is approved`,
    },
    {
      label: "Profit",
      value: formatMoney(policy.realized_profit, true),
      note: `expected ${formatMoney(policy.expected_profit, true)} · ${uplift >= 0 ? "+" : ""}${formatPercent(uplift, 0)} vs approving everyone`,
      tone: policy.realized_profit >= all.realized_profit ? "gain" : "loss",
    },
    {
      label: "Losses on approved loans",
      value: formatMoney(policy.realized_loss),
      note: `expected ${formatMoney(policy.expected_loss)}`,
    },
  ];
  return (
    <div className="kpis">
      {tiles.map((tile, i) => (
        <div
          key={tile.label}
          className={`kpi load-in ${tile.tone ? `kpi--${tile.tone}` : ""}`}
          style={{ animationDelay: `${350 + i * 70}ms` }}
        >
          <span className="kpi-label">{tile.label}</span>
          <span className="kpi-value">{tile.value}</span>
          <span className="kpi-note">{tile.note}</span>
        </div>
      ))}
    </div>
  );
}

function Comparison({ data }: { data: PortfolioResponse }) {
  const rows: [string, PortfolioOutcome][] = [
    ["Your policy", data.policy],
    ["Approve everyone", data.approve_all],
    ["50% cut-off", data.cutoff_50],
  ];
  return (
    <section className="card load-in" style={{ animationDelay: "550ms" }}>
      <h2 className="card-title">Against the naive rules</h2>
      <table className="compare">
        <thead>
          <tr>
            <th>Rule</th>
            <th>Cut-off</th>
            <th>Approved</th>
            <th>Default among approved</th>
            <th>Expected profit</th>
            <th>Realized profit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, o]) => (
            <tr key={name} className={name === "Your policy" ? "is-yours" : undefined}>
              <td>{name}</td>
              <td>{o.threshold > 1 ? "none" : formatPercent(o.threshold)}</td>
              <td>{formatPercent(o.approval_rate)}</td>
              <td>{formatPercent(o.bad_rate, 2)}</td>
              <td>{formatMoney(o.expected_profit, true)}</td>
              <td>{formatMoney(o.realized_profit, true)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

/** Deciles of predicted risk: is the model honest where the cut-off actually sits? */
function Calibration({ bins, threshold }: { bins: CalibrationBin[]; threshold: number }) {
  return (
    <section className="card load-in" style={{ animationDelay: "600ms" }}>
      <h2 className="card-title">Is the model honest near the cut-off?</h2>
      <p className="card-note">
        Applicants sorted by predicted risk into ten equal groups. The cut-off formula only works if predicted and
        actual default rates agree, especially in the highlighted group that contains your cut-off. ± is one
        standard error of the actual rate.
      </p>
      <table className="compare">
        <thead>
          <tr>
            <th>Decile</th>
            <th>Predictions</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Actual − predicted</th>
          </tr>
        </thead>
        <tbody>
          {bins.map((bin) => {
            const containsCutoff = bin.low <= threshold && threshold <= bin.high;
            return (
              <tr key={bin.decile} className={containsCutoff ? "is-yours" : undefined}>
                <td>{bin.decile}</td>
                <td>
                  {formatPercent(bin.low)} – {formatPercent(bin.high)}
                </td>
                <td>{formatPercent(bin.mean_predicted)}</td>
                <td>
                  {formatPercent(bin.actual_rate)} <span className="se">± {formatPercent(bin.actual_se)}</span>
                </td>
                <td>
                  {bin.actual_rate >= bin.mean_predicted ? "+" : "−"}
                  {(Math.abs(bin.actual_rate - bin.mean_predicted) * 100).toFixed(1)} pts
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

/** Approval and default rates by gender and age band at the current cut-off. */
function Fairness({ data }: { data: PortfolioResponse }) {
  const table = (title: string, groups: FairnessGroup[], names: Record<string, string>) => (
    <table className="compare">
      <thead>
        <tr>
          <th>{title}</th>
          <th>Applicants</th>
          <th>Actual default</th>
          <th>Predicted</th>
          <th>Approved</th>
          <th>Approval ratio</th>
          <th>Repayers declined</th>
        </tr>
      </thead>
      <tbody>
        {groups.map((g) => (
          <tr key={g.group}>
            <td>{names[g.group] ?? g.group}</td>
            <td>{g.n.toLocaleString("en-US")}</td>
            <td>{formatPercent(g.default_rate)}</td>
            <td>{formatPercent(g.mean_predicted)}</td>
            <td>{formatPercent(g.approval_rate)}</td>
            <td className={g.approval_ratio < 0.8 ? "flag" : undefined}>{g.approval_ratio.toFixed(2)}</td>
            <td>{formatPercent(g.repayers_declined)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  return (
    <section className="card load-in" style={{ animationDelay: "650ms" }}>
      <h2 className="card-title">Fairness check at this cut-off</h2>
      <p className="card-note">
        Gender and age are not model inputs; they're kept aside only for this check. Approval ratio is each group's
        approval rate relative to the most-approved group: below 0.8 (red) is the common "four-fifths" warning sign,
        a rule of thumb, not a legal test. Compare actual and predicted default rates to see whether a gap comes from
        different risk or from the model misjudging a group.
      </p>
      {table("Gender", data.fairness.gender, { F: "Women", M: "Men" })}
      {table("Age", data.fairness.age_band, { "<30": "Under 30", "60+": "60 and over" })}
      <p className="card-note fairness-proxy">
        Age proxy: {formatPercent(data.fairness.proxy.share_60plus_not_employed, 0)} of applicants aged 60+ carry the
        model's "not employed" flag, against {formatPercent(data.fairness.proxy.share_under_45_not_employed, 1)} of
        those under 45. Leaving age out of the model doesn't remove it.
      </p>
    </section>
  );
}
