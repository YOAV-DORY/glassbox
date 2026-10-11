import { simulatePortfolio } from "../api";
import { KineticText } from "../components/kinetic";
import { PolicyControls } from "../components/PolicyControls";
import { ProfitCurve } from "../components/ProfitCurve";
import { formatMoney, formatPercent } from "../format";
import { Link } from "../router";
import type { Policy, PortfolioOutcome, PortfolioResponse } from "../types";
import { useLiveRequest } from "../useLiveRequest";

const ASSUMPTIONS = [
  "One period: no interest timing, prepayment or partial recoveries.",
  "The same loss and margin for every loan; the whole amount is at risk.",
  "The test applicants are loans Home Credit had already approved, not everyone who applied.",
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
          earns, and see what that policy would have done to {data ? data.applicants.toLocaleString("en-US") : "the"}{" "}
          test applicants the model never saw, using what really happened to them.
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
          {data && (
            <>
              <Kpis data={data} />
              <section className="card chart-card load-in" style={{ animationDelay: "450ms" }}>
                <header className="card-head">
                  <h2 className="card-title">Profit by cut-off</h2>
                  <p className="card-note">
                    Blue: your cut-off. Dashed red: the most profitable cut-off in hindsight. When the probabilities are
                    honest, the formula lands next to it. Hover to read values, click to set a manual cut-off.
                  </p>
                </header>
                <ProfitCurve data={data} onPick={(t) => setPolicy({ ...policy, threshold_override: t })} />
              </section>
              <Comparison data={data} />
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
  const uplift = (policy.profit - all.profit) / Math.abs(all.profit);
  const tiles = [
    { label: "Approved", value: formatPercent(policy.approval_rate), note: `${policy.approved.toLocaleString("en-US")} of ${data.applicants.toLocaleString("en-US")} applicants` },
    { label: "Default among approved", value: formatPercent(policy.bad_rate), note: `vs ${formatPercent(all.bad_rate)} if everyone is approved` },
    {
      label: "Profit",
      value: formatMoney(policy.profit, true),
      note: `${uplift >= 0 ? "+" : ""}${formatPercent(uplift, 0)} vs approving everyone`,
      tone: policy.profit >= all.profit ? "gain" : "loss",
    },
    {
      label: "Losses on approved loans",
      value: formatMoney(policy.realized_loss),
      note: `the model forecast ${formatMoney(policy.expected_loss)}`,
    },
  ];
  return (
    <div className="kpis">
      {tiles.map((tile, i) => (
        <div key={tile.label} className={`kpi load-in ${tile.tone ? `kpi--${tile.tone}` : ""}`} style={{ animationDelay: `${350 + i * 70}ms` }}>
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
            <th>Profit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, o]) => (
            <tr key={name} className={name === "Your policy" ? "is-yours" : undefined}>
              <td>{name}</td>
              <td>{o.threshold > 1 ? "none" : formatPercent(o.threshold)}</td>
              <td>{formatPercent(o.approval_rate)}</td>
              <td>{formatPercent(o.bad_rate, 2)}</td>
              <td>{formatMoney(o.profit, true)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
