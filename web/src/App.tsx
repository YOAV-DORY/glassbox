import { useEffect, useState } from "react";

import { fetchModelInfo } from "./api";
import { TopBar } from "./components/TopBar";
import { Assess } from "./pages/Assess";
import { Landing } from "./pages/Landing";
import { Portfolio } from "./pages/Portfolio";
import { usePolicy } from "./policy";
import { usePath } from "./router";
import type { ModelInfo } from "./types";

export function App() {
  const path = usePath();
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [offline, setOffline] = useState(false);
  // One lending policy for the whole app: the simulator sets it, the desk decides with it.
  const [policy, setPolicy] = usePolicy();

  // Model facts (version, metrics, policy) are shared by both pages, so fetch them once here.
  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(() => setOffline(true));
  }, []);

  const page = path.startsWith("/assess") ? "assess" : path.startsWith("/portfolio") ? "portfolio" : "landing";

  return (
    <div className="app">
      <TopBar page={page} info={info} offline={offline} />
      {info?.dataset === "synthetic-demo" && (
        <p className="demo-banner" role="status">
          <strong>Demo mode.</strong> This model was trained on synthetic applicants generated in code, so every
          number on these pages is illustrative. The real results are in the README.
        </p>
      )}
      {/* key={page}: switching pages remounts it, which replays its entrance animations */}
      {page === "assess" && <Assess key={page} info={info} policy={policy} />}
      {page === "portfolio" && <Portfolio key={page} policy={policy} setPolicy={setPolicy} />}
      {page === "landing" && <Landing key={page} info={info} />}
      <footer className="colophon">
        <span>Glassbox · Trained on the Home Credit Default Risk dataset (Kaggle).</span>
        <span>Python · scikit-learn · LightGBM · FastAPI · React. A portfolio project, not a lending product.</span>
      </footer>
    </div>
  );
}
