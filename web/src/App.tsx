import { useEffect, useState } from "react";

import { fetchModelInfo } from "./api";
import { TopBar } from "./components/TopBar";
import { Assess } from "./pages/Assess";
import { Landing } from "./pages/Landing";
import { usePath } from "./router";
import type { ModelInfo } from "./types";

export function App() {
  const path = usePath();
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [offline, setOffline] = useState(false);

  // Model facts (version, metrics, policy) are shared by both pages, so fetch them once here.
  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(() => setOffline(true));
  }, []);

  const page = path.startsWith("/assess") ? "assess" : "landing";

  return (
    <div className="app">
      <TopBar page={page} info={info} offline={offline} />
      {/* key={page}: switching pages remounts it, which replays its entrance animations */}
      {page === "assess" ? <Assess key={page} info={info} /> : <Landing key={page} info={info} />}
      <footer className="colophon">
        <span>Glassbox · Trained on the Home Credit Default Risk dataset (Kaggle).</span>
        <span>Python · scikit-learn · LightGBM · FastAPI · React. A portfolio project, not a lending product.</span>
      </footer>
    </div>
  );
}
