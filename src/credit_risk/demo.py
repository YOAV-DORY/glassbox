"""Train a demo model on synthetic applicants, so the app runs without the Kaggle download.

Run: python -m credit_risk.demo
Then serve it: GLASSBOX_ARTIFACTS_DIR=artifacts/demo uvicorn api.serve:app

Everything lands in artifacts/demo (gitignored), including its own reports, so the real
results in reports/ are never overwritten. The app shows a "demo mode" banner for this model.
"""

from credit_risk.config import ARTIFACTS_DIR
from credit_risk.synthetic import make_synthetic_raw
from credit_risk.train import run

DEMO_DIR = ARTIFACTS_DIR / "demo"


def main() -> None:
    print("Training the demo model on 20,000 synthetic applicants (about 10 seconds)...")
    run(make_synthetic_raw(n=20_000, seed=0), DEMO_DIR, DEMO_DIR / "reports", dataset="synthetic-demo",
        n_boot=200, verbose=False)
    print(f"Demo model saved to {DEMO_DIR}")


if __name__ == "__main__":
    main()
