"""Rebuild analysis outputs and check that the fitted-model file is unchanged."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = ROOT / "results/models.pkl"
    output = ROOT / "results/analysis"
    expected = pd.read_csv(output / "models_baseline.csv").iloc[0].sha256
    before = hashlib.sha256(cache.read_bytes()).hexdigest()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for module in ("tables", "plots"):
        subprocess.run([sys.executable, "-m", f"scripts.analysis.{module}"],
                       cwd=ROOT, env=env, check=True)
    after = hashlib.sha256(cache.read_bytes()).hexdigest()
    matched = before == after == expected
    pd.DataFrame([dict(file="models.pkl", baseline_sha256=expected,
                       before_sha256=before, after_sha256=after, passed=matched)]).to_csv(
        output / "models_check.csv", index=False)
    print(f"models.pkl matches the original: {matched}")
    print("Analysis: results/analysis/; figures: results/figures/")
    if not matched:
        raise SystemExit("models.pkl differs from the original; see models_check.csv.")


if __name__ == "__main__":
    main()
