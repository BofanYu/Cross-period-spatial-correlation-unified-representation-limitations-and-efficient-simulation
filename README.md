# Cross-period spatial correlation models

```text
data/
  common_records.csv       # processed residual panel
  periods/                 # nine period-specific datasets
  samples/                 # fixed event and station selections
scripts/
  fitting/                 # semivariogram.py and mle.py
  models/                  # one Python file per model family
  analysis/                # tables, plots, prediction and experiments
  data.py
  refit.py
  reproduce.py
results/
  models.pkl               # updated full-data fits: all 8 model families / 10 fitted variants
  analysis/
    evaluation/            # saved event likelihoods and holdout scores
    benchmarks/            # fitting and sampling times, experiment summaries
    data_requirement/      # LMC full-data fit used by the subsampling experiment
    *.csv                  # analysis tables and model consistency checks
  figures/                 # figures 01-13
```

Use Python 3.11 and run commands from this folder:

```bash
python -m pip install -r requirements.txt
python -m scripts.reproduce
```

Reproduction loads `results/models.pkl`, regenerates CSV tables and figures using the saved experiment results, and checks that the PKL exactly matches the original file before and after the run. The check is saved as `results/analysis/models_check.csv`. Computational settings, implementation details and verification commands are described in [PAPER_SYNC.md](PAPER_SYNC.md). The saved `refit_check.csv` records the current Pairwise, weighted-Goulard LMC and powered-exponential IOX refits. Earlier GH08/PCA checks are retained under `results/analysis/historical`. Run `python -m scripts.analysis.verify_paper` for selected-event likelihood/holdout and structured-IOX consistency checks.

To refit models (updates the single PKL and retains unselected models):

```bash
python -m scripts.fitting.semivariogram                   # eight models
python -m scripts.fitting.semivariogram --models gh08 pca # selected models
python -m scripts.fitting.mle                            # PCA MLE
python -m scripts.fitting.mle --models pca lmc            # includes the long LMC fit
```

Use `--output` for an alternative fit file and `--reference` to choose the model file to compare against. Reproduction's exact-file check uses the original baseline, so intentionally refitted files can differ. The refitting command reports numerical curve differences as CSV.

Generate plots with `python -m scripts.analysis.plots` and tables with `python -m scripts.analysis.tables`. After refitting, recompute prediction scores before aggregating them:

```bash
python -m scripts.analysis.predictive --dataset full
python -m scripts.analysis.predictive --dataset complete
python -m scripts.analysis.tables --evaluation-dir results/analysis/prediction/full/full_run
```

`predictive --smoke` runs a small check. `checks`, `check_sampling`, `data_requirements`, and `benchmark_sampling` in `scripts.analysis` provide additional numerical checks and experiment runs. Reports and settings are CSV only. The processed residuals are the starting point; upstream NGA-West2/CY14 preprocessing is not included. Saved timing measurements are used for plots; rerun timings depend on hardware.

## September 11 paper version

See [PAPER_SYNC.md](PAPER_SYNC.md) for current fit/evaluation policies, separate complete-case models, historical benchmark provenance and unresolved manuscript rounding/wording differences. `results/models_complete.pkl` is selected automatically by `predictive --dataset complete`.
