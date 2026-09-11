# Computational settings, implementation and verification

## Data and model files

- Full data: 134 events and 110225 observed scalar residuals; missing periods
  are permitted. Fitted parameters and input state are in `results/models.pkl`.
- Complete cases: 41 events, 5555 records and 49995 scalar residuals across
  nine periods. Separate fits are in `results/models_complete.pkl`.
- Periods in seconds: 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3 and 6.

## Fitting and covariance construction

Full-data Pairwise and LMC semivariogram fitting use 45 unique period-pair
targets (9 direct and 36 cross), weighted by inverse distance. Reported WLS
counts each unique target once. Model implementations are in `scripts/models/`.

The full-data LMC uses sequential Goulard updates, rho(0)/3 initialization,
fixed spatial kernels and 30 sweeps. Its update is
`P_PSD(B + W*(N/D - B))`, with diagonal weights 1 and off-diagonal weights 1/2.
The shortest spatial length scale is 0.01 km. The complete-case LMC fit uses
the unweighted Goulard update; the full-data refit command uses weighted updates.

IOX uses `exp(-(h/length)**gamma)` with fitted gamma, zero nugget and no
spatial Cholesky jitter. Structured sampling factors the period covariance
as `Sigma = A A.T`, forms `U = Z A.T` from independent standard normal entries,
and applies each period's spatial factor: `w_r = L_r U[:, r]`.
This gives `Cov(w_r, w_s) = Sigma[r,s] L_r L_s.T` without factoring the full
joint covariance. See `scripts/analysis/iox_sampling.py`.

## Likelihood and predictive holdouts

`scripts/analysis/predictive.py` evaluates fixed fitted models. It constructs
event covariance matrices, selects observed entries, and computes Gaussian
log likelihood using Cholesky solves and log determinants. Holdout prediction
uses conditional Gaussian means and variances without refitting per split.

- Entry holdout: randomly withhold 10% of observed scalar entries per event.
- Station holdout: randomly withhold 10% of stations and their observed periods.
- Both random tasks use 20 replicates, each task starting with seed 0, with
  identical masks across models.
- Period holdout: withhold each of the nine periods in turn.

For full data, GH08 and Pairwise covariance matrices with minimum eigenvalue
below -1e-8 undergo diagonal-preserving Qi-Sun repair (`eps=1e-7`); 0.01 is
then added to the observation covariance diagonal. IOX and Independent use
no nugget or repair. Other full-data models use training shrinkage 1e-7 and
numerical jitter 1e-6. Complete-case evaluation uses 1e-6 diagonal
stabilization and no PSD repair. Likelihood uses the observation covariance
with the stated diagonal stabilization, not the training shrinkage matrix.

Pooled RMSE is `sqrt(sum(SSE)/sum(n))`; pooled CRPS is `sum(CRPS_sum)/sum(n)`.
Replicate means, sample SD (ddof=1), SE, wins and mean ranks are calculated
separately from matched folds. Period variability describes nine distinct
held-out periods; no SE is assigned to these folds. Event-level inputs are
in `results/analysis/evaluation/`; aggregation is in `scripts/analysis/tables.py`.

## Experiments and figures

- Data requirements use 40, 60, 80, 100 and 120 events, with respectively
  100, 97, 100, 94 and 100 successful reference replicates; 134 events form
  the full-data reference. Inputs are under `results/analysis/data_requirement/`
  and `results/analysis/benchmarks/`.
- Fitting-time figures use one recorded timing per model from a shared table.
- Sampling figures read saved benchmark measurements. Running the sampler
  executes the model in the selected cache; timings depend on hardware and
  batching. The saved first-realization metric is setup plus batch-average
  sampling time, not a separately timed first draw.
- PSD diagnostics use seed 123, 40 replicates, 40 distinct complete-case
  stations and all nine periods. Each replicate first selects an eligible
  event uniformly. There is no repair, nugget, jitter or refit; PSD tolerance
  is 1e-8. Configuration, samples and minimum eigenvalues are in
  `results/analysis/psd/`. The summary reports the median and range of the
  replicate minimum eigenvalues.
- Figure and prediction workflows use `cross_period_psd.curves` in the model
  cache. The 40-replicate PSD experiment is read from the separate directory
  above, not the cache's auxiliary `summary` and `detail` fields.

## Running and verifying

Use Python 3.11 and run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m scripts.reproduce
python -m scripts.analysis.verify_paper
python -m scripts.analysis.predictive --dataset full --smoke
python -m scripts.analysis.predictive --dataset complete --smoke
```

`scripts.reproduce` regenerates tables and figures from saved results without
refitting or rerunning benchmarks, and checks the model-file hash against its
baseline. `scripts.analysis.verify_paper` verifies dataset sizes, the full-data
mask hash, 45 selected-event/task score comparisons (including likelihoods),
and the structured IOX covariance identity. This is a selected-event check,
not a rerun of every event. Its output is
`results/analysis/paper_sync_validation.csv`. Smoke tests check a small subset;
omit `--smoke` to run the full evaluation for the chosen dataset.

To check refitting without overwriting the bundled model file:

```bash
python -m scripts.fitting.semivariogram --models pairwise lmc iox --output results/test_models.pkl
```

Refit checks compare covariance curves with the reference cache and record
numerical differences. Recompute predictive scores before aggregating results
for newly fitted parameters.
