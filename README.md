# Cross-period spatial correlation

The notebooks follow the manuscript and supplementary material, from model fitting to each evaluation. Open a notebook and run its cells from top to bottom. Tables are displayed in cells; figures are saved in `results/figures/`.

## Setup

Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3 --display-name "Python 3"
python -m jupyterlab
```

Run these commands from this repository. In JupyterLab, select the Python 3 kernel and open a notebook below. VS Code notebooks work with the same environment.

## Reading and running order

| Step | Notebook under `scripts/analysis/` | Article output |
|---|---|---|
| 1. Model fit | `model_fit/model_fit.ipynb` | Table 1, Figure 1; full and complete model files |
| 2. PSD | `psd/psd.ipynb` | Section 6.1; GH08/Pairwise, 40 stations x 40 repetitions |
| 3. Full predictive accuracy | `predictive_accuracy/full.ipynb` | Table 2 and supplementary Table S1 |
| 4. Complete predictive accuracy | `predictive_accuracy/complete.ipynb` | Table 3 |
| 5. Structural flexibility | `structural_flexibility/structural_flexibility.ipynb` | Figure 2 and supplementary Figure S1 |
| 6. CPC | `structural_flexibility/cpc.ipynb` | Figure 3 |
| 7. Data requirements, main | `data_requirements/main.ipynb` | Figure 4 |
| 8. Data requirements, supplementary | `data_requirements/supp.ipynb` | Supplementary Figure S2 |
| 9. Fitting computation | `computation/fit.ipynb` | Figure 5 and supplementary Figure S3 |
| 10. Sampling computation | `computation/sampling.ipynb` | Figures 6-8 and supplementary Figures S4-S6 |

## Running the notebooks

Saved results are included. Run cells from top to bottom for the default examples. To recompute full experiments, enable the corresponding `RUN_*` or `RERUN*` switch in each notebook.

Run all notebooks with their current settings:

```bash
python -m scripts.reproduce
```

To run one notebook, for example:

```bash
python -m scripts.reproduce --notebook psd/psd.ipynb
```

## Files

```text
data/                         processed residual CSVs only
scripts/
  fitting/                    semivariogram and MLE algorithms
  models/                     one Python file per model
  common/                     shared numerical and plotting helpers
  analysis/                   article sections and their notebooks
  refit.py                    full/complete fitting workflow
  reproduce.py                notebook execution entry point
results/
  analysis/model_fit/
    models.pkl                full-data fitted models
    models_complete.pkl       five complete-case fitted variants
    samples/                  fixed event/station selections for subsampling
    lmc_full_reference.csv     reference fit for the data-requirement experiment
  figures/                    main Figures 1-8 and supplementary Figures S1-S6
```

Main images begin with `fig01` through `fig08`; supplementary images begin with `figS01` through `figS06`. The processed within-event residuals are the starting point; upstream NGA-West2/CY14 residual preprocessing is not included.
