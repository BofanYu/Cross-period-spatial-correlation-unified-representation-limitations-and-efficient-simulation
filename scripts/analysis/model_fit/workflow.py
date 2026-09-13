"""Small display helpers for the model-fit notebook."""
import numpy as np
import pandas as pd

from scripts.common import core
from scripts.refit import MODEL_DIR, prepare_data, fit_models, save_models

COMPLETE_LABELS = (
    "Kronecker semivariogram", "PCA MLE", "PCA semivariogram",
    "LMC block MLE", "LMC semivariogram",
)


def data_summary(data):
    rows = []
    for period, frame in zip(data.periods, data.period_dfs):
        counts = frame.groupby("eqid").size()
        rows.append(dict(
            period_s=period, events=len(counts), records=len(frame),
            station_locations=len(frame[["station_latitude", "station_longitude"]].drop_duplicates()),
            median_stations_per_event=counts.median(), min_stations_per_event=counts.min(),
            max_stations_per_event=counts.max(), station_pairs=int((counts * (counts - 1) // 2).sum()),
        ))
    return pd.DataFrame(rows)


def model_summary(payload, dataset="full"):
    curves = payload["cross_period_psd"]["curves"]
    labels = COMPLETE_LABELS if dataset == "complete" else tuple(curves)
    rows = []
    for label in labels:
        h, rho = curves[label]
        rows.append(dict(
            model=label, periods=np.asarray(rho).shape[0], distance_points=len(h),
            maximum_distance_km=float(np.max(h)),
        ))
    return pd.DataFrame(rows)


def load_models(dataset="full"):
    filename = "models.pkl" if dataset == "full" else "models_complete.pkl"
    return core.load_pickle_cross_platform(MODEL_DIR / filename)
