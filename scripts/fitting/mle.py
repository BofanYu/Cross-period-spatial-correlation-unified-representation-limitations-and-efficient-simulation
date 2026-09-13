"""Gaussian maximum-likelihood fitting for PCA and LMC.

Run: python -m scripts.fitting.mle --models pca
Add lmc to rerun the paper's full three-stage LMC optimization.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scripts.data import haversine_km, powered_exponential_corr


def event_data_variance(events):
    """Estimate pooled variance/cross-variance from event residual vectors."""
    values = [
        np.asarray(ev["y"], dtype=float)
        for ev in events
        if "y" in ev and np.asarray(ev["y"]).size > 0
    ]
    if not values:
        return 1.0

    if all(value.ndim == 1 for value in values):
        y = np.concatenate([value.ravel() for value in values])
        return float(np.var(y, ddof=0))

    matrices = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        matrices.append(arr.reshape(-1, arr.shape[-1]))
    y = np.vstack(matrices)
    centered = y - np.mean(y, axis=0, keepdims=True)
    return (centered.T @ centered) / max(len(centered), 1)


def apply_data_variance_to_kernel(kernel, variance):
    """Scale a correlation kernel by scalar variance or cross-variance matrix."""
    kernel = np.asarray(kernel, dtype=float)
    variance = np.asarray(variance, dtype=float)

    if variance.ndim == 0:
        return max(float(variance), 1e-12) * kernel

    if kernel.ndim == 2 and kernel.shape == variance.shape:
        return variance * kernel

    if kernel.ndim == 2 and variance.ndim == 2:
        return np.kron(kernel, variance)

    raise ValueError(
        f"Cannot apply variance with shape {variance.shape} to kernel shape {kernel.shape}"
    )


def stable_mvn_nll(y, sigma, jitter_grid=(0.0, 1e-8, 1e-6, 1e-4, 1e-3)):
    """Negative log-likelihood with a small jitter backoff for near-singular matrices."""
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ident = np.eye(len(y))

    last_error = None
    for jitter in jitter_grid:
        try:
            sigma_use = sigma if jitter == 0.0 else sigma + jitter * ident
            chol, lower = cho_factor(sigma_use, lower=True, check_finite=False)
            alpha = cho_solve((chol, lower), y, check_finite=False)
            logdet = 2.0 * np.sum(np.log(np.diag(chol)))
            return 0.5 * (len(y) * np.log(2.0 * np.pi) + logdet + y @ alpha)
        except np.linalg.LinAlgError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("stable_mvn_nll failed unexpectedly")


def prepare_univariate_events(
    df,
    value_col="scaled_deltaW",
    event_col="eqid",
    lat_col="station_latitude",
    lon_col="station_longitude",
    min_stations=5,
    scale=None,
):
    """Build standardized event records for a single spatial field."""
    cols = [event_col, lat_col, lon_col, value_col]
    df_use = df[cols].dropna().copy()

    if scale is None:
        scale = float(np.std(df_use[value_col].to_numpy(dtype=float), ddof=0))
    scale = max(float(scale), 1e-8)

    events = []
    for eqid, df_event in df_use.groupby(event_col):
        if len(df_event) < min_stations:
            continue

        y = df_event[value_col].to_numpy(dtype=float) / scale
        d = haversine_km(
            df_event[lat_col].to_numpy(dtype=float)[:, None],
            df_event[lon_col].to_numpy(dtype=float)[:, None],
            df_event[lat_col].to_numpy(dtype=float)[None, :],
            df_event[lon_col].to_numpy(dtype=float)[None, :],
        )
        events.append({
            "eqid": eqid,
            "n": len(df_event),
            "y": y,
            "d": d,
        })

    return events, scale


def fit_pe_mle(
    events,
    initial=(20.0, 0.4),
    bounds=((1.0, 300.0), (0.05, 1.8)),
    jitter=1e-6,
    maxiter=200,
    use_data_variance=False,
):
    """Fit one powered-exponential curve by Gaussian likelihood.

    By default this fits on correlation scale. If use_data_variance=True,
    the likelihood covariance is scaled by the pooled sample variance of y.
    For vector-valued y, the pooled cross-variance matrix is used.
    """
    if not events:
        return {
            "success": False,
            "message": "no events supplied",
            "LE": float(initial[0]),
            "gammaE": float(initial[1]),
            "nll": np.nan,
            "data_variance": np.nan,
            "result": None,
        }

    data_variance = event_data_variance(events) if use_data_variance else 1.0

    def objective(params):
        length_scale, exponent = params
        total = 0.0
        for ev in events:
            kernel = powered_exponential_corr(ev["d"], length_scale, exponent, jitter=0.0)
            sigma = apply_data_variance_to_kernel(kernel, data_variance)
            if np.ndim(sigma) == 2:
                sigma = sigma.copy()
                sigma.flat[:: sigma.shape[0] + 1] += float(jitter)
            total += stable_mvn_nll(np.asarray(ev["y"], dtype=float).ravel(), sigma)
        return float(total)

    result = minimize(
        objective,
        x0=np.asarray(initial, dtype=float),
        bounds=list(bounds),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "LE": float(result.x[0]),
        "gammaE": float(result.x[1]),
        "nll": float(result.fun),
        "data_variance": data_variance,
        "result": result,
    }


def pairwise_gaussian_nll(
    y,
    distance,
    length_scale,
    exponent,
    cutoff_km=100.0,
    data_variance=1.0,
):
    """Composite Gaussian pairwise negative log-likelihood within a distance cutoff."""
    y = np.asarray(y, dtype=float)
    distance = np.asarray(distance, dtype=float)
    idx_i, idx_j = np.where(np.triu(np.ones_like(distance, dtype=bool), k=1))
    if cutoff_km is not None:
        keep = distance[idx_i, idx_j] <= float(cutoff_km)
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]

    if len(idx_i) == 0:
        variance = float(np.asarray(data_variance, dtype=float))
        variance = max(variance, 1e-12)
        return 0.5 * np.sum(np.log(2.0 * np.pi * variance) + (y ** 2) / variance)

    rho = powered_exponential_corr(distance[idx_i, idx_j], length_scale, exponent)
    rho = np.clip(rho, -0.999999, 0.999999)
    variance = float(np.asarray(data_variance, dtype=float))
    variance = max(variance, 1e-12)
    yi = y[idx_i]
    yj = y[idx_j]
    quad = (yi ** 2 - 2.0 * rho * yi * yj + yj ** 2) / (
        variance * np.clip(1.0 - rho ** 2, 1e-12, None)
    )
    logdet = np.log(np.clip((variance ** 2) * (1.0 - rho ** 2), 1e-12, None))
    return float(0.5 * np.sum(2.0 * np.log(2.0 * np.pi) + logdet + quad))


def fit_pe_pairwise_composite_mle(
    events,
    initial=(20.0, 0.4),
    bounds=((1.0, 300.0), (0.05, 1.8)),
    cutoff_km=100.0,
    maxiter=200,
    use_data_variance=False,
):
    """Fit one powered-exponential curve by pairwise composite likelihood."""
    if not events:
        return {
            "success": False,
            "message": "no events supplied",
            "LE": float(initial[0]),
            "gammaE": float(initial[1]),
            "nll": np.nan,
            "data_variance": np.nan,
            "result": None,
        }

    data_variance = event_data_variance(events) if use_data_variance else 1.0

    def objective(params):
        length_scale, exponent = params
        total = 0.0
        for ev in events:
            total += pairwise_gaussian_nll(
                ev["y"],
                ev["d"],
                length_scale,
                exponent,
                cutoff_km=cutoff_km,
                data_variance=data_variance,
            )
        return float(total)

    result = minimize(
        objective,
        x0=np.asarray(initial, dtype=float),
        bounds=list(bounds),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "LE": float(result.x[0]),
        "gammaE": float(result.x[1]),
        "nll": float(result.fun),
        "data_variance": data_variance,
        "result": result,
    }


def fit_fixed_gamma_mle(
    events,
    exponent,
    initial_length_scale=20.0,
    bounds=((1.0, 300.0),),
    jitter=1e-6,
    maxiter=200,
    use_data_variance=False,
):
    """Fit one shared length scale with the exponent fixed."""
    if not events:
        return {
            "success": False,
            "message": "no events supplied",
            "LE": float(initial_length_scale),
            "gammaE": float(exponent),
            "nll": np.nan,
            "data_variance": np.nan,
            "result": None,
        }

    exponent = float(exponent)
    data_variance = event_data_variance(events) if use_data_variance else 1.0

    def objective(params):
        length_scale = float(params[0])
        total = 0.0
        for ev in events:
            kernel = powered_exponential_corr(ev["d"], length_scale, exponent, jitter=0.0)
            sigma = apply_data_variance_to_kernel(kernel, data_variance)
            if np.ndim(sigma) == 2:
                sigma = sigma.copy()
                sigma.flat[:: sigma.shape[0] + 1] += float(jitter)
            total += stable_mvn_nll(np.asarray(ev["y"], dtype=float).ravel(), sigma)
        return float(total)

    result = minimize(
        objective,
        x0=np.asarray([initial_length_scale], dtype=float),
        bounds=list(bounds),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "LE": float(result.x[0]),
        "gammaE": exponent,
        "nll": float(result.fun),
        "data_variance": data_variance,
        "result": result,
    }


def fit_kernel_mixture_weights_mle(
    events,
    length_scales,
    exponents,
    initial_weights=None,
    jitter=1e-6,
    maxiter=200,
):
    """Fit convex weights over a fixed kernel dictionary by Gaussian likelihood."""
    if not events:
        nstruct = len(length_scales)
        weights = np.full(nstruct, 1.0 / max(nstruct, 1))
        return {
            "success": False,
            "message": "no events supplied",
            "weights": weights,
            "nll": np.nan,
            "result": None,
        }

    length_scales = np.asarray(length_scales, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    nstruct = len(length_scales)

    if initial_weights is None:
        initial_weights = np.full(nstruct, 1.0 / nstruct)
    initial_weights = np.asarray(initial_weights, dtype=float)
    initial_weights = np.clip(initial_weights, 1e-6, None)
    initial_weights /= initial_weights.sum()
    theta0 = np.log(initial_weights)

    def unpack(theta):
        theta = np.asarray(theta, dtype=float)
        shifted = theta - np.max(theta)
        exp_theta = np.exp(shifted)
        return exp_theta / exp_theta.sum()

    def objective(theta):
        weights = unpack(theta)
        total = 0.0
        for ev in events:
            sigma = np.zeros_like(ev["d"], dtype=float)
            for weight, length_scale, exponent in zip(weights, length_scales, exponents):
                sigma += weight * powered_exponential_corr(
                    ev["d"],
                    length_scale,
                    exponent,
                    jitter=0.0,
                )
            sigma = sigma.copy()
            sigma.flat[:: sigma.shape[0] + 1] += float(jitter)
            total += stable_mvn_nll(ev["y"], sigma, jitter_grid=(0.0, 1e-8, 1e-6, 1e-4))
        return float(total)

    result = minimize(
        objective,
        x0=theta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "weights": unpack(result.x),
        "nll": float(result.fun),
        "result": result,
    }


def fit_lmc_complete_mle(data, iterations=110):
    """Complete-case exact LMC likelihood in sample-covariance-scaled coordinates.

    Replay the selected fit: 110 L-BFGS-B iterations at gtol=1e-5, followed
    by the final gtol=1e-4 convergence check. No centering, cap or ridge.
    The transformation changes optimizer coordinates, not the covariance.
    """
    from scripts.models import lmc

    q = len(data.period_cols)
    lengths, exponents = np.array([15., 70., 0.]), np.array([.4, 1., 1.])
    blocks = lmc.prepare_lmc_block_events(
        data.df_pca, data.period_cols, lengths, exponents,
        max_stations_per_event=None, center=False)
    objective = lmc.LMCBlockObjective(blocks, q=q, jitter=1e-8, ridge=0.)
    sample = np.cov(data.df_pca[data.period_cols].to_numpy(float), rowvar=False, bias=True)
    scale = np.linalg.cholesky(sample + 1e-8 * np.eye(q))
    tri = np.tril_indices(q)
    ntri = len(tri[0])
    diagonal = tri[0] == tri[1]

    def transform(theta):
        physical_theta, scaled_factors, physical_factors = [], [], []
        for k in range(3):
            values = theta[k*ntri:(k+1)*ntri].copy()
            values[diagonal] = np.exp(values[diagonal])
            factor = np.zeros((q, q))
            factor[tri] = values
            lower = scale @ factor
            packed = lower[tri].copy()
            packed[diagonal] = np.log(packed[diagonal])
            physical_theta.extend(packed)
            scaled_factors.append(factor)
            physical_factors.append(lower)
        return np.asarray(physical_theta), scaled_factors, physical_factors

    def fun(theta):
        physical_theta, scaled, physical = transform(theta)
        value, gradient = objective(physical_theta)
        scaled_gradient = []
        for k in range(3):
            packed = gradient[k*ntri:(k+1)*ntri].copy()
            packed[diagonal] /= np.diag(physical[k])
            matrix = np.zeros((q, q))
            matrix[tri] = packed
            transformed = (scale.T @ matrix)[tri].copy()
            transformed[diagonal] *= np.diag(scaled[k])
            scaled_gradient.extend(transformed)
        return value, np.asarray(scaled_gradient)

    theta = []
    for weight in (.15, .35, .50):
        values = np.zeros(ntri)
        values[diagonal] = .5 * np.log(weight)
        theta.extend(values)
    bounds = [(np.log(1e-5), np.log(10.)) if is_diag else (-10., 10.)
              for _ in range(3) for is_diag in diagonal]
    options = dict(ftol=1e-9, maxls=60, maxcor=20)
    first = minimize(fun, np.asarray(theta), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(options, maxiter=iterations, gtol=1e-5))
    final = minimize(fun, first.x, jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(options, maxiter=1000, gtol=1e-4))
    physical_theta, _, _ = transform(final.x)
    value, gradient = objective(physical_theta)
    b_raw, _ = objective.unpack(physical_theta)
    b_norm = lmc.normalize_B_mle(b_raw)
    rho = lmc.lmc_auto_correlations_mle(b_norm, lengths, exponents, data.h_fine)
    return dict(
        name="LMC exact block likelihood MLE, scaled coordinates",
        success=bool(final.success), message=str(final.message), nll=float(value),
        final_objective_per_dim=float(value),
        final_gradient_inf_norm=float(np.max(np.abs(gradient))),
        scaled_gradient_inf_norm=float(np.max(np.abs(final.jac))),
        length_scales=lengths, gamma_pe=exponents, B_raw=b_raw, B_norm=b_norm,
        sills_raw=np.diag(sum(b_raw)), rho=rho, theta=physical_theta,
        num_events=len(blocks), rows_used=len(data.df_pca),
        max_stations_per_event=None, ridge=0.,
        optimizer_options=dict(options, first_maxiter=iterations, first_gtol=1e-5,
                               final_maxiter=1000, final_gtol=1e-4),
        iterations=int(first.nit + final.nit),
    )


def main():
    from scripts.refit import run_cli
    run_cli("mle", ("pca", "lmc"), default=("pca",))


if __name__ == "__main__":
    main()
