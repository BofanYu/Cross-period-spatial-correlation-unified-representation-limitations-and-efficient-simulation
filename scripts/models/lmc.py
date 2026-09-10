"""Linear model of coregionalization: fixed-kernel WLS and exact block likelihood."""
from dataclasses import dataclass
from os import PathLike

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

from scripts.data import effective_ranges, empirical_sills, haversine_km, powered_exponential_corr
from scripts.fitting.mle import stable_mvn_nll



# Likelihood fitting


@dataclass
class LMCBlock:
    """One independent event block for exact LMC likelihood."""

    eqid: object
    y: np.ndarray
    kernels: list[np.ndarray]


@dataclass
class LMCPartialBlock:
    """One event block with an arbitrary station-period observation mask."""

    eqid: object
    y: np.ndarray
    observed_indices: np.ndarray
    n_stations: int
    kernels: list[np.ndarray]


def lmc_variogram_pe_mle(b_list, length_scales, h, gamma_pe=0.4):
    """LMC variogram with powered-exponential kernels."""
    nper = b_list[0].shape[0]
    result = np.zeros((nper, nper))

    gammas = np.atleast_1d(np.asarray(gamma_pe, dtype=float))
    if len(gammas) == 1:
        gammas = np.repeat(gammas, len(length_scales))

    for s, length_scale in enumerate(length_scales):
        kernel = 1.0 - corr_kernel(h, length_scale, gammas[s])
        result += kernel * b_list[s]
    return result


def normalize_B_mle(b_list):
    """Normalize B matrices so per-period diagonal sills sum to one."""
    nstruct = len(b_list)
    nper = b_list[0].shape[0]
    b_norm = [b.copy() for b in b_list]

    diag_sum = np.zeros(nper)
    for s in range(nstruct):
        diag_sum += np.diag(b_norm[s])

    diag_sum = np.clip(diag_sum, 1e-12, None)
    for i in range(nper):
        for j in range(nper):
            denom = np.sqrt(diag_sum[i] * diag_sum[j])
            for s in range(nstruct):
                b_norm[s][i, j] /= denom
    return b_norm


def lmc_auto_correlations_mle(b_norm, length_scales, gamma_pe, h_grid):
    """Return auto-correlations for each period over a distance grid."""
    nper = b_norm[0].shape[0]
    rho = np.zeros((nper, len(h_grid)))

    for ih, hval in enumerate(h_grid):
        if hval == 0:
            rho[:, ih] = 1.0
            continue

        gamma_h = lmc_variogram_pe_mle(b_norm, length_scales, hval, gamma_pe)
        for i in range(nper):
            rho[i, ih] = 1.0 - gamma_h[i, i]
    return rho


def nearest_psd_corr(rho):
    """Project a symmetric matrix onto the PSD cone while keeping unit diagonal."""
    rho = 0.5 * (np.asarray(rho, dtype=float) + np.asarray(rho, dtype=float).T)
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 1e-8, None)
    rho_psd = evecs @ np.diag(evals) @ evecs.T
    scale = np.sqrt(np.clip(np.diag(rho_psd), 1e-12, None))
    rho_psd = rho_psd / np.outer(scale, scale)
    return 0.5 * (rho_psd + rho_psd.T)


def prepare_lmc_events(
    df_use,
    period_cols,
    scales,
    event_col="eqid",
    min_stations=5,
    max_stations_per_event=100,
    random_state=0,
):
    """Build complete-case standardized event records for multivariate LMC likelihood."""
    cols = [event_col, "station_latitude", "station_longitude", *period_cols]
    df_model = df_use[cols].dropna().copy()
    scale_vec = np.asarray(scales, dtype=float)
    scale_vec = np.clip(scale_vec, 1e-8, None)

    events = []
    for eqid, df_event in df_model.groupby(event_col):
        if len(df_event) < min_stations:
            continue

        if max_stations_per_event is not None and len(df_event) > max_stations_per_event:
            df_event = df_event.sample(
                n=int(max_stations_per_event),
                random_state=int(random_state + int(eqid)),
            )

        y = df_event[period_cols].to_numpy(dtype=float) / scale_vec[None, :]
        lats = df_event["station_latitude"].to_numpy(dtype=float)
        lons = df_event["station_longitude"].to_numpy(dtype=float)
        dist = haversine_km(
            lats[:, None],
            lons[:, None],
            lats[None, :],
            lons[None, :],
        )
        events.append({
            "eqid": eqid,
            "n": len(df_event),
            "y": y,
            "d": dist,
        })
    return events


def corr_kernel(distance, length_scale, exponent, cutoff_km=100.0):
    """Powered-exponential correlation kernel."""
    del cutoff_km
    if float(length_scale) <= 0.0:
        return np.where(np.isclose(distance, 0.0), 1.0, 0.0)
    return powered_exponential_corr(
        distance,
        length_scale,
        exponent,
        jitter=0.0,
    )


def vector_to_lower(theta, q):
    """Map an unconstrained vector to a lower-triangular Cholesky factor."""
    theta = np.asarray(theta, dtype=float)
    lower = np.zeros((q, q), dtype=float)
    idx = 0
    for i in range(q):
        lower[i, i] = np.exp(theta[idx])
        idx += 1
        for j in range(i):
            lower[i, j] = theta[idx]
            idx += 1
    return lower


def lower_to_vector(lower):
    """Pack a lower-triangular factor into unconstrained optimization coordinates."""
    q = lower.shape[0]
    values = []
    for i in range(q):
        values.append(np.log(max(lower[i, i], 1e-8)))
        for j in range(i):
            values.append(lower[i, j])
    return np.asarray(values, dtype=float)


def vector_to_b_matrices(theta, q, nstruct):
    """Convert an unconstrained parameter vector to PSD B matrices."""
    n_per_struct = q * (q + 1) // 2
    b_list = []
    offset = 0
    for _ in range(nstruct):
        lower = vector_to_lower(theta[offset : offset + n_per_struct], q)
        b_list.append(lower @ lower.T)
        offset += n_per_struct
    return b_list


def b_matrices_to_vector(b_list):
    """Initialize the unconstrained parameter vector from PSD B matrices."""
    parts = []
    for b_mat in b_list:
        jitter = 1e-8
        while True:
            try:
                lower = np.linalg.cholesky(b_mat + jitter * np.eye(b_mat.shape[0]))
                parts.append(lower_to_vector(lower))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
    return np.concatenate(parts)


def prepare_lmc_block_events(
    df_use,
    period_cols,
    length_scales,
    gamma_pe,
    event_col="eqid",
    min_stations=5,
    max_stations_per_event=80,
    random_state=123,
    center=True,
):
    """Build event blocks for the exact block-diagonal LMC likelihood.

    The covariance convention matches the semivariogram LMC form:
    spatial structures use exp(-(h / ell) ** gamma), while a non-positive
    length scale denotes an exact nugget/identity kernel.
    """
    cols = [event_col, "station_latitude", "station_longitude", *period_cols]
    df_model = df_use[cols].dropna().copy()
    y_all = df_model[period_cols].to_numpy(dtype=float)
    if center:
        y_all = y_all - y_all.mean(axis=0, keepdims=True)
    for j, col in enumerate(period_cols):
        df_model[col] = y_all[:, j]

    length_scales = np.asarray(length_scales, dtype=float)
    gamma_pe = np.atleast_1d(np.asarray(gamma_pe, dtype=float))
    if len(gamma_pe) == 1:
        gamma_pe = np.repeat(gamma_pe, len(length_scales))

    blocks = []
    rng = np.random.default_rng(random_state)
    for eqid, df_event in df_model.groupby(event_col):
        if len(df_event) < int(min_stations):
            continue
        if max_stations_per_event is not None and len(df_event) > int(max_stations_per_event):
            df_event = df_event.sample(
                n=int(max_stations_per_event),
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )

        lat = df_event["station_latitude"].to_numpy(dtype=float)
        lon = df_event["station_longitude"].to_numpy(dtype=float)
        distance = haversine_km(
            lat[:, None],
            lon[:, None],
            lat[None, :],
            lon[None, :],
        )
        kernels = [
            corr_kernel(distance, length_scale, exponent)
            for length_scale, exponent in zip(length_scales, gamma_pe)
        ]
        blocks.append(LMCBlock(
            eqid=eqid,
            y=df_event[period_cols].to_numpy(dtype=float),
            kernels=kernels,
        ))

    blocks.sort(key=lambda block: block.y.shape[0], reverse=True)
    return blocks


def prepare_lmc_partial_block_events(
    df_use,
    period_cols,
    length_scales,
    gamma_pe,
    event_col="eqid",
    min_stations=5,
    max_stations_per_event=80,
    random_state=123,
    center=True,
):
    """Build exact-likelihood event blocks while retaining partial rows.

    Missing station-period entries are not imputed.  Each block stores the
    observed values and their site-major indices in the theoretical ``n*q``
    vector.  The likelihood later selects the corresponding Gaussian
    marginal covariance.
    """
    period_cols = list(period_cols)
    cols = [event_col, "station_latitude", "station_longitude", *period_cols]
    df_model = df_use[cols].dropna(
        subset=[event_col, "station_latitude", "station_longitude"]
    ).copy()
    df_model = df_model.loc[df_model[period_cols].notna().any(axis=1)].copy()
    if df_model.empty:
        return []

    y_all = df_model[period_cols].to_numpy(dtype=float)
    if np.any(np.sum(np.isfinite(y_all), axis=0) == 0):
        missing_periods = [
            period_cols[j]
            for j in np.flatnonzero(np.sum(np.isfinite(y_all), axis=0) == 0)
        ]
        raise ValueError(f"No observations are available for periods: {missing_periods}")
    if center:
        y_all = y_all - np.nanmean(y_all, axis=0, keepdims=True)
    for j, col in enumerate(period_cols):
        df_model[col] = y_all[:, j]

    length_scales = np.asarray(length_scales, dtype=float)
    gamma_pe = np.atleast_1d(np.asarray(gamma_pe, dtype=float))
    if len(gamma_pe) == 1:
        gamma_pe = np.repeat(gamma_pe, len(length_scales))

    blocks = []
    rng = np.random.default_rng(random_state)
    for eqid, df_event in df_model.groupby(event_col):
        if len(df_event) < int(min_stations):
            continue
        if max_stations_per_event is not None and len(df_event) > int(max_stations_per_event):
            df_event = df_event.sample(
                n=int(max_stations_per_event),
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )

        lat = df_event["station_latitude"].to_numpy(dtype=float)
        lon = df_event["station_longitude"].to_numpy(dtype=float)
        distance = haversine_km(
            lat[:, None],
            lon[:, None],
            lat[None, :],
            lon[None, :],
        )
        kernels = [
            corr_kernel(distance, length_scale, exponent)
            for length_scale, exponent in zip(length_scales, gamma_pe)
        ]
        y_full = df_event[period_cols].to_numpy(dtype=float).reshape(-1)
        observed_indices = np.flatnonzero(np.isfinite(y_full)).astype(np.int64)
        if observed_indices.size == 0:
            continue
        blocks.append(LMCPartialBlock(
            eqid=eqid,
            y=y_full[observed_indices],
            observed_indices=observed_indices,
            n_stations=int(len(df_event)),
            kernels=kernels,
        ))

    blocks.sort(key=lambda block: block.y.size, reverse=True)
    return blocks


class LMCBlockObjective:
    """Exact Gaussian block likelihood for fixed-kernel LMC B matrices."""

    def __init__(self, blocks, q, jitter=1e-8, ridge=0.0):
        self.blocks = list(blocks)
        self.q = int(q)
        self.jitter = float(jitter)
        self.ridge = float(ridge)
        self.tril = np.tril_indices(self.q)
        self.ntri = len(self.tril[0])
        self.nstruct = len(self.blocks[0].kernels) if self.blocks else 0
        self.ndata = sum(block.y.shape[0] * self.q for block in self.blocks)
        self.eye_cache = {}

    def pack(self, b_list):
        parts = []
        for b_mat in b_list:
            b_mat = 0.5 * (np.asarray(b_mat, dtype=float) + np.asarray(b_mat, dtype=float).T)
            min_eig = float(np.linalg.eigvalsh(b_mat)[0])
            if min_eig <= self.jitter:
                b_mat = b_mat + (self.jitter - min_eig + 1e-6) * np.eye(self.q)
            lower = np.linalg.cholesky(b_mat)
            for i, j in zip(*self.tril):
                parts.append(np.log(max(lower[i, j], 1e-15)) if i == j else lower[i, j])
        return np.asarray(parts, dtype=float)

    def unpack(self, theta):
        b_list = []
        lower_list = []
        offset = 0
        for _ in range(self.nstruct):
            lower = np.zeros((self.q, self.q), dtype=float)
            for pos, (i, j) in enumerate(zip(*self.tril)):
                value = theta[offset + pos]
                lower[i, j] = np.exp(value) if i == j else value
            offset += self.ntri
            b_list.append(lower @ lower.T + self.jitter * np.eye(self.q))
            lower_list.append(lower)
        return b_list, lower_list

    def __call__(self, theta):
        b_list, lower_list = self.unpack(theta)
        q = self.q
        nll = 0.0
        grad_b = [np.zeros((q, q), dtype=float) for _ in range(self.nstruct)]

        for block in self.blocks:
            n = block.y.shape[0]
            yvec = block.y.reshape(n * q)
            sigma = np.zeros((n * q, n * q), dtype=float)
            for kernel, b_mat in zip(block.kernels, b_list):
                sigma += np.kron(kernel, b_mat)
            sigma = 0.5 * (sigma + sigma.T)
            sigma.flat[:: sigma.shape[0] + 1] += self.jitter

            try:
                chol = cholesky(sigma, lower=True, check_finite=False, overwrite_a=True)
            except Exception:
                return np.inf, np.zeros_like(theta)

            z = solve_triangular(chol, yvec, lower=True, check_finite=False)
            alpha = solve_triangular(chol.T, z, lower=False, check_finite=False)
            nll += 0.5 * (
                2.0 * np.log(np.diag(chol)).sum()
                + yvec.dot(alpha)
                + len(yvec) * np.log(2.0 * np.pi)
            )

            eye = self.eye_cache.get(n)
            if eye is None:
                eye = np.eye(n * q, dtype=float)
                self.eye_cache[n] = eye
            inv = cho_solve((chol, True), eye.copy(), check_finite=False, overwrite_b=True)
            inv4 = inv.reshape(n, q, n, q)
            alpha_mat = alpha.reshape(n, q)
            for s, kernel in enumerate(block.kernels):
                grad_b[s] += 0.5 * (
                    np.einsum("ij,iajb->ab", kernel, inv4, optimize=True)
                    - alpha_mat.T @ kernel @ alpha_mat
                )

        if self.ridge > 0:
            for s, b_mat in enumerate(b_list):
                nll += 0.5 * self.ridge * self.ndata * float(np.sum(b_mat * b_mat))
                grad_b[s] += self.ridge * self.ndata * b_mat

        grad = []
        for grad_mat, lower in zip(grad_b, lower_list):
            grad_mat = 0.5 * (grad_mat + grad_mat.T)
            d_lower = 2.0 * grad_mat @ lower
            for i, j in zip(*self.tril):
                grad.append(d_lower[i, j] * lower[i, j] if i == j else d_lower[i, j])
        return nll / self.ndata, np.asarray(grad, dtype=float) / self.ndata


class LMCPartialBlockObjective(LMCBlockObjective):
    """Exact Gaussian likelihood marginalized to each block's observations."""

    def __init__(self, blocks, q, jitter=1e-8, ridge=0.0):
        self.blocks = list(blocks)
        self.q = int(q)
        self.jitter = float(jitter)
        self.ridge = float(ridge)
        self.tril = np.tril_indices(self.q)
        self.ntri = len(self.tril[0])
        self.nstruct = len(self.blocks[0].kernels) if self.blocks else 0
        self.ndata = int(sum(block.y.size for block in self.blocks))
        self.eye_cache = {}

    def __call__(self, theta):
        b_list, lower_list = self.unpack(theta)
        q = self.q
        nll = 0.0
        grad_b = [np.zeros((q, q), dtype=float) for _ in range(self.nstruct)]

        for block in self.blocks:
            obs = np.asarray(block.observed_indices, dtype=np.int64)
            site_index = obs // q
            period_index = obs % q
            m = int(obs.size)
            sigma = np.zeros((m, m), dtype=float)
            for kernel, b_mat in zip(block.kernels, b_list):
                sigma += (
                    kernel[site_index[:, None], site_index[None, :]]
                    * b_mat[period_index[:, None], period_index[None, :]]
                )
            sigma = 0.5 * (sigma + sigma.T)
            sigma.flat[:: m + 1] += self.jitter

            try:
                chol = cholesky(sigma, lower=True, check_finite=False, overwrite_a=True)
            except Exception:
                return np.inf, np.zeros_like(theta)

            yvec = block.y
            z = solve_triangular(chol, yvec, lower=True, check_finite=False)
            alpha = solve_triangular(chol.T, z, lower=False, check_finite=False)
            nll += 0.5 * (
                2.0 * np.log(np.diag(chol)).sum()
                + yvec.dot(alpha)
                + m * np.log(2.0 * np.pi)
            )

            inv = cho_solve(
                (chol, True),
                np.eye(m, dtype=float),
                check_finite=False,
                overwrite_b=True,
            )
            gaussian_score = 0.5 * (inv - np.outer(alpha, alpha))
            period_one_hot = np.eye(q, dtype=float)[period_index]
            for s, kernel in enumerate(block.kernels):
                kernel_observed = kernel[
                    site_index[:, None], site_index[None, :]
                ]
                grad_b[s] += (
                    period_one_hot.T
                    @ (gaussian_score * kernel_observed)
                    @ period_one_hot
                )

        if self.ridge > 0:
            for s, b_mat in enumerate(b_list):
                nll += 0.5 * self.ridge * self.ndata * float(np.sum(b_mat * b_mat))
                grad_b[s] += self.ridge * self.ndata * b_mat

        grad = []
        for grad_mat, lower in zip(grad_b, lower_list):
            grad_mat = 0.5 * (grad_mat + grad_mat.T)
            d_lower = 2.0 * grad_mat @ lower
            for i, j in zip(*self.tril):
                grad.append(d_lower[i, j] * lower[i, j] if i == j else d_lower[i, j])
        return nll / self.ndata, np.asarray(grad, dtype=float) / self.ndata


def make_lmc_block_bounds(objective, diag_lower=1e-5, diag_upper=10.0, offdiag_bound=10.0):
    """Create L-BFGS-B bounds for Cholesky coordinates."""
    if diag_lower <= 0 and diag_upper <= 0 and offdiag_bound <= 0:
        return None
    bounds = []
    for _ in range(objective.nstruct):
        for i, j in zip(*objective.tril):
            if i == j:
                lo = np.log(diag_lower) if diag_lower > 0 else None
                hi = np.log(diag_upper) if diag_upper > 0 else None
                bounds.append((lo, hi))
            elif offdiag_bound > 0:
                bounds.append((-offdiag_bound, offdiag_bound))
            else:
                bounds.append((None, None))
    return bounds


def load_lmc_block_initial_theta(init, objective, default_theta):
    """Warm-start from an npz path, a fit dict, B matrices, or theta."""
    if init is None or init == "":
        return default_theta

    if isinstance(init, (str, bytes, PathLike)):
        z = np.load(init, allow_pickle=True)
        keys = set(z.files)
        if "theta" in keys:
            theta = np.asarray(z["theta"], dtype=float)
            if theta.shape == default_theta.shape:
                return theta
            raise ValueError(f"theta in {init} has shape {theta.shape}, expected {default_theta.shape}")
        if {"B1", "B2", "B3"}.issubset(keys):
            return objective.pack([
                np.asarray(z["B1"], dtype=float),
                np.asarray(z["B2"], dtype=float),
                np.asarray(z["B3"], dtype=float),
            ])
        raise ValueError(f"{init} must contain either theta or B1/B2/B3")

    if isinstance(init, dict):
        if "theta" in init:
            theta = np.asarray(init["theta"], dtype=float)
            if theta.shape == default_theta.shape:
                return theta
            raise ValueError(f"init theta has shape {theta.shape}, expected {default_theta.shape}")
        if "result" in init and hasattr(init["result"], "x"):
            theta = np.asarray(init["result"].x, dtype=float)
            if theta.shape == default_theta.shape:
                return theta
        if "B_raw" in init:
            return objective.pack(init["B_raw"])

    if isinstance(init, (list, tuple)):
        return objective.pack(init)

    theta = np.asarray(init, dtype=float)
    if theta.shape == default_theta.shape:
        return theta
    raise ValueError(f"Cannot use init with shape {theta.shape}; expected {default_theta.shape}")


def fit_lmc_block_mle(
    df_use,
    period_cols,
    h_grid,
    length_scales=(15.0, 70.0, 0.0),
    gamma_pe=(0.4, 1.0, 1.0),
    weights=(0.15, 0.35, 0.50),
    min_stations=5,
    max_stations_per_event=80,
    random_state=123,
    maxiter=250,
    jitter=1e-8,
    ridge=0.0,
    init=None,
    ftol=1e-7,
    gtol=1e-4,
    maxls=60,
    maxcor=20,
    diag_lower=1e-5,
    diag_upper=10.0,
    offdiag_bound=10.0,
    log_every=0,
    center=True,
):
    """Fit fixed-kernel LMC by exact block-diagonal Gaussian likelihood."""
    q = len(period_cols)
    length_scales = np.asarray(length_scales, dtype=float)
    gamma_pe = np.asarray(gamma_pe, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    blocks = prepare_lmc_block_events(
        df_use,
        period_cols,
        length_scales,
        gamma_pe,
        min_stations=min_stations,
        max_stations_per_event=max_stations_per_event,
        random_state=random_state,
        center=center,
    )
    if not blocks:
        raise ValueError("No events were available for exact LMC block MLE.")

    y_all = np.vstack([block.y for block in blocks])
    sample_cov = np.cov(y_all, rowvar=False, bias=True)
    sample_cov = 0.5 * (sample_cov + sample_cov.T) + 1e-5 * np.eye(q)

    objective = LMCBlockObjective(blocks, q=q, jitter=jitter, ridge=ridge)
    default_theta0 = objective.pack([weight * sample_cov for weight in weights])
    theta0 = load_lmc_block_initial_theta(init, objective, default_theta0)
    bounds = make_lmc_block_bounds(
        objective,
        diag_lower=diag_lower,
        diag_upper=diag_upper,
        offdiag_bound=offdiag_bound,
    )
    f0, g0 = objective(theta0)

    evals = {"n": 1}
    iters = {"n": 0}
    last = {
        "f": float(f0),
        "g_norm": float(np.linalg.norm(g0)),
        "g_inf": float(np.max(np.abs(g0))),
    }
    history = []

    def fun(theta):
        f, g = objective(theta)
        evals["n"] += 1
        last["f"] = float(f)
        last["g_norm"] = float(np.linalg.norm(g))
        last["g_inf"] = float(np.max(np.abs(g)))
        return f, g

    def callback(_theta):
        iters["n"] += 1
        record = {
            "iter": iters["n"],
            "nfev": evals["n"],
            "objective_per_dim": last["f"],
            "gradient_norm": last["g_norm"],
            "gradient_inf_norm": last["g_inf"],
        }
        history.append(record)
        if log_every and (iters["n"] == 1 or iters["n"] % int(log_every) == 0):
            print(record, flush=True)

    result = minimize(
        fun,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
            "maxcor": int(maxcor),
        },
    )

    b_raw, _ = objective.unpack(result.x)
    f_final, g_final = objective(result.x)
    b_norm = normalize_B_mle(b_raw)
    rho = lmc_auto_correlations_mle(b_norm, length_scales, gamma_pe, h_grid)
    sills_raw = np.sum([np.diag(b_mat) for b_mat in b_raw], axis=0)

    return {
        "name": "LMC exact block likelihood MLE",
        "success": bool(result.success),
        "message": str(result.message),
        "nll": float(result.fun),
        "initial_nll": float(f0),
        "initial_gradient_norm": float(np.linalg.norm(g0)),
        "initial_gradient_inf_norm": float(np.max(np.abs(g0))),
        "final_objective_per_dim": float(f_final),
        "final_gradient_norm": float(np.linalg.norm(g_final)),
        "final_gradient_inf_norm": float(np.max(np.abs(g_final))),
        "length_scales": length_scales,
        "gamma_pe": gamma_pe,
        "B_raw": b_raw,
        "B_norm": b_norm,
        "sills_raw": sills_raw,
        "rho": rho,
        "le_eff": effective_ranges(rho, h_grid),
        "num_events": len(blocks),
        "rows_used": int(sum(block.y.shape[0] for block in blocks)),
        "max_stations_per_event": max_stations_per_event,
        "ridge": float(ridge),
        "bounds": None if bounds is None else {
            "diag_lower": float(diag_lower),
            "diag_upper": float(diag_upper),
            "offdiag_bound": float(offdiag_bound),
        },
        "optimizer_options": {
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
            "maxcor": int(maxcor),
        },
        "history": history,
        "result": result,
        "theta": result.x,
        "blocks": blocks,
        "kernel_note": (
            "Uses covariance kernels exp(-(h/ell)^gamma) for ell=15,gamma=0.4 "
            "and ell=70,gamma=1, plus an exact identity nugget. This is the "
            "covariance counterpart of the semivariogram kernels "
            "1 - exp(-(h/ell)^gamma)."
        ),
    }


def _nan_pairwise_covariance(values, min_eigenvalue=1e-5):
    """Pairwise-observed covariance projected to a strictly PSD matrix."""
    values = np.asarray(values, dtype=float)
    q = values.shape[1]
    covariance = np.zeros((q, q), dtype=float)
    fallback_variance = np.nanvar(values, axis=0)
    finite_fallback = fallback_variance[np.isfinite(fallback_variance) & (fallback_variance > 0)]
    global_fallback = float(np.median(finite_fallback)) if finite_fallback.size else 1.0

    for i in range(q):
        for j in range(i + 1):
            keep = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            if np.count_nonzero(keep) >= 2:
                x = values[keep, i]
                y = values[keep, j]
                value = float(np.mean((x - x.mean()) * (y - y.mean())))
            elif i == j:
                value = float(fallback_variance[i])
            else:
                value = 0.0
            if not np.isfinite(value):
                value = global_fallback if i == j else 0.0
            covariance[i, j] = value
            covariance[j, i] = value

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, float(min_eigenvalue), None)
    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def fit_lmc_partial_block_mle(
    df_use,
    period_cols,
    h_grid,
    length_scales=(15.0, 70.0, 0.0),
    gamma_pe=(0.4, 1.0, 1.0),
    weights=(0.15, 0.35, 0.50),
    min_stations=5,
    max_stations_per_event=80,
    random_state=123,
    maxiter=250,
    jitter=1e-8,
    ridge=0.0,
    init=None,
    ftol=1e-7,
    gtol=1e-4,
    maxls=60,
    maxcor=20,
    diag_lower=1e-5,
    diag_upper=10.0,
    offdiag_bound=10.0,
    log_every=0,
    center=True,
):
    """Fit fixed-kernel LMC using every observed station-period value.

    The objective is the exact Gaussian marginal likelihood under the stated
    LMC model.  Missing values are integrated out by selecting the observed
    rows and columns of each event covariance; they are never imputed.
    """
    q = len(period_cols)
    length_scales = np.asarray(length_scales, dtype=float)
    gamma_pe = np.asarray(gamma_pe, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    blocks = prepare_lmc_partial_block_events(
        df_use,
        period_cols,
        length_scales,
        gamma_pe,
        min_stations=min_stations,
        max_stations_per_event=max_stations_per_event,
        random_state=random_state,
        center=center,
    )
    if not blocks:
        raise ValueError("No events were available for partial-data LMC block MLE.")

    sample_cov = _nan_pairwise_covariance(
        df_use[list(period_cols)].to_numpy(dtype=float),
        min_eigenvalue=max(1e-5, jitter),
    )
    objective = LMCPartialBlockObjective(blocks, q=q, jitter=jitter, ridge=ridge)
    default_theta0 = objective.pack([weight * sample_cov for weight in weights])
    theta0 = load_lmc_block_initial_theta(init, objective, default_theta0)
    bounds = make_lmc_block_bounds(
        objective,
        diag_lower=diag_lower,
        diag_upper=diag_upper,
        offdiag_bound=offdiag_bound,
    )
    f0, g0 = objective(theta0)

    evals = {"n": 1}
    iters = {"n": 0}
    last = {
        "f": float(f0),
        "g_norm": float(np.linalg.norm(g0)),
        "g_inf": float(np.max(np.abs(g0))),
    }
    history = []

    def fun(theta):
        f, g = objective(theta)
        evals["n"] += 1
        last["f"] = float(f)
        last["g_norm"] = float(np.linalg.norm(g))
        last["g_inf"] = float(np.max(np.abs(g)))
        return f, g

    def callback(_theta):
        iters["n"] += 1
        record = {
            "iter": iters["n"],
            "nfev": evals["n"],
            "objective_per_observation": last["f"],
            "gradient_norm": last["g_norm"],
            "gradient_inf_norm": last["g_inf"],
        }
        history.append(record)
        if log_every and (iters["n"] == 1 or iters["n"] % int(log_every) == 0):
            print(record, flush=True)

    result = minimize(
        fun,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
            "maxcor": int(maxcor),
        },
    )

    b_raw, _ = objective.unpack(result.x)
    f_final, g_final = objective(result.x)
    b_norm = normalize_B_mle(b_raw)
    rho = lmc_auto_correlations_mle(b_norm, length_scales, gamma_pe, h_grid)
    sills_raw = np.sum([np.diag(b_mat) for b_mat in b_raw], axis=0)
    rows_used = int(sum(block.n_stations for block in blocks))
    observations_used = int(sum(block.y.size for block in blocks))
    theoretical_entries = int(rows_used * q)

    return {
        "name": "LMC exact partial-data block likelihood MLE",
        "success": bool(result.success),
        "message": str(result.message),
        "nll": float(result.fun),
        "initial_nll": float(f0),
        "initial_gradient_norm": float(np.linalg.norm(g0)),
        "initial_gradient_inf_norm": float(np.max(np.abs(g0))),
        "final_objective_per_dim": float(f_final),
        "final_gradient_norm": float(np.linalg.norm(g_final)),
        "final_gradient_inf_norm": float(np.max(np.abs(g_final))),
        "length_scales": length_scales,
        "gamma_pe": gamma_pe,
        "B_raw": b_raw,
        "B_norm": b_norm,
        "sills_raw": sills_raw,
        "rho": rho,
        "le_eff": effective_ranges(rho, h_grid),
        "num_events": len(blocks),
        "rows_used": rows_used,
        "observations_used": observations_used,
        "missing_entries": theoretical_entries - observations_used,
        "observation_fraction": observations_used / theoretical_entries,
        "max_stations_per_event": max_stations_per_event,
        "partial_data": True,
        "ridge": float(ridge),
        "bounds": None if bounds is None else {
            "diag_lower": float(diag_lower),
            "diag_upper": float(diag_upper),
            "offdiag_bound": float(offdiag_bound),
        },
        "optimizer_options": {
            "maxiter": int(maxiter),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
            "maxcor": int(maxcor),
        },
        "history": history,
        "result": result,
        "theta": result.x,
        "blocks": blocks,
        "kernel_note": (
            "Uses fixed LMC covariance kernels and exact observed-data Gaussian "
            "marginals. Missing station-period entries are integrated out by "
            "covariance subsetting, without imputation."
        ),
    }


def build_lmc_event_covariance(distance, b_std_list, length_scales, gamma_pe, jitter=1e-6):
    """Build one full block covariance matrix for an event."""
    q = b_std_list[0].shape[0]
    kernels = [
        corr_kernel(distance, length_scale, exponent)
        for length_scale, exponent in zip(length_scales, gamma_pe)
    ]

    blocks = [[None] * q for _ in range(q)]
    for i in range(q):
        for j in range(q):
            block = np.zeros_like(distance, dtype=float)
            for b_mat, kernel in zip(b_std_list, kernels):
                block += b_mat[i, j] * kernel
            blocks[i][j] = block

    sigma = np.block(blocks)
    sigma = 0.5 * (sigma + sigma.T)
    sigma.flat[:: sigma.shape[0] + 1] += float(jitter)
    return sigma


def lmc_nll(theta, events, q, length_scales, gamma_pe, jitter=1e-6):
    """Joint Gaussian negative log-likelihood for fixed-kernel LMC."""
    b_std_list = vector_to_b_matrices(theta, q=q, nstruct=len(length_scales))
    total = 0.0
    for ev in events:
        sigma = build_lmc_event_covariance(
            ev["d"],
            b_std_list,
            length_scales,
            gamma_pe,
            jitter=jitter,
        )
        yvec = np.concatenate([ev["y"][:, j] for j in range(q)])
        total += stable_mvn_nll(yvec, sigma, jitter_grid=(0.0, 1e-8, 1e-6, 1e-4))
    return float(total)


def lmc_wss(gamma_emp, bin_size, b_list, length_scales, gamma_pe):
    """Weighted sum of squares against the empirical cross-variograms."""
    nper = len(gamma_emp)
    nmax = len(gamma_emp[0][0])
    wss = 0.0

    for k in range(nmax):
        h = bin_size * (k + 1)
        gamma_fit = lmc_variogram_pe_mle(b_list, length_scales, h, gamma_pe)
        for i in range(nper):
            for j in range(nper):
                wss += (1.0 / h) * (gamma_emp[i][j][k] - gamma_fit[i, j]) ** 2
    return float(wss)


def make_b_initial(rho_emp_0_psd, weights):
    """Create a PSD starting point by splitting lag-zero correlation across structures."""
    rho_emp_0_psd = np.asarray(rho_emp_0_psd, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return [weight * rho_emp_0_psd for weight in weights]


def fit_lmc_c4_mle(
    df_use,
    period_cols,
    gamma_emp,
    periods,
    bin_size,
    rho_emp_0,
    h_grid,
    n_iterations=30,
    min_stations=5,
    max_stations_per_event=100,
    random_state=0,
):
    """Fit the retained LMC C4 model by MLE over Cholesky-parameterized B matrices."""
    del periods
    q = len(period_cols)
    length_scales = np.array([0.01, 15.0, 70.0], dtype=float)
    gamma_pe = np.array([1.0, 0.4, 1.0], dtype=float)

    emp_sills = empirical_sills(gamma_emp, tail_bins=min(10, len(gamma_emp[0][0])))
    scale_vec = np.sqrt(np.clip(emp_sills, 1e-8, None))
    events = prepare_lmc_events(
        df_use,
        period_cols,
        scale_vec,
        min_stations=min_stations,
        max_stations_per_event=max_stations_per_event,
        random_state=random_state,
    )

    rho_emp_0_psd = nearest_psd_corr(rho_emp_0)
    b_init_std = make_b_initial(rho_emp_0_psd, weights=np.array([0.05, 0.55, 0.40]))
    theta0 = b_matrices_to_vector(b_init_std)

    result = minimize(
        lmc_nll,
        theta0,
        args=(events, q, length_scales, gamma_pe),
        method="L-BFGS-B",
        options={"maxiter": int(max(50, n_iterations * 5))},
    )

    b_std = vector_to_b_matrices(result.x, q=q, nstruct=len(length_scales))
    scale_outer = np.outer(scale_vec, scale_vec)
    b_raw = [b_mat * scale_outer for b_mat in b_std]
    b_norm = normalize_B_mle(b_raw)
    rho = lmc_auto_correlations_mle(b_norm, length_scales, gamma_pe, h_grid)
    sills_raw = np.sum([np.diag(b) for b in b_raw], axis=0)
    wss = lmc_wss(gamma_emp, bin_size, b_raw, length_scales, gamma_pe)

    return {
        "name": "LMC C4: Nugget + PE + Exp (MLE on B1/B2/B3)",
        "success": bool(result.success),
        "message": str(result.message),
        "nll": float(result.fun),
        "wss": wss,
        "length_scales": np.asarray(length_scales, dtype=float),
        "gamma_pe": gamma_pe,
        "B_raw": b_raw,
        "B_norm": b_norm,
        "B_std": b_std,
        "sills_raw": sills_raw,
        "rho": rho,
        "le_eff": effective_ranges(rho, h_grid),
        "rho_0_psd": rho_emp_0_psd,
        "emp_sills": emp_sills,
        "scales": scale_vec,
        "num_events": len(events),
        "max_stations_per_event": max_stations_per_event,
        "result": result,
    }


def lmc_cross_corr_mle(i, j, h_arr, b_list, length_scales, gamma_pe):
    """Cross-period correlation from raw LMC B matrices."""
    sill_ii = sum(b[i, i] for b in b_list)
    sill_jj = sum(b[j, j] for b in b_list)
    sill_ij = sum(b[i, j] for b in b_list)

    rho = np.zeros(len(h_arr))
    for ih, hval in enumerate(h_arr):
        if hval == 0:
            rho[ih] = sill_ij / np.sqrt(np.clip(sill_ii * sill_jj, 1e-12, None))
            continue

        gamma_h = lmc_variogram_pe_mle(b_list, length_scales, hval, gamma_pe)
        cov_ij = sill_ij - gamma_h[i, j]
        rho[ih] = cov_ij / np.sqrt(np.clip(sill_ii * sill_jj, 1e-12, None))
    return rho



# Semivariogram fitting


def lmc_variogram_pe_semivariogram(b_list, length_scales, h, gamma_pe=0.4):
    """LMC variogram with powered-exponential kernels."""
    nper = b_list[0].shape[0]
    result = np.zeros((nper, nper))

    gammas = np.atleast_1d(np.asarray(gamma_pe, dtype=float))
    if len(gammas) == 1:
        gammas = np.repeat(gammas, len(length_scales))

    for s, length_scale in enumerate(length_scales):
        kernel = 1.0 - np.exp(-((h / length_scale) ** gammas[s]))
        result += kernel * b_list[s]
    return result


def goulard_pe(
    gamma_emp,
    periods,
    bin_size,
    b_init,
    length_scales,
    gamma_pe=0.4,
    n_iterations=10,
):
    """Goulard update algorithm with fixed powered-exponential kernels."""
    nper = len(periods)
    nstruct = len(length_scales)
    nmax = len(gamma_emp[0][0])

    gammas = np.atleast_1d(np.asarray(gamma_pe, dtype=float))
    if len(gammas) == 1:
        gammas = np.repeat(gammas, nstruct)

    gamma_exper = []
    for k in range(nmax):
        mat = np.zeros((nper, nper))
        for i in range(nper):
            for j in range(nper):
                mat[i, j] = gamma_emp[i][j][k]
        gamma_exper.append(mat)

    b_current = [b.copy() for b in b_init]

    for _ in range(n_iterations):
        for l0 in range(nstruct):
            numerator = np.zeros((nper, nper))
            denominator = 0.0

            for k in range(nmax):
                h = bin_size * (k + 1)
                b_removed = [b.copy() for b in b_current]
                b_removed[l0] = np.zeros((nper, nper))

                gamma_trunc = lmc_variogram_pe_semivariogram(
                    b_removed,
                    length_scales,
                    h,
                    gammas,
                )
                kernel = 1.0 - np.exp(-((h / length_scales[l0]) ** gammas[l0]))

                numerator += (1.0 / h) * (gamma_exper[k] - gamma_trunc) * kernel
                denominator += (1.0 / h) * kernel ** 2

            evals, evecs = np.linalg.eigh(numerator)
            evals[evals < 0.0] = 1e-10
            numerator_psd = evecs @ np.diag(evals) @ evecs.T
            b_current[l0] = numerator_psd / denominator

    wss = 0.0
    for k in range(nmax):
        h = bin_size * (k + 1)
        gamma_fit = lmc_variogram_pe_semivariogram(b_current, length_scales, h, gammas)
        for i in range(nper):
            for j in range(nper):
                wss += (1.0 / h) * (gamma_exper[k][i, j] - gamma_fit[i, j]) ** 2

    return wss, b_current


def normalize_B_semivariogram(b_list):
    """Normalize B matrices so per-period diagonal sills sum to one."""
    nstruct = len(b_list)
    nper = b_list[0].shape[0]
    b_norm = [b.copy() for b in b_list]

    diag_sum = np.zeros(nper)
    for s in range(nstruct):
        diag_sum += np.diag(b_norm[s])

    for i in range(nper):
        for j in range(nper):
            denom = np.sqrt(diag_sum[i] * diag_sum[j])
            for s in range(nstruct):
                b_norm[s][i, j] /= denom
    return b_norm


def make_B_init(nper, nstruct, rho_emp_0):
    """Initialize B matrices from empirical lag-zero correlations."""
    return [rho_emp_0.copy() / nstruct for _ in range(nstruct)]


def lmc_auto_correlations_semivariogram(b_norm, length_scales, gamma_pe, h_grid):
    """Return auto-correlations for each period over a distance grid."""
    nper = b_norm[0].shape[0]
    rho = np.zeros((nper, len(h_grid)))

    for ih, hval in enumerate(h_grid):
        if hval == 0:
            rho[:, ih] = 1.0
            continue

        gamma_h = lmc_variogram_pe_semivariogram(b_norm, length_scales, hval, gamma_pe)
        for i in range(nper):
            rho[i, ih] = 1.0 - gamma_h[i, i]
    return rho


def fit_lmc_case(
    name,
    gamma_emp,
    periods,
    bin_size,
    rho_emp_0,
    h_grid,
    length_scales,
    gamma_pe,
    n_iterations=30,
):
    """Fit one fixed-kernel LMC case and return diagnostics."""
    nper = len(periods)
    nstruct = len(length_scales)
    b_init = make_B_init(nper, nstruct, rho_emp_0)

    wss, b_raw = goulard_pe(
        gamma_emp,
        periods,
        bin_size,
        b_init,
        length_scales,
        gamma_pe=gamma_pe,
        n_iterations=n_iterations,
    )
    b_norm = normalize_B_semivariogram(b_raw)

    gammas = np.atleast_1d(np.asarray(gamma_pe, dtype=float))
    if len(gammas) == 1:
        gammas = np.repeat(gammas, nstruct)

    rho = lmc_auto_correlations_semivariogram(b_norm, length_scales, gammas, h_grid)
    sills_raw = np.sum([np.diag(b) for b in b_raw], axis=0)

    return {
        "name": name,
        "wss": wss,
        "length_scales": np.asarray(length_scales, dtype=float),
        "gamma_pe": gammas,
        "B_raw": b_raw,
        "B_norm": b_norm,
        "sills_raw": sills_raw,
        "rho": rho,
        "le_eff": effective_ranges(rho, h_grid),
    }


def fit_lmc_c4_semivariogram(
    gamma_emp,
    periods,
    bin_size,
    rho_emp_0,
    h_grid,
    n_iterations=30,
):
    """Fit the retained LMC C4 model: nugget + PE + exponential."""
    return fit_lmc_case(
        "LMC C4: Nugget + PE + Exp",
        gamma_emp,
        periods,
        bin_size,
        rho_emp_0,
        h_grid,
        length_scales=[0.01, 15.0, 70.0],
        gamma_pe=[1.0, 0.4, 1.0],
        n_iterations=n_iterations,
    )


def lmc_cross_corr_semivariogram(i, j, h_arr, b_list, length_scales, gamma_pe):
    """Cross-period correlation from raw LMC B matrices."""
    sill_ii = sum(b[i, i] for b in b_list)
    sill_jj = sum(b[j, j] for b in b_list)
    sill_ij = sum(b[i, j] for b in b_list)

    rho = np.zeros(len(h_arr))
    for ih, hval in enumerate(h_arr):
        if hval == 0:
            rho[ih] = sill_ij / np.sqrt(sill_ii * sill_jj)
            continue

        gamma_h = lmc_variogram_pe_semivariogram(b_list, length_scales, hval, gamma_pe)
        cov_ij = sill_ij - gamma_h[i, j]
        rho[ih] = cov_ij / np.sqrt(sill_ii * sill_jj)
    return rho
