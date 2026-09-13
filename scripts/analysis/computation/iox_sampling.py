"""Native IOX sampling: period mixing followed by period-specific spatial factors."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist

from scripts.analysis.computation.samplers import StructuredPrior, FactorDiagnostics, psd_rectangular_factor


@dataclass
class IOXPrior(StructuredPrior):
    spatial_lowers: list
    period_factor: np.ndarray
    order: np.ndarray
    inverse_order: np.ndarray
    spatial_factor_info: list
    period_factor_info: FactorDiagnostics
    name: str = "IOX full semivariogram"

    @property
    def n_stations(self):
        return len(self.order)

    @property
    def n_periods(self):
        return len(self.spatial_lowers)

    def sample(self, rng, n_samples=1):
        count = int(n_samples)
        if count < 1:
            raise ValueError("n_samples must be positive")
        n, q = self.n_stations, self.n_periods
        out = np.empty((n, q, count))
        # Bound temporary storage for large realization sweeps. Each batch uses
        # GEMM, not a Python loop over individual realizations.
        batch = min(count, max(1, 64_000_000 // (8*n*(q+self.period_factor.shape[1]))))
        for start in range(0, count, batch):
            stop = min(count, start + batch)
            z = rng.standard_normal((n*(stop-start), self.period_factor.shape[1]))
            u = (z @ self.period_factor.T).reshape(n, stop-start, q)
            for r, lower in enumerate(self.spatial_lowers):
                out[:, r, start:stop] = (lower @ u[:, :, r])[self.inverse_order]
        return out

    def matvec(self, flat_vector):
        v = np.asarray(flat_vector).reshape(self.n_stations, self.n_periods)[self.order]
        temp = np.column_stack([lower.T @ v[:, r]
                                for r, lower in enumerate(self.spatial_lowers)])
        temp = (temp @ self.period_factor) @ self.period_factor.T
        out = np.column_stack([lower @ temp[:, r]
                               for r, lower in enumerate(self.spatial_lowers)])
        return out[self.inverse_order].ravel()

    def diagonal_flat(self):
        variances = np.sum(self.period_factor**2, axis=1)
        diagonal = np.column_stack([np.einsum("ij,ij->i", lower, lower)*variances[r]
                                    for r, lower in enumerate(self.spatial_lowers)])
        return diagonal[self.inverse_order].ravel()

    def zero_lag_period_covariance(self):
        # IOX colocated cross-covariance depends on station/order. Return the
        # first original station's exact block, matching the dense interface.
        index = self.inverse_order[0]
        rows = np.stack([lower[index] for lower in self.spatial_lowers])
        return (rows @ rows.T) * (self.period_factor @ self.period_factor.T)


def build_iox_prior(payload, lat, lon, period_idx, period_variances):
    from scripts.common.core import (
        _earth_chord_coordinates_native, _iox_maxmin_order,
        _matern_correlation_native,
    )
    fit = payload["iox_full_semivariogram"]
    periods = np.asarray(payload["data_state"]["periods"])[period_idx]
    fit_periods = np.asarray(fit["periods"])
    fit_idx = []
    rows = []
    marginal = fit["marginal_parameters"]
    for period in periods:
        match = np.flatnonzero(np.isclose(fit_periods, period, rtol=0, atol=1e-10))
        row = np.flatnonzero(np.isclose(marginal["period_s"], period, rtol=0, atol=1e-10))
        if len(match) != 1 or len(row) != 1:
            raise ValueError(f"Ambiguous IOX parameters for period {period}")
        fit_idx.append(int(match[0]))
        rows.append(int(row[0]))
    sigma = np.asarray(fit["sigma_core"])[np.ix_(fit_idx, fit_idx)]
    scale = np.sqrt(np.asarray(period_variances) / np.clip(np.diag(sigma), 1e-15, None))
    a, info = psd_rectangular_factor(sigma)
    a = scale[:, None]*a
    coords = _earth_chord_coordinates_native(lon, lat)
    order = _iox_maxmin_order(coords)
    distance = cdist(coords[order], coords[order])
    factors, infos = [], []
    for row_index in rows:
        row = marginal.iloc[row_index]
        if 'length_scale_km' in row.index:
            corr=np.exp(-(distance/float(row['length_scale_km']))**float(row['gamma_pe']))
        else:
            corr=(1-float(row['nugget_fraction_alpha']))*_matern_correlation_native(
                distance,nu=float(row['nu']),alpha=float(row['phi_per_km']))
        np.fill_diagonal(corr,1.)
        lower=cholesky(corr,lower=True,check_finite=False)
        jitter=0.0
        factors.append(lower)
        infos.append(FactorDiagnostics(jitter=jitter, rank=len(lat)))
        del corr
    return IOXPrior(factors, a, order, np.argsort(order), infos, info)

