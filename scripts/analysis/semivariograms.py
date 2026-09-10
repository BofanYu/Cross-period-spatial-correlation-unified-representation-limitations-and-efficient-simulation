"""Fitted covariance and semivariogram curves used in the paper WLS objective."""
import numpy as np

def lmc_covariance_curves(fit: dict, h: np.ndarray) -> np.ndarray:
    b_list = [np.asarray(b, dtype=float) for b in fit["B_raw"]]
    length_scales = np.asarray(fit["length_scales"], dtype=float)
    exponents = np.atleast_1d(np.asarray(fit["gamma_pe"], dtype=float))
    if exponents.size == 1:
        exponents = np.repeat(exponents, len(length_scales))

    q = b_list[0].shape[0]
    covariance = np.zeros((q, q, len(h)), dtype=float)
    for b_mat, length_scale, exponent in zip(b_list, length_scales, exponents):
        if length_scale <= 0.0:
            kernel = np.isclose(h, 0.0).astype(float)
        else:
            kernel = np.exp(-((np.maximum(h, 1e-12) / length_scale) ** exponent))
            kernel[np.isclose(h, 0.0)] = 1.0
        covariance += b_mat[:, :, None] * kernel[None, None, :]
    return 0.5 * (covariance + np.swapaxes(covariance, 0, 1))


def pca_covariance_curves(fit: dict, h: np.ndarray) -> np.ndarray:
    loadings = np.asarray(fit["U"], dtype=float)
    eigenvalues = np.asarray(fit["evals"], dtype=float)
    std_pool = np.asarray(fit["std_pool"], dtype=float)
    length_scales = np.asarray(fit["LE"], dtype=float)
    exponents = np.asarray(fit["gammaE"], dtype=float)
    nuggets = np.asarray(fit.get("nugget_fraction", np.zeros_like(length_scales)), dtype=float)

    q = loadings.shape[0]
    covariance = np.zeros((q, q, len(h)), dtype=float)
    std_outer = np.outer(std_pool, std_pool)
    for k, (length_scale, exponent, nugget) in enumerate(
        zip(length_scales, exponents, nuggets)
    ):
        kernel = (1.0 - nugget) * np.exp(
            -((np.maximum(h, 1e-12) / length_scale) ** exponent)
        )
        kernel[np.isclose(h, 0.0)] = 1.0
        component = (
            eigenvalues[k]
            * np.outer(loadings[:, k], loadings[:, k])
            * std_outer
        )
        covariance += component[:, :, None] * kernel[None, None, :]
    return 0.5 * (covariance + np.swapaxes(covariance, 0, 1))


def pairwise_semivariogram_and_correlation_curves(
    fit: dict,
    rho_0: np.ndarray,
    h: np.ndarray,
):
    length_scales = np.asarray(fit["LE"], dtype=float)
    exponents = np.asarray(fit["gammaE"], dtype=float)
    sills = np.asarray(fit["sills"], dtype=float)
    q = length_scales.shape[0]
    semivariogram = np.zeros((q, q, len(h)), dtype=float)
    correlation = np.zeros_like(semivariogram)
    for i in range(q):
        for j in range(q):
            kernel = np.exp(
                -((np.maximum(h, 1e-12) / length_scales[i, j]) ** exponents[i, j])
            )
            kernel[np.isclose(h, 0.0)] = 1.0
            semivariogram[i, j] = sills[i, j] * (1.0 - kernel)
            amplitude = 1.0 if i == j else float(rho_0[i, j])
            correlation[i, j] = amplitude * kernel
    return semivariogram, correlation


def gh08_semivariogram_and_correlation_curves(
    fit: dict,
    rho_0: np.ndarray,
    h: np.ndarray,
):
    length_scales = np.asarray(fit["LE"], dtype=float)
    exponents = np.asarray(fit["gammaE"], dtype=float)
    sills = np.asarray(fit["emp_sills"], dtype=float)
    q = len(length_scales)
    semivariogram = np.zeros((q, q, len(h)), dtype=float)
    correlation = np.zeros_like(semivariogram)
    for i in range(q):
        for j in range(q):
            kmax = max(i, j)
            kernel = np.exp(
                -((np.maximum(h, 1e-12) / length_scales[kmax]) ** exponents[kmax])
            )
            kernel[np.isclose(h, 0.0)] = 1.0
            amplitude = 1.0 if i == j else float(rho_0[i, j])
            cross_sill = np.sqrt(sills[i] * sills[j]) * amplitude
            semivariogram[i, j] = cross_sill * (1.0 - kernel)
            correlation[i, j] = amplitude * kernel
    return semivariogram, correlation


def kronecker_semivariogram_and_correlation_curves(
    fit: dict,
    rho_0: np.ndarray,
    h: np.ndarray,
):
    spatial = np.exp(
        -((np.maximum(h, 1e-12) / float(fit["LE"])) ** float(fit["gammaE"]))
    )
    spatial[np.isclose(h, 0.0)] = 1.0
    sills = np.asarray(fit["emp_sills"], dtype=float)
    q = len(sills)
    semivariogram = np.zeros((q, q, len(h)), dtype=float)
    correlation = np.zeros_like(semivariogram)
    for i in range(q):
        for j in range(q):
            amplitude = 1.0 if i == j else float(rho_0[i, j])
            cross_sill = np.sqrt(sills[i] * sills[j]) * amplitude
            semivariogram[i, j] = cross_sill * (1.0 - spatial)
            correlation[i, j] = amplitude * spatial
    return semivariogram, correlation


def covariance_to_semivariogram(covariance: np.ndarray) -> np.ndarray:
    return covariance[:, :, [0]] - covariance


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    variances = np.clip(np.diag(covariance[:, :, 0]), 1e-12, None)
    scale = np.sqrt(np.outer(variances, variances))
    return covariance / scale[:, :, None]


def build_curves(payload: dict, data, h_model: np.ndarray):
    pairwise_semi, pairwise_corr = pairwise_semivariogram_and_correlation_curves(
        payload["pairwise_empirical"]["fit"], data.rho_0, h_model
    )
    gh08_semi, gh08_corr = gh08_semivariogram_and_correlation_curves(
        payload["gh08"]["direct_semi"], data.rho_0, h_model
    )
    lmc_covariance = lmc_covariance_curves(payload["lmc_semivariogram"], h_model)
    pca_covariance = pca_covariance_curves(payload["pca"]["semivariogram"], h_model)
    kronecker_semi, kronecker_corr = kronecker_semivariogram_and_correlation_curves(
        payload["kronecker"]["semivariogram"], data.rho_0, h_model
    )

    semivariograms = {
        "Pairwise semivariogram": pairwise_semi,
        "GH08 semivariogram": gh08_semi,
        "LMC semivariogram": covariance_to_semivariogram(lmc_covariance),
        "PCA semivariogram": covariance_to_semivariogram(pca_covariance),
        "Kronecker semivariogram": kronecker_semi,
    }
    correlations = {
        "Pairwise semivariogram": pairwise_corr,
        "GH08 semivariogram": gh08_corr,
        "LMC semivariogram": covariance_to_correlation(lmc_covariance),
        "PCA semivariogram": covariance_to_correlation(pca_covariance),
        "Kronecker semivariogram": kronecker_corr,
    }
    return semivariograms, correlations

