"""Model names and pooled prediction metrics."""
import numpy as np
import pandas as pd

MAIN = ['Pairwise empirical semivariogram', 'GH08 semivariogram',
        'Kronecker semivariogram', 'PCA MLE', 'PCA semivariogram',
        'LMC block MLE', 'LMC semivariogram']

EXTRA = ['Multivariate Matern semivariogram', 'SBSS semivariogram', 'IOX full semivariogram']

LABELS = {'Kronecker semivariogram': 'Separable kernel semivariogram',
          'LMC block MLE': 'LMC MLE', 'Multivariate Matern semivariogram': 'Multivariate Matern',
          'SBSS semivariogram': 'SBSS', 'IOX full semivariogram': 'IOX'}

METRICS = ['log_likelihood', 'pseudo_rmse', 'pseudo_crps', 'station_rmse',
           'station_crps', 'period_rmse', 'period_crps']

def pooled_scores(detail):
    """Pool errors and CRPS by observation, matching the stored research tables."""
    good = detail.loc[detail.fit_success] if 'fit_success' in detail else detail
    sums = good.groupby('method')[['n_observations', 'sse', 'crps_sum']].sum()
    return pd.DataFrame({'rmse': np.sqrt(sums.sse / sums.n_observations),
                         'crps': sums.crps_sum / sums.n_observations})
