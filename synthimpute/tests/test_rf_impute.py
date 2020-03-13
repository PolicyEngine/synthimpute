import pytest
import numpy as np
import pandas as pd
from sklearn import ensemble
import synthimpute as si


def test_rf_impute():
    N = 1000
    x = pd.DataFrame({'x1': np.random.randn(N),
                      'x2': np.random.randn(N)})
    # Construct example relationship.
    y = x.x1 + np.power(x.x2, 3) + np.random.randn(N)
    rf = ensemble.RandomForestRegressor(random_state=3)
    rf.fit(x, y)
    median_preds = si.rf_quantile(rf, x, 0.5)
    assert median_preds.size == N
    # Test multiple quantiles.
    quantiles = np.arange(N) / N
    multiple_q_preds = si.rf_quantile(rf, x, quantiles)
    assert multiple_q_preds.size == N
