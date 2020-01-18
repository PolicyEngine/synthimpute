import pytest
import numpy as np
import pandas as pd
from sklearn import ensemble
import synthimpute as si


def test_tax():
    N = 1000
    train = pd.DataFrame({'x1': np.random.randn(N),
                          'x2': np.random.randn(N)})
    # Construct example relationship.
    train['y'] = train.x1 + np.power(train.x2, 3) + np.random.randn(N)
    rf = ensemble.RandomForestRegressor(random_state=3)
    xcols = ['x1', 'x2']
    rf.fit(train[xcols], train.y)
    median_preds = si.rf_quantile(rf, train[xcols], 0.5)
    # Test multiple quantiles.
    quantiles = np.arange(N) / N
    multiple_q_preds = si.rf_quantile(rf, train[xcols], quantiles)
