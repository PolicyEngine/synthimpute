import pytest
import numpy as np
import pandas as pd
from sklearn import ensemble
import synthimpute as si


def test_tax():
    train = pd.DataFrame({'x1': np.random.randn(1000),
                          'x2': np.random.randn(1000)})
    # Construct example relationship.
    train['y'] = train.x1 + np.power(train.x2, 3) + np.random.randn(1000)
    rf = ensemble.RandomForestRegressor(n_estimators=100,
                                        min_samples_leaf=1, random_state=3, 
                                        verbose=True, 
                                        n_jobs=-1)  # Use maximum number of cores.
    xcols = ['x1', 'x2']
    rf.fit(train[xcols], train.y)
    median_preds = si.rf_quantile(rf, train[xcols], 0.5)
