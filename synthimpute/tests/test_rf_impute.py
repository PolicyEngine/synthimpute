import synthimpute as si
import pandas as pd
import numpy as np
from sklearn import ensemble


def test_rf_impute():
    N_TRAIN = 1000
    N_NEW = 1000
    n = N_TRAIN + N_NEW
    x = pd.DataFrame({'x1': np.random.randn(n),
                      'x2': np.random.randn(n)})
    # Construct example relationship.
    y = x.x1 + np.power(x.x2, 3) + np.random.randn(n)
    si.rf_impute(x.iloc[:N_TRAIN], y.iloc[:N_TRAIN], x.iloc[N_TRAIN:])
    # Try with some args.
    si.rf_impute(x.iloc[:N_TRAIN], y.iloc[:N_TRAIN], x.iloc[N_TRAIN:],
                 random_state=10, n_estimators=200)
