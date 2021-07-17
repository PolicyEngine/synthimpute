import synthimpute as si
import pandas as pd
import numpy as np
from sklearn import ensemble


def test_rf_impute():
    N_TRAIN = 2000
    N_NEW = 1000
    n = N_TRAIN + N_NEW
    x = pd.DataFrame({"x1": np.random.randn(n), "x2": np.random.randn(n)})
    # Construct example relationship.
    y = x.x1 + np.power(x.x2, 3) + np.random.randn(n)
    # Split into test and train.
    x_train = x.iloc[:N_TRAIN]
    y_train = y.iloc[:N_TRAIN]
    x_test = x.iloc[N_TRAIN:]
    base = si.rf_impute(x_train, y_train, x_test)
    # Try with some args.
    si.rf_impute(x_train, y_train, x_test, random_state=10, n_estimators=200)
    # Try with sample_weight_train.
    si.rf_impute(x_train, y_train, x_test, sample_weight_train=np.random.randn(N_TRAIN))
    # Check that values are higher when using a higher mean quantile.
    higher_q = si.rf_impute(x_train, y_train, x_test, mean_quantile=0.9)
    assert higher_q.mean() > base.mean()
