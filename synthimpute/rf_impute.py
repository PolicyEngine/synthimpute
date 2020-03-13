import numpy as np
from sklearn import ensemble


def percentile_qarray_np(dat, q):
    """Get percentiles with a vector of quantiles.

    Args:
        dat: A float numpy array of data to calculate percentiles on.
        q: A float numpy array of quantiles. Should be the same length as dat.

    Returns:
        A float numpy array of the respective percentiles.
    """
    # From https://stackoverflow.com/a/52615768/1840471.
    return np.apply_along_axis(
        lambda x: np.percentile(x[1:], x[0]),
        1,
        np.concatenate([np.array(q)[:, np.newaxis], dat], axis=1)
    )


def rf_quantile(m, X, q):
    """Get quantile predictions from a random forests model.

    Args:
        m: Random forests sklearn model.
        X: New data to predict on.
        q: Quantile(s) to predict (between 0 and 1).
               If multiple quantiles, should be list-like of the same
               length as the number of rows in X.

    Returns:
        A float numpy array of the quantiles, with one row per row of X.
    """
    rf_preds = []
    for estimator in m.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose()  # One row per record.
    # Use simple percentile function if a single quantile.
    if isinstance(q, (int, float)):
        return np.percentile(rf_preds, np.repeat(q * 100, X.shape[0]))
    # percentile_qarray_np is needed for a list of quantiles.
    return percentile_qarray_np(rf_preds, q * 100)


def rf_impute(x_train, y_train, x_new, x_cols=None, random_state=None):
    """Impute labels from a training set to a new data set using 
       random forests quantile regression.
       
    Args:
        x_train: Training data.
        y_train: Training labels.
        x_new: New x data for which imputed labels are generated.
        x_cols: List of columns to use. If not provided, uses all columns from
            x_train (these must also be in x_new).
        random_state: Optional random seed passed to RandomForestRegressor and
            for uniform distribution of quantiles.
        
    Returns:
        Imputed labels for new_x.
    """
    rf = ensemble.RandomForestRegressor(random_state=random_state)
    rf.fit(x_train, y_train)
    if random_state is not None:
        np.random.seed(random_state)
    quantiles = np.random.rand(y_train.size)  # Uniform distribution.
    return rf_quantile(rf, x_new, quantiles)
