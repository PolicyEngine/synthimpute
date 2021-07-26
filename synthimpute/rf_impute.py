import numpy as np
from sklearn import ensemble
from scipy.optimize import bisect


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
        np.concatenate([np.array(q)[:, np.newaxis], dat], axis=1),
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


def rf_impute(
    x_train,
    y_train,
    x_new,
    x_cols=None,
    random_state=None,
    sample_weight_train=None,
    new_weight=None,
    target=None,
    mean_quantile=0.5,
    rtol: float = 0.05,
    rf: ensemble.RandomForestRegressor = None,
    **kwargs
):
    """Impute labels from a training set to a new data set using
       random forests quantile regression.

    Args:
        x_train: The training predictors. If a MicroDataFrame or MicroSeries is
            passed, train weights are captured automatically
        y_train: The training labels. If a MicroDataFrame or MicroSeries is passed,
            train weights and the new target are captured automatically
        x_new: The new predictors. If a MicroDataFrame or MicroSeries is passed,
            new weights are captured automatically
        x_cols: List of columns to use. If not provided, uses all columns from
            x_train (these must also be in x_new).
        random_state: Optional random seed passed to RandomForestRegressor and
            for uniform distribution of quantiles.
        sample_weight_train: Vector indicating the weights associated with each
            row of x_train/y_train. Defaults to None.
        new_weight: Vector indicating the weights associated with each
            row of x_new. Defaults to None.
        target: Numerical target for the weighted sum of y_new.
        mean_quantile: The mean quantile to use, via a Beta distribution.
            Defaults to 0.5.
        rtol (float): The relative tolerance for matching the target aggregate.
            Defaults to 0.05.
        rf (ensemble.RandomForestRegressor): The fitted model to use.
            Defaults to None.
        **kwargs: Other args passed to RandomForestRegressor, e.g.
            `n_estimators=50`.  rf_impute uses all RandomForestRegressor
            defaults unless otherwise specified.

    Returns:
        Imputed labels for new_x.
    """
    if all(
        map(
            lambda arg: type(arg).__name__
            in ("MicroDataFrame", "MicroSeries"),
            (x_train, y_train, x_new),
        )
    ):
        sample_weight_train = x_train.weights
        new_weight = x_new.weights
        if target is None:
            target = y_train.sum()
        x_train = x_train.values
        y_train = y_train.values
        x_new = x_new.values
    if rf is None:
        rf = ensemble.RandomForestRegressor(
            random_state=random_state, **kwargs
        )
        if sample_weight_train is None:
            rf.fit(x_train, y_train)
        else:
            rf.fit(x_train, y_train, sample_weight=sample_weight_train)
    # Set alpha parameter of Beta(a, 1) distribution.
    # Generate quantiles from Beta(a, 1) distribution.
    rng = np.random.default_rng(random_state)
    if target is not None:

        def aggregate_error(mean_quantile):
            pred = get_result(rf, x_new, mean_quantile, rng)
            pred_agg = (pred * new_weight).sum()
            error = pred_agg - target
            return error

        mean_quantile = bisect(aggregate_error, 0.01, 0.99, rtol=rtol)
    return get_result(rf, x_new, mean_quantile, rng)


def get_result(rf, x_new, mean_quantile, rng):
    """Generates the resulting array from a random forest regression model
    using a specified mean quantile.

    Args:
        rf: The random forest model (fitted)
        x_new: New input values
        mean_quantile: The mean quantile using the Beta distribution
        rng: The random number generator

    Returns:
        Imputed labels for x_new
    """
    a = mean_quantile / (1 - mean_quantile)
    quantiles = rng.beta(a, 1, x_new.shape[0])
    return rf_quantile(rf, x_new, quantiles)
