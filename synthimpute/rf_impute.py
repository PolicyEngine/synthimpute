import numpy as np
from pandas.core.frame import DataFrame
from sklearn import ensemble
from scipy.optimize import bisect
import pandas as pd
from tqdm import trange


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
    random_state=None,
    sample_weight_train=None,
    new_weight=None,
    target=None,
    mean_quantile=0.5,
    rtol: float = 0.05,
    rf: ensemble.RandomForestRegressor = None,
    verbose: bool = False,
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
        verbose (bool): Whether to print progress. Defaults to False.
        **kwargs: Other args passed to RandomForestRegressor, e.g.
            `n_estimators=50`.  rf_impute uses all RandomForestRegressor
            defaults unless otherwise specified.

    Returns:
        Imputed labels for new_x.
    """

    # If labels are multidimensional, impute each separately
    if isinstance(y_train, pd.DataFrame):
        result = pd.DataFrame()
        task = (
            trange(len(y_train.columns), desc="Imputing columns")
            if verbose
            else range(len(y_train.columns))
        )
        for i in task:
            not_yet_predicted_cols = y_train.columns[i:]
            
            x_train_expanded = pd.concat(
                [x_train, y_train.drop(not_yet_predicted_cols, axis=1)],
                axis=1,
            )
            if type(x_train).__name__ == "MicroDataFrame":
                x_train_expanded = type(x_train)(x_train_expanded, weights=x_train.weights)
            
            x_new_expanded = pd.concat([x_new, result], axis=1)

            if type(x_new).__name__ == "MicroDataFrame":
                x_new_expanded = type(x_new)(x_new_expanded, weights=x_new.weights)

            if verbose:
                task.set_description(f"Imputing column {y_train.columns[i]} (targeting {int(target or y_train[y_train.columns[i]].sum()/1e9):,}bn)")

            result[y_train.columns[i]] = rf_impute(
                x_train=x_train_expanded,
                y_train=y_train[y_train.columns[i]],
                x_new=x_new_expanded,
                random_state=random_state,
                sample_weight_train=sample_weight_train,
                new_weight=new_weight,
                target=target,
                mean_quantile=mean_quantile,
                rtol=rtol,
                rf=rf,
                verbose=verbose,
                **kwargs
            )
        return result

    # If Micro(Series, DataFrame) passed, extract weights
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

    cast_to_array = (
        lambda x: x.values if isinstance(x, (pd.Series, pd.DataFrame)) else x
    )
    x_train = cast_to_array(x_train)
    y_train = cast_to_array(y_train)
    x_new = cast_to_array(x_new)

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
