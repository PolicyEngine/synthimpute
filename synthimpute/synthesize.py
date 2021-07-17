from sklearn import ensemble
import numpy as np
import synthimpute as si


def rf_synth(
    X,
    seed_cols,
    classification_cols=None,
    n=None,
    trees=100,
    random_state=0,
    synth_cols=None,
):
    """Synthesize data via random forests.

    Args:
        X: Data to synthesize, as a pandas DataFrame.
        seed_cols: Columns to seed the synthesis, via sampling with replacement.
        classification_cols: Optional list of numeric columns to synthesize via classification. 
                             String columns are classified by default.
        n: Number of records to produce. Defaults to the number of records in X.
        trees: Number of trees in each model (n_estimators).
        random_state: Random state to use for initial sampling with replacement and random forest models.
        synth_cols: List of variables to synthesize not as seeds. Order is respected.
                    If not provided, set to list(set(X.columns) - set(seed_cols)).

    Returns:
        DataFrame with synthesized data.
    """
    # Start with the seed synthesis.
    if n is None:
        n = X.shape[0]
    synth = X.copy()[seed_cols].sample(n=n, replace=True, random_state=random_state)
    # Initialize random forests model object.
    rf = ensemble.RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=1, random_state=random_state, n_jobs=-1
    )  # Use maximum number of cores.
    # Loop through each variable.
    if synth_cols is None:
        synth_cols = list(set(X.columns) - set(seed_cols))
    np.random.seed(random_state)
    for i, col in enumerate(synth_cols):
        print(
            "Synthesizing feature "
            + str(i + 1)
            + " of "
            + str(len(synth_cols))
            + ": "
            + col
            + "..."
        )
        rf.fit(X[synth.columns], X[col])
        synth[col] = si.rf_quantile(rf, synth, np.random.rand(n))
    return synth.reset_index(drop=True)[X.columns]
