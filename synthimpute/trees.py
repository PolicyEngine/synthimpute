from sklearn.tree import DecisionTreeRegressor
import pandas as pd


def decision_tree_regressor_predict_proba(X_train, y_train, X_test, **kwargs):
    """Trains DecisionTreeRegressor model and predicts probabilities of each y.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: New data to predict on.
        **kwargs: Other arguments passed to DecisionTreeRegressor.

    Returns:
        DataFrame with columns for record_id (row of X_test), y 
        (predicted value), and prob (of that y value).
        The sum of prob equals 1 for each record_id.
    """
    # Train model.
    m = DecisionTreeRegressor(**kwargs).fit(X_train, y_train)
    # Get y values corresponding to each node.
    node_ys = pd.DataFrame({"node_id": m.apply(X_train), "y": y_train})
    # Calculate probability as 1 / number of y values per node.
    node_ys["prob"] = 1 / node_ys.groupby(node_ys.node_id).transform("count")
    # Aggregate per node-y, in case of multiple training records with the same y.
    node_ys_dedup = (
        node_ys.groupby(["node_id", "y"]).prob.sum().to_frame().reset_index()
    )
    # Extract predicted leaf node for each new observation.
    leaf = (
        pd.DataFrame(m.decision_path(X_test).toarray())
        .apply(lambda x: x.nonzero()[0].max(), axis=1)
        .to_frame(name="node_id")
    )
    leaf["record_id"] = leaf.index
    # Merge with y values and drop node_id.
    return (
        leaf.merge(node_ys_dedup, on="node_id")
        .drop("node_id", axis=1)
        .sort_values(["record_id", "y"])
    )
