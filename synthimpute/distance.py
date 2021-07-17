import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def cdist_long(XA, XB, preserve_index=True, **kwargs):
    """Melt the result of scipy.cdist.
    
    Args:
        XA: DataFrame 1.
        XB: DataFrame 2.
        preserve_index: Preserve index values from XA and XB. If False, row
                        numbers are returned instead. Defaults to True.
        **kwargs: Other arguments passed to scipy.cdist.
        
    Returns:
        DataFrame with id1, id2, and dist.
    """
    # Ensure the same column order, pending scipy/scipy#9616.
    XB = XB[XA.columns]
    res = pd.DataFrame(cdist(XA, XB, **kwargs)).reset_index().melt("index")
    res.columns = ["id1", "id2", "dist"]
    # id2 is sometimes returned as an object.
    res["id2"] = res.id2.astype(int)
    if preserve_index:
        Amap = pd.DataFrame({"id1": np.arange(XA.shape[0]), "index1": XA.index.values})
        Bmap = pd.DataFrame({"id2": np.arange(XB.shape[0]), "index2": XB.index.values})
        res = (
            res.merge(Amap, on="id1").merge(Bmap, on="id2").drop(["id1", "id2"], axis=1)
        )
        res.columns = ["dist", "id1", "id2"]
    return res


def subset_from_row(df, row):
    """Subset a DataFrame based on values from a row.
    
    Args:
        df: DataFrame.
        row: Row to subset based on.
        
    Returns:
        DataFrame subsetting df based on values in row.
    """
    row_df = pd.DataFrame(row).transpose()
    # Nonstandard merge to retain the index in df.
    return df.reset_index().merge(row_df).set_index("index")


def nearest_record_single(XA1, XB, k=None, **kwargs):
    """Get the nearest record in XA for each record in XB.
    Args:
        XA1: Series.
        XB: DataFrame.
        k: Number of nearest elements to return.
           Defaults to none (just single nearest element).
        **kwargs: Other arguments passed to scipy.cdist, e.g. 
                  `metric='euclidean'`.
    
    Returns:
        DataFrame with columns for id and corresponding distance of nearest:
        If k is None: id_B (from XB) and dist.
        If k is not None: id_b[k] and dist[k] for each k.
    """
    dist = cdist(XA1.values.reshape(1, -1), XB, **kwargs)[0]
    if k is None:
        return pd.Series({"dist": np.amin(dist), "id_B": np.argmin(dist)})
    idx = np.argpartition(dist, k)[:k].tolist()
    distx = dist[idx].tolist()
    dist_ind = ("dist" + np.char.array(np.arange(1, k + 1).astype(str))).tolist()
    idx_ind = ("id_B" + np.char.array(np.arange(1, k + 1).astype(str))).tolist()
    return pd.Series(idx + distx, index=idx_ind + dist_ind)


def nearest_record(XA, XB, k=None, scale=False, **kwargs):
    """Get the nearest record in XA for each record in XB.

    Args:
        XA: DataFrame. Each record is matched against the nearest in XB.
        XB: DataFrame.
        k: Number of nearest items to return. Defaults to None (single nearest).
        scale: Whether to scale the data by XA's mean and standard deviation.
               Defaults to False.
        **kwargs: Other arguments passed to scipy.distance.cdist.

    Returns:
        DataFrame with columns for id_A (from XA), id_B (from XB), and dist.
        Each id_A maps to a single id_B, which is the nearest record from XB.
    """
    # Scale all features according to the XA distribution.
    if scale:
        means = XA.mean()
        stds = XA.std()
        XA = (XA - means) / stds
        XB = (XB - means) / stds
    assert XA.columns.equals(
        XB.columns
    ), "XA and XB must have the same columns (in the same order)."
    res = XA.apply(lambda x: nearest_record_single(x, XB, k, **kwargs), axis=1)
    res["id_A"] = XA.index
    # id_B is sometimes returned as an object.
    id_B_cols = [col for col in res if col.startswith("id_B")]
    dist_cols = [col for col in res if col.startswith("dist")]
    res[id_B_cols] = res[id_B_cols].astype(int)
    # Reorder columns, interleaving IDs and distance.
    return res[["id_A"] + [val for pair in zip(id_B_cols, dist_cols) for val in pair]]


def nearest_synth_train_test(synth, train, test=None, k=None, scale=True, **kwargs):
    """Get the nearest record from synth to each of train and test.

    Args:
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        test: Test/holdout DataFrame. Defaults to None.
        k: Number of nearest elements to return. Defaults to None (single nearest).
        scale: Whether to scale the datasets by means and standard deviations
               in `train`. This avoids using standardized distance metrics
               which will scale datasets differently. Defaults to True.
        **kwargs: Other arguments passed to scipy.cdist, e.g. 
                  `metric='euclidean'`.

    Returns:
        DataFrame with these columns (will vary if k is not None or test is None):
        * synth_id: Index in the synthetic file.
        * train_dist: Shortest distance to a record in the training file.
        * train_id: Index of record in training file with the shortest distance.
        * test_dist: Shortest distance to a record in the test file.
        * test_id: Index of record in test file with the shortest distance.
        * dist_diff: train_dist - test_diff.
        * dist_ratio: train_dist / test_diff.
    """
    # Scale all features according to training distribution.
    if scale:
        means = train.mean()
        stds = train.std()
        train = (train - means) / stds
        synth = (synth - means) / stds
        if test is not None:
            test = (test - means) / stds
    # Calculate the nearest record from each synthetic record in both
    # training and testing sets.
    print("Calculating nearest records to training set...")
    nearest_train = nearest_record(synth, train, k, **kwargs)
    if test is None:  # We're done.
        return nearest_train
    # This will break if k is not None.
    nearest_train.columns = ["synth_id", "train_id", "train_dist"]
    print("Calculating nearest records to test set...")
    nearest_test = nearest_record(synth, test, k, **kwargs)
    nearest_test.columns = ["synth_id", "test_id", "test_dist"]
    # Merge on synth_id, calculate difference in distances, and return.
    nearest = nearest_train.merge(nearest_test, on="synth_id")
    if k is not None:  # Columns won't align.
        nearest["dist_diff"] = nearest.train_dist - nearest.test_dist
        nearest["dist_ratio"] = nearest.train_dist / nearest.test_dist
    return nearest


def print_dist(r):
    """Print a record from a dist DataFrame as a sentence.

    Args:
        r: Record from a dist DataFrame, i.e. from nearest_synth_train_test().

    Returns:
        Nothing. Prints the record as a sentence.
    """
    print(
        "Synthetic record "
        + str(int(r.synth_id))
        + " is closest to training record "
        + str(int(r.train_id))
        + " (distance="
        + str(r.train_dist.round(2))
        + ") and closest to test record "
        + str(int(r.test_id))
        + " (distance="
        + str(r.test_dist.round(2))
        + ")."
    )


def nearest_synth_train_test_record(dist, synth, train, test, verbose=True):
    """Produce DataFrame with a synthetic record and nearest records in the
       train and test sets.

    Args:
        dist: Record from a distance DataFrame, i.e. produced from 
              nearest_synth_train_test().
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        test: Test/holdout DataFrame.
        verbose: Whether to print the dist record as a sentence. Defaults to True.

    Returns:
        DataFrame with three columns--synth, train, test--and rows for
        each column.
    """
    if isinstance(dist, pd.DataFrame):
        dist = dist.iloc[0]
    if verbose:
        print_dist(dist)
    synth_record = synth.iloc[int(dist.synth_id)]
    train_record = train.iloc[int(dist.train_id)]
    test_record = test.iloc[int(dist.test_id)]
    res = pd.concat([train_record, synth_record, test_record], axis=1, sort=True)
    res.columns = ["train", "synth", "test"]
    return res


def nearest_synth_train_records(
    dist, synth, train, k=2, label_distance=True, verbose=True
):
    """Produce DataFrame with a synthetic record and nearest records in the
       train and test sets.

    Args:
        dist: Record from a distance DataFrame, i.e. produced from 
              nearest_synth_train_test().
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        k: Number of nearest to consider. Defaults to 2.
        label_distance: Add the distance to the column header. Defaults to True.
        verbose: Whether to print the dist record as a sentence. Defaults to True.

    Returns:
        DataFrame with three columns--synth, train, test--and rows for
        each column.
    """
    if isinstance(dist, pd.DataFrame):
        dist = dist.iloc[0]
    if verbose:
        print(dist)
    synth_record = synth.iloc[int(dist.id_A)]
    train_record1 = train.iloc[int(dist.id_B1)]
    train_record2 = train.iloc[int(dist.id_B2)]
    if k == 2:
        res = pd.concat([train_record1, synth_record, train_record2], axis=1, sort=True)
        if label_distance:
            res.columns = [
                "train1 (" + str(round(dist.dist1, 2)) + ")",
                "synth",
                "train2 (" + str(round(dist.dist2, 2)) + ")",
            ]
        else:
            res.columns = ["train1", "synth", "train2"]
    else:  # Only k=3 right now.
        train_record3 = train.iloc[int(dist.id_B3)]
        res = pd.concat(
            [synth_record, train_record1, train_record2, train_record3],
            axis=1,
            sort=True,
        )
        if label_distance:
            res.columns = [
                "synth",
                "train1 (" + str(round(dist.dist1, 2)) + ")",
                "train2 (" + str(round(dist.dist2, 2)) + ")",
                "train3 (" + str(round(dist.dist3, 2)) + ")",
            ]
        else:
            res.columns = ["synth", "train1", "train2", "train3"]
    return res
