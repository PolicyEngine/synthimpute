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
    res = pd.DataFrame(cdist(XA, XB, **kwargs)).reset_index().melt('index')
    res.columns = ['id1', 'id2', 'dist']
    # id2 is sometimes returned as an object.
    res['id2'] = res.id2.astype(int)
    if preserve_index:
        Amap = pd.DataFrame({'id1': np.arange(XA.shape[0]),
                             'index1': XA.index.values})
        Bmap = pd.DataFrame({'id2': np.arange(XB.shape[0]),
                             'index2': XB.index.values})
        res = res.merge(Amap, on='id1').merge(Bmap, on='id2').drop(
            ['id1', 'id2'], axis=1)
        res.columns = ['dist', 'id1', 'id2']
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
    return df.reset_index().merge(row_df).set_index('index')


def nearest_record_single(XA1, XB, **kwargs):
    """Get the nearest record in XA for each record in XB.

    Args:
        XA1: Series.
        XB: DataFrame.
        **kwargs: Other arguments passed to scipy.cdist, e.g. 
                  `metric='euclidean'`.
    
    Returns:
        DataFrame with columns for id_B (from XB) and dist.
    """
    dist = cdist(XA1.values.reshape(1, -1), XB, **kwargs)[0]
    return pd.Series({'dist': np.amin(dist), 'id_B': np.argmin(dist)})


def nearest_record(XA, XB, **kwargs):
    """Get the nearest record in XA for each record in XB.

    Args:
        XA: DataFrame. Each record is matched against the nearest in XB.
        XB: DataFrame.

    Returns:
        DataFrame with columns for id_A (from XA), id_B (from XB), and dist.
        Each id_A maps to a single id_B, which is the nearest record from XB.
    """
    # Reorder XB columns to match XA (scipy treats it as a matrix).
    XB = XB[XA.columns]
    res = XA.apply(lambda x: nearest_record_single(x, XB, **kwargs), axis=1)
    res['id_A'] = XA.index
    # id_B is sometimes returned as an object.
    res['id_B'] = res.id_B.astype(int)
    # Reorder columns.
    return res[['id_A', 'id_B', 'dist']]


def nearest_synth_train_test(synth, train, test, scale=True, **kwargs):
    """Get the nearest record from synth to each of train and test.

    Args:
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        test: Test/holdout DataFrame.
        scale: Whether to scale the datasets by means and standard deviations
               in `train`. This avoids using standardized distance metrics
               which will scale datasets differently. Defaults to True.
        **kwargs: Other arguments passed to scipy.cdist, e.g. 
                  `metric='euclidean'`.

    Returns:
        DataFrame with these columns:
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
        test = (test - means) / stds
        synth = (synth - means) / stds
    # Calculate the nearest record from each synthetic record in both
    # training and testing sets.
    print("Calculating nearest records to training set...")
    nearest_train = nearest_record(synth, train, **kwargs)
    nearest_train.columns = ['synth_id', 'train_id', 'train_dist']
    print("Calculating nearest records to test set...")
    nearest_test = nearest_record(synth, test, **kwargs)
    nearest_test.columns = ['synth_id', 'test_id', 'test_dist']
    # Merge on synth_id, calculate difference in distances, and return.
    nearest = nearest_train.merge(nearest_test, on='synth_id')
    nearest['dist_diff'] = nearest.train_dist - nearest.test_dist
    nearest['dist_ratio'] = nearest.train_dist / nearest.test_dist
    return nearest


def print_dist(r):
    """Print a record from a dist DataFrame as a sentence.

    Args:
        r: Record from a dist DataFrame, i.e. from nearest_synth_train_test().

    Returns:
        Nothing. Prints the record as a sentence.
    """
    print("Synthetic record " + str(int(r.synth_id)) +
          " is closest to training record " +
          str(int(r.train_id)) + " (distance=" + str(r.train_dist.round(2)) +
          ") and closest to test record " +
          str(int(r.test_id)) + " (distance=" + str(r.test_dist.round(2)) + ").")


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
    res.columns = ['train', 'synth', 'test']
    return res
