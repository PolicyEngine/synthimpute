import pandas as pd
from scipy.spatial.distance import cdist

def cdist_long(XA, XB, **kwargs):
    """Melt the result of scipy.cdist.
    
    Args:
        XA: DataFrame 1.
        XB: DataFrame 2.
        **kwargs: Other arguments passed to scipy.cdist.
        
    Returns:
        DataFrame with id1, id2, and dist.
    """
    res = pd.DataFrame(cdist(XA, XB, **kwargs)).reset_index().melt('index')
    res.columns = ['id1', 'id2', 'dist']
    return res


def subset_from_row(df, row):
    """Subset a DataFrame based on values from a row.
    
    Args:
        df: DataFrame.
        row: Row to subset based on.
        
    Returns:
        DataFrame subsetting df based on values in row.
    """
    return pd.DataFrame(row).transpose().merge(df, on=row.index.values.tolist())


def block_cdist(XA, XB, block_vars=None, adjacent_vars=None,
                verbose=True, **kwargs):
    """Calculate distance between each record in df1 and df2, blocked on
       certain variables for efficiency.
    
    Args:
        XA: DataFrame.
        XB: DataFrame.
        block_vars: List of variables to block on, i.e. only compare records
                    where they match.
        adjacent_vars: List of integer variables to block on, including adjacent
                       values. For example, records with a value of 2 will be
                       compared against other records with values of 1, 2, and 3.
        verbose: Print status along blocks. Defaults to True.
        **kwargs: Other arguments passed to scipy.cdist, e.g. 
                  `metric='euclidean'`.
    
    Returns:
        DataFrame with id1, id2, distance, for all compared pairs.
    """
    if block_vars is None:
        return cdist_long(XA, XB, **kwargs)
    # TODO: Use adjacent_vars.
    A_blocks = XA[block_vars].drop_duplicates()
    B_blocks = XB[block_vars].drop_duplicates()
    # TODO: Warn when some blocks are dropped.
    blocks = A_blocks.merge(B_blocks, on=block_vars)
    n_blocks = blocks.shape[0]
    res = pd.DataFrame()
    for index, row in blocks.iterrows():
        if verbose:
            print('Running block ' + str(index + 1) + ' of ' + str(n_blocks) +
                  '...')
        res = res.append(cdist_long(subset_from_row(XA, row),
                                    subset_from_row(XB, row), **kwargs))
    return res


def nearest_record(XA, XB, block_vars=None, **kwargs):
    """Get the nearest record in XA to each record in XB.
    
    Args:
        XA: DataFrame.
        XB: DataFrame.
        block_vars: List of variables to block on, i.e. only compare
                    records where they match. Passed to block_cdist.
        **kwargs: Other arguments passed to scipy.cdist, e.g. 
                  `metric='euclidean'`.
    
    Returns:
        A float numpy array of the quantiles, with one row per row of X.
    """
    dist = block_cdist(XA, XB, block_vars)
    # Use nsmallest over min to capture the index of the nearest match.
    nearest = dist.groupby('id1').dist.nsmallest(1).reset_index()
    # Join on level_1 = id2.
    return nearest.set_index('level_1').join(dist.id2).reset_index(drop=True)


def nearest_synth_train_test(synth, train, test, **kwargs):
    """Get the nearest record from synth to each of train and test.

    Args:
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        test: Test/holdout DataFrame.
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
    nearest_train = nearest_record(synth, train, **kwargs)
    nearest_train.columns = ['synth_id', 'train_dist', 'train_id']

    nearest_test = nearest_record(synth, test, **kwargs)
    nearest_test.columns = ['synth_id', 'test_dist', 'test_id']

    nearest = nearest_train.merge(nearest_test, on='synth_id')
    nearest['dist_diff'] = nearest.train_dist - nearest.test_dist
    nearest['dist_ratio'] = nearest.train_dist / nearest.test_dist
    
    return nearest


def nearest_synth_train_test_record(dist, synth, train, test):
    """Produce DataFrame with a synthetic record and nearest records in the
       train and test sets.

    Args:
        dist: Record from a distance DataFrame, i.e. produced from 
              nearest_synth_train_test().
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        test: Test/holdout DataFrame.

    Returns:
        DataFrame with three columns--synth, train, test--and rows for
        each column.
    """
    if isinstance(dist, pd.DataFrame):
        dist = dist.iloc[0]
    synth_record = synth.iloc[dist.synth_id]
    train_record = train.iloc[dist.train_id]
    test_record = test.iloc[dist.test_id]
    res = pd.concat([synth_record, train_record, test_record], axis=1)
    res.columns = ['synth', 'train', 'test']
    return res
