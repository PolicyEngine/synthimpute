import pandas as pd
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
    res = pd.DataFrame(cdist(XA, XB, **kwargs)).reset_index().melt('index')
    res.columns = ['id1', 'id2', 'dist']
    if preserve_index:
        Amap = pd.DataFrame({'id1': np.arange(XA.shape[0]),
                             'index1': XA.index})
        Bmap = pd.DataFrame({'id2': np.arange(XB.shape[0]),
                             'index2': XB.index})
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
        DataFrame with columns for id1 (from XA), id2 (from XB), and dist.
        Each id1 maps to a single id2, which is the nearest record from XB.
    """
    dist = block_cdist(XA, XB, block_vars)
    # Use nsmallest over min to capture the index of the nearest match.
    nearest = dist.groupby('id1').dist.nsmallest(1).reset_index()
    # Join on level_1 = id2.
    return nearest.set_index('level_1').join(dist.id2).reset_index(drop=True)


def nearest_synth_train_test(synth, train, test, block_vars=None, scale=True,
                             **kwargs):
    """Get the nearest record from synth to each of train and test.

    Args:
        synth: Synthetic DataFrame.
        train: Training DataFrame.
        test: Test/holdout DataFrame.
        block_vars: List of variables to block on.
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
    nearest_train = nearest_record(synth, train, block_vars, **kwargs)
    nearest_train.columns = ['synth_id', 'train_dist', 'train_id']
    nearest_test = nearest_record(synth, test, block_vars, **kwargs)
    nearest_test.columns = ['synth_id', 'test_dist', 'test_id']
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
    print("Synthetic record " + str(r.synth_id) + " is closest to training record " +
          str(r.train_id) + " (distance=" + str(r.train_dist.round(2)) +
          ") and closest to test record " +
          str(r.test_id) + " (distance=" + str(r.test_dist.round(2)) + ").")


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
    synth_record = synth.iloc[dist.synth_id]
    train_record = train.iloc[dist.train_id]
    test_record = test.iloc[dist.test_id]
    res = pd.concat([synth_record, train_record, test_record], axis=1)
    res.columns = ['synth', 'train', 'test']
    return res


