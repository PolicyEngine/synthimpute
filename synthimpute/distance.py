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
