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
