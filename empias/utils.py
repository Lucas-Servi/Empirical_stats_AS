import numpy as np

def p_adjust_fdr_bh(p):
    """
    Perform Benjamini-Hochberg p-value correction (FDR).

    Parameters
    ----------
    p : array-like
        List or array of p-values.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values (FDR).
    """
    p = np.asarray(p)
    n = len(p)
    ascending_order = np.argsort(p)
    descending_order = np.argsort(ascending_order)
    adjusted_p = np.minimum.accumulate((p[ascending_order] * n) / (np.arange(n) + 1))[descending_order]
    adjusted_p = np.clip(adjusted_p, 0, 1)
    return adjusted_p

def log2FC_func(df, cond1, cond2):
    """
    Calculate log2 Fold Change (log2FC) between two conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing conditions as columns.
    cond1 : str or tuple
        Name of the first condition or multiindex column.
    cond2 : str or tuple
        Name of the second condition or multiindex column.

    Returns
    -------
    pandas.Series
        Series with log2 fold changes.
    """
    mean_cond1 = df[cond1].mean(axis=1)
    mean_cond2 = df[cond2].mean(axis=1)
    fc = mean_cond1 / mean_cond2
    log2fc = np.log2(fc.replace(0, np.nan))
    log2fc = log2fc.replace([np.inf, -np.inf], np.nan)
    return log2fc
