import numpy as np

def p_adjust_fdr_bh(p):
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    Nan values are replaced by pvalue = 1
    
    Return
    ======
    Sorted Numpy Array
    """
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    #p_clean = p[~np.isnan(p)]
    p_clean = np.asfarray(p)#
    by_descend = p_clean.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p_clean)) / np.arange(len(p_clean), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p_clean[by_descend]))
    return q[by_orig]

def log2FC_func(x, cond1, cond2):
    """
    Takes a pandas DataFrame and calculates logFC between two
    conditions
    
    Return
    ======
    Sorted Numpy Array
    """
    FC = np.mean(x[[cond1]])/np.mean(x[[cond2]])
    log2FC = np.log2(FC)
    return log2FC
