import numpy as np
import pandas as pd
from math import inf
from bisect import bisect_left
from itertools import combinations
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF

np.seterr(divide='ignore', invalid="ignore")



###############################

def create_between_conditions_distribution(df, tpm_th):
    log_new_df = df.apply(lambda x: np.log2(x) if np.issubdtype(x.dtype, np.number) else x)
    log_new_df = log_new_df.replace(-inf, np.nan)  # Change -inf to NaN to ignore them on the mean
    log_new_df[log_new_df < np.log2(tpm_th)] = np.nan  # Corte de 1 TPM
    log_new_df_mean_per_treat = log_new_df.groupby("condition", axis=1).mean()
    log_new_df_logFC = log_new_df_mean_per_treat["wt"] - log_new_df_mean_per_treat["RS31OX"]
    log_new_df_tpm_mean = df.groupby("gene").sum().apply(lambda x:
                                                         np.log2(x) if np.issubdtype(x.dtype,
                                                                                     np.number) else x,
                                                         axis=0).dropna().mean(axis=1)
    between_conditions_distribution = log_new_df_logFC.to_frame("logFC").join(log_new_df_tpm_mean.to_frame("tpm_mean"))
    return between_conditions_distribution


def LogFC(a, b):
    return np.log2(a / b)


def avg_log_TPM(a, b):
    return np.mean(np.log10([a, b]))


def create_replicates_distribution(df):
    x = np.empty((0, 0))
    for name, cond in df.groupby("condition", axis=1):
        col_pairs = list(combinations(cond.columns, r=2))
        for pair in col_pairs:
            name = f"{pair[0][0]}_{pair[0][1][-1:]}_{pair[1][1][-1:]}"
            if x.size == 0:
                col = pd.Series(LogFC(df.loc[:, pair[0]], df.loc[:, pair[1]]), name=name)
                sub_df = df.loc[:, [pair[0], pair[1]]]
                avg_log_tpm_mean = sub_df.groupby("gene").sum().apply(lambda fc:
                                                                      np.log10(fc) if np.issubdtype(fc.dtype,
                                                                                                    np.number) else fc,
                                                                      axis=0).dropna().mean(axis=1)
                x_tpm = col.to_frame(name).join(avg_log_tpm_mean.to_frame(f"{name}_TPM")).to_numpy()
                x = x_tpm
                print(x.shape)
            else:
                col = pd.Series(LogFC(df.loc[:, pair[0]], df.loc[:, pair[1]]), name=name)
                sub_df = df.loc[:, [pair[0], pair[1]]]
                avg_log_tpm_mean = sub_df.groupby("gene").sum().apply(lambda fc:
                                                                      np.log10(fc) if np.issubdtype(fc.dtype,
                                                                                                    np.number) else fc,
                                                                      axis=0).dropna().mean(axis=1)
                x_tpm = col.to_frame(name).join(avg_log_tpm_mean.to_frame(f"{name}_TPM")).to_numpy()
                x = np.vstack((x, x_tpm))
                print(x.shape)
                # x = pd.concat([x, x_tpm], axis=0, ignore_index=True)
    print(x)
    replicates_distribution = x[x[:, 1].argsort()]
    # remove Nans
    replicates_distribution = replicates_distribution[~np.isnan(replicates_distribution).any(axis=1)]
    # remove inf/-inf
    replicates_distribution = replicates_distribution[~np.isinf(replicates_distribution).any(axis=1)]
    return replicates_distribution


###
def slice_list(lst, index, slice_len):
    """
    Extract a slice from a list centered around an index with given length.
    
    Args:
        lst: The list to slice
        index: The center index of the slice
        slice_len: The desired length of the slice
    
    Returns:
        A slice of the list centered around index
    """
    half_len = int(slice_len * 0.5)
    diff = index - half_len
    
    # Handle edge cases near beginning of list
    if diff < 0:
        left_bound = 0
        right_bound = min(index + half_len + (-diff) + 1, len(lst))
    # Handle edge cases near end of list
    elif index + half_len >= len(lst):
        upper_diff = index + half_len - len(lst) + 1
        left_bound = max(diff - upper_diff, 0)
        right_bound = len(lst)
    # Standard case
    else:
        left_bound = diff
        right_bound = index + half_len + 1
        
    return lst[left_bound:right_bound]


def get_closest_number(lst, n):
    """
    Find the closest value to n in a sorted list.
    
    Args:
        lst: A sorted list of numbers
        n: The target number
    
    Returns:
        The value in lst that is closest to n
    """
    pos = bisect_left(lst, n)
    if pos == 0:
        return lst[0]
    if pos == len(lst):
        return lst[-1]
    
    before = lst[pos - 1]
    after = lst[pos]
    
    if after - n < n - before:
        return after
    else:
        return before


def get_local_distribution_np(ev_logtpm, replicates_distribution, replicates_logtpms, windows_len):
    """
    Get a local distribution based on a log TPM value.
    
    Args:
        ev_logtpm: The log TPM value
        replicates_distribution: The distribution of replicates
        replicates_logtpms: The log TPM values of replicates
        windows_len: The length of the window to extract
    
    Returns:
        A local distribution slice
    """
    # Find the closest log TPM value
    flat_logtpms = replicates_logtpms.flatten()
    close_rep_logtpm = get_closest_number(flat_logtpms, ev_logtpm)
    
    # Find the index of this value
    index = np.nonzero(replicates_logtpms == close_rep_logtpm)[0][0]
    
    # Get the local distribution
    local_dist = slice_list(replicates_distribution, index, windows_len)
    
    return local_dist


def calculate_empirical_pvalue(local_area, logfc_abs_value):
    """
    Calculate an empirical p-value using ECDF.
    
    Args:
        local_area: Local area values
        logfc_abs_value: Absolute log fold change value
    
    Returns:
        An empirical p-value
    """
    # Use numpy for efficiency
    abs_local_area = np.abs(local_area)
    ecdf = ECDF(abs_local_area)
    # It is divided by 2 because we are using abs(LogFC) values - one-tailed test
    event_pvalue = (1.0 - ecdf(logfc_abs_value)) * 0.5
    return event_pvalue


def pvalue_calc(row, replicates_distribution, area, cutoff):
    """
    Calculate p-value for a single row.
    
    Args:
        row: A row from the between conditions distribution
        replicates_distribution: The distribution of replicates
        area: The area size for local distribution
        cutoff: Cutoff value for log fold change
    
    Returns:
        A p-value
    """
    # Extract values from row
    between_cond_obs_logfc = abs(row[0])
    ev_logtpm = row[1]
    replicates_logtpms = replicates_distribution[:, [1]]
    
    # Apply cutoff filter for efficiency
    if -cutoff < between_cond_obs_logfc < cutoff:
        return 1.0
    
    # Calculate p-value
    local_dist = get_local_distribution_np(ev_logtpm, replicates_distribution, replicates_logtpms, area)
    event_pval = calculate_empirical_pvalue(local_dist[:, 0], between_cond_obs_logfc)
    
    return event_pval


def calculate_events_pvals_multi(between_conditions_distribution,
                                 replicates_distribution, area=1000,
                                 cutoff=0.5, n_jobs=-1, batch_size=100):
    """
    Calculate p-values for multiple events using joblib parallelization.
    
    Args:
        between_conditions_distribution: DataFrame of between conditions distribution
        replicates_distribution: The distribution of replicates
        area: The area size for local distribution (default: 1000)
        cutoff: Cutoff value for log fold change (default: 0.5)
        n_jobs: Number of parallel jobs (-1 uses all available cores)
        batch_size: Size of batches for parallel processing
    
    Returns:
        DataFrame with calculated p-values
    """
    b_c_d_df = between_conditions_distribution.copy()
    
    # Extract rows for parallel processing
    rows = [row.values for _, row in b_c_d_df.iterrows()]
    
    # Use joblib for parallel processing
    pvalues = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        delayed(pvalue_calc)(row, replicates_distribution, area, cutoff)
        for row in rows
    )
    
    # Add results back to DataFrame
    b_c_d_df["pvalues"] = pvalues
    
    return b_c_d_df