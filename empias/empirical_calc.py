import numpy as np
import pandas as pd
from math import inf
from bisect import bisect_left
from itertools import combinations
from pandarallel import pandarallel
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
    half_len = int(slice_len * 0.5)
    diff = index - half_len
    if diff < 0:
        left_bound = 0
        right_bound = index + half_len + (-diff) + 1
    elif index + half_len >= len(lst):
        upper_diff = index + half_len - len(lst) + 1
        left_bound = diff - upper_diff
        right_bound = index + half_len + 1
    else:
        left_bound = diff
        right_bound = index + half_len + 1
    local_logfc = lst[left_bound:right_bound]
    return local_logfc


def get_closest_number(lst, n):
    """
    Assumes lst is sorted. Returns the closest value to n.
    If two numbers are equally close, return the smallest number.
    Source: http://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/
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
    close_rep_logtpm = get_closest_number(replicates_logtpms, ev_logtpm)[0]
    local_dist = slice_list(replicates_distribution, np.where(replicates_logtpms == close_rep_logtpm)[0][0],
                            windows_len)
    return local_dist


def calculate_empirical_pvalue(local_area, logfc_abs_value):
    abs_local_area = [abs(val) for val in local_area]
    ecdf = ECDF(abs_local_area)
    # It is divided by 2 because we are using abs(LogFC) values and therefore it is a one-tailed test
    event_pvalue = (1.0 - ecdf(logfc_abs_value)) * 0.5
    return event_pvalue


###

def calculate_events_pvals_multi(between_conditions_distribution,
                                 replicates_distribution, AREA: int = 1000,
                                 CUTOFF: float = 0.5):
    b_c_d_df = between_conditions_distribution.copy()
    b_c_d_df["pvalues"] = b_c_d_df.parallel_apply(lambda x: pvalue_calc(x, replicates_distribution, AREA, CUTOFF),
                                                  axis=1)
    return b_c_d_df


def pvalue_calc(x, replicates_distribution, AREA, CUTOFF):
    # for index, row in between_conditions_distribution.iterrows():
    replicates_logtpms = replicates_distribution[:, [1]]
    between_cond_obs_logfc = abs(x[0])
    ev_logtpm = x[1]
    local_dist = get_local_distribution_np(ev_logtpm, replicates_distribution, replicates_logtpms, AREA)
    if -CUTOFF < between_cond_obs_logfc < CUTOFF:
        event_pval = 1.0
    else:
        event_pval = calculate_empirical_pvalue(local_dist[:, 0], between_cond_obs_logfc)
    return event_pval
