"""
Module for calculating empirical p-values in gene expression analysis.

This module provides functions to create distributions from replicates and
between conditions, and calculate empirical p-values using these distributions.
"""

import logging

import numpy as np
import pandas as pd
from bisect import bisect_left
from itertools import combinations
from joblib import Parallel, delayed
from statsmodels.distributions.empirical_distribution import ECDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure numpy to handle specific warnings
np.seterr(divide='ignore', invalid="ignore")


def create_between_conditions_distribution(
    df: pd.DataFrame,
    tpm_threshold: float,
    condition1: str = "wt",
    condition2: str = "RS31OX"
) -> pd.DataFrame:
    """
    Create a distribution comparing gene expression between two conditions.
    
    Args:
        df: DataFrame containing gene expression data
        tpm_threshold: Threshold for Transcripts Per Million (TPM)
        condition1: First condition name (default: "wt")
        condition2: Second condition name (default: "RS31OX")
    
    Returns:
        DataFrame with logFC and tpm_mean for genes between conditions
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(tpm_threshold, (int, float)):
        raise TypeError("tpm_threshold must be a number")
    
    # Check if conditions exist in the DataFrame
    conditions = df.columns.get_level_values('condition').unique()
    if condition1 not in conditions or condition2 not in conditions:
        available = ", ".join(conditions)
        raise ValueError(f"Conditions {condition1} and/or {condition2} not found. Available: {available}")
    
    # Log transformation with proper handling of zeros and very small values
    log_new_df = df.apply(lambda x: np.log2(x) if np.issubdtype(x.dtype, np.number) else x)
    log_new_df = log_new_df.replace(-np.inf, np.nan)  # Change -inf to NaN to ignore them on the mean
    
    # Apply TPM threshold
    log_new_df[log_new_df < np.log2(tpm_threshold)] = np.nan
    
    # Calculate mean per treatment
    log_new_df_mean_per_treat = log_new_df.groupby("condition", axis=1).mean()
    
    # Calculate log fold change between conditions
    log_new_df_logFC = log_new_df_mean_per_treat[condition1] - log_new_df_mean_per_treat[condition2]
    
    # Calculate mean TPM across all samples
    log_new_df_tpm_mean = df.groupby("gene").sum().apply(
        lambda x: np.log2(x) if np.issubdtype(x.dtype, np.number) else x,
        axis=0
    ).dropna().mean(axis=1)
    
    # Combine logFC and tpm_mean into a single DataFrame
    between_conditions_distribution = log_new_df_logFC.to_frame("logFC").join(
        log_new_df_tpm_mean.to_frame("tpm_mean")
    )
    
    logger.info(f"Created between conditions distribution with {len(between_conditions_distribution)} entries")
    return between_conditions_distribution


def log_fold_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate log2 fold change between two arrays.
    
    Args:
        a: Numerator array
        b: Denominator array
    
    Returns:
        Array of log2 fold change values
    """
    # Handle division by zero and negative values
    valid_indices = (a > 0) & (b > 0)
    result = np.zeros_like(a, dtype=float)
    result[valid_indices] = np.log2(a[valid_indices] / b[valid_indices])
    result[~valid_indices] = np.nan
    return result


def avg_log_tpm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate mean of log10 values for two arrays.
    
    Args:
        a: First array
        b: Second array
    
    Returns:
        Array of mean log10 values
    """
    # Handle negative or zero values
    valid_indices = (a > 0) & (b > 0)
    result = np.zeros_like(a, dtype=float)
    result[valid_indices] = np.mean(
        np.log10([a[valid_indices], b[valid_indices]]), 
        axis=0
    )
    result[~valid_indices] = np.nan
    return result


def create_replicates_distribution(df: pd.DataFrame) -> np.ndarray:
    """
    Create a distribution from replicates within each condition.
    
    This function computes log fold changes between all pairs of replicates
    within each condition and combines them with average log TPM values.
    
    Args:
        df: DataFrame containing gene expression data
    
    Returns:
        Sorted numpy array with logFC and TPM values
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    distributions = []
    
    # Process each condition separately
    for condition_name, condition_df in df.groupby("condition", axis=1):
        logger.info(f"Processing condition: {condition_name}")
        
        # Get all possible pairs of replicates
        col_pairs = list(combinations(condition_df.columns, r=2))
        
        for pair in col_pairs:
            pair_name = f"{pair[0][0]}_{pair[0][1][-1:]}_{pair[1][1][-1:]}"
            
            # Calculate log fold change between replicates
            logfc_values = log_fold_change(
                df.loc[:, pair[0]].values,
                df.loc[:, pair[1]].values
            )
            
            # Calculate average log TPM
            sub_df = df.loc[:, [pair[0], pair[1]]]
            avg_log_tpm_mean = sub_df.groupby("gene").sum().apply(
                lambda fc: np.log10(fc) if np.issubdtype(fc.dtype, np.number) else fc,
                axis=0
            ).dropna().mean(axis=1)
            
            # Combine logFC and TPM values
            pair_distribution = np.column_stack((
                logfc_values,
                avg_log_tpm_mean.values
            ))
            
            distributions.append(pair_distribution)
            logger.debug(f"Added distribution for {pair_name} with shape {pair_distribution.shape}")
    
    # Combine all distributions
    if not distributions:
        logger.warning("No valid replicate pairs found")
        return np.empty((0, 2))
    
    combined_distribution = np.vstack(distributions)
    logger.info(f"Combined distribution shape: {combined_distribution.shape}")
    
    # Sort by TPM values (column 1)
    sorted_distribution = combined_distribution[combined_distribution[:, 1].argsort()]
    
    # Remove rows with NaN or infinite values
    valid_rows = ~(np.isnan(sorted_distribution).any(axis=1) | 
                  np.isinf(sorted_distribution).any(axis=1))
    clean_distribution = sorted_distribution[valid_rows]
    
    logger.info(f"Final distribution shape after cleaning: {clean_distribution.shape}")
    return clean_distribution


def slice_list(lst: list, index: int, slice_len: int) -> list:
    """
    Extract a slice from a list centered around an index with given length.
    
    Args:
        lst: The list to slice
        index: The center index of the slice
        slice_len: The desired length of the slice
    
    Returns:
        A slice of the list centered around index
    """
    if not lst:
        return []
    
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


def get_closest_number(lst: list[float], n: float) -> float:
    """
    Find the closest value to n in a sorted list.
    
    Args:
        lst: A sorted list of numbers
        n: The target number
    
    Returns:
        The value in lst that is closest to n
    """
    if not lst:
        raise ValueError("List cannot be empty")
    
    # Handle special cases
    if n <= lst[0]:
        return lst[0]
    if n >= lst[-1]:
        return lst[-1]
    
    # Find insertion point
    pos = bisect_left(lst, n)
    
    # Compare distances
    before = lst[pos - 1]
    after = lst[pos]
    
    return after if after - n < n - before else before


def get_local_distribution(
    ev_logtpm: float,
    replicates_distribution: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Get a local distribution based on a log TPM value.
    
    Args:
        ev_logtpm: The log TPM value
        replicates_distribution: The distribution of replicates
        window_size: The size of the window to extract
    
    Returns:
        A local distribution slice
    """
    if replicates_distribution.size == 0:
        logger.warning("Empty replicates distribution")
        return np.empty((0, 2))
    
    # Extract TPM values (column 1)
    replicates_logtpms = replicates_distribution[:, 1]
    
    # Find the closest log TPM value
    try:
        close_rep_logtpm = get_closest_number(replicates_logtpms.tolist(), ev_logtpm)
    except ValueError as e:
        logger.error(f"Error finding closest number: {e}")
        return np.empty((0, 2))
    
    # Find the index of this value
    indices = np.where(replicates_logtpms == close_rep_logtpm)[0]
    if len(indices) == 0:
        logger.warning(f"No matching TPM value found for {ev_logtpm}")
        return np.empty((0, 2))
    
    index = indices[0]
    
    # Get the local distribution
    local_dist = slice_list(replicates_distribution.tolist(), index, window_size)
    
    return np.array(local_dist)


def calculate_empirical_pvalue(local_area: np.ndarray, logfc_abs_value: float) -> float:
    """
    Calculate an empirical p-value using ECDF.
    
    Args:
        local_area: Local area values
        logfc_abs_value: Absolute log fold change value
    
    Returns:
        An empirical p-value
    """
    if local_area.size == 0:
        logger.warning("Empty local area, returning p-value of 1.0")
        return 1.0
    
    # Use numpy for efficiency
    abs_local_area = np.abs(local_area)
    
    # Create ECDF
    try:
        ecdf = ECDF(abs_local_area)
        # It is divided by 2 because we are using abs(LogFC) values - one-tailed test
        event_pvalue = (1.0 - ecdf(logfc_abs_value)) * 0.5
        return float(event_pvalue)
    except Exception as e:
        logger.error(f"Error calculating ECDF: {e}")
        return 1.0


def pvalue_calc(
    row: np.ndarray,
    replicates_distribution: np.ndarray,
    area: int,
    cutoff: float
) -> float:
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
    
    # Apply cutoff filter for efficiency
    if -cutoff < between_cond_obs_logfc < cutoff:
        return 1.0
    
    # Get local distribution and calculate p-value
    local_dist = get_local_distribution(ev_logtpm, replicates_distribution, area)
    
    if local_dist.size == 0:
        return 1.0
    
    # Extract logFC values (column 0)
    local_logfc = local_dist[:, 0]
    event_pval = calculate_empirical_pvalue(local_logfc, between_cond_obs_logfc)
    
    return event_pval


def calculate_events_pvals_multi(
    between_conditions_distribution: pd.DataFrame,
    replicates_distribution: np.ndarray,
    area: int = 1000,
    cutoff: float = 0.5,
    n_jobs: int = -1,
    batch_size: int = 100,
    progress_bar: bool = False
) -> pd.DataFrame:
    """
    Calculate p-values for multiple events using joblib parallelization.
    
    Args:
        between_conditions_distribution: DataFrame of between conditions distribution
        replicates_distribution: The distribution of replicates
        area: The area size for local distribution (default: 1000)
        cutoff: Cutoff value for log fold change (default: 0.5)
        n_jobs: Number of parallel jobs (-1 uses all available cores)
        batch_size: Size of batches for parallel processing
        progress_bar: Whether to show a progress bar
    
    Returns:
        DataFrame with calculated p-values
    """
    # Input validation
    if not isinstance(between_conditions_distribution, pd.DataFrame):
        raise TypeError("between_conditions_distribution must be a pandas DataFrame")
    if not isinstance(replicates_distribution, np.ndarray):
        raise TypeError("replicates_distribution must be a numpy array")
    
    logger.info(f"Calculating p-values for {len(between_conditions_distribution)} events")
    
    # Create a copy to avoid modifying the original DataFrame
    b_c_d_df = between_conditions_distribution.copy()
    
    # Extract rows for parallel processing
    rows = [row.values for _, row in b_c_d_df.iterrows()]
    
    # Use joblib for parallel processing
    try:
        pvalues = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=10 if progress_bar else 0)(
            delayed(pvalue_calc)(row, replicates_distribution, area, cutoff)
            for row in rows
        )
        
        # Add results back to DataFrame
        b_c_d_df["pvalues"] = pvalues
        
        # Calculate FDR-adjusted p-values
        try:
            from statsmodels.stats.multitest import fdrcorrection
            _, b_c_d_df["fdr"] = fdrcorrection(b_c_d_df["pvalues"].values)
            logger.info("FDR correction applied successfully")
        except ImportError:
            logger.warning("statsmodels.stats.multitest not available, skipping FDR correction")
        except Exception as e:
            logger.error(f"Error applying FDR correction: {e}")
        
        logger.info(f"P-values calculated successfully for {len(b_c_d_df)} events")
        return b_c_d_df
    
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise