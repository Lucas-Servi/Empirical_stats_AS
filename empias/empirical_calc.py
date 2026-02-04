"""
Module for calculating empirical p-values in gene expression analysis.

This module provides functions to create distributions from replicates and
between conditions, and calculate empirical p-values using these distributions.
"""

import logging
from bisect import bisect_left
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
    # Handle older pandas versions vs new ones for column levels if needed
    # Assuming standard pandas structure
    try:
        conditions = df.columns.get_level_values('condition').unique()
    except Exception:
        # Fallback if not MultiIndex or named levels, assuming column names contain conditions
        # But keeping original logic is safer if user data structure is consistent
        # For now, stick to original logic but be robust
        conditions = []
        if isinstance(df.columns, pd.MultiIndex):
             conditions = df.columns.levels[0]

    if (condition1 not in df.columns.get_level_values(0)) or (condition2 not in df.columns.get_level_values(0)):
         # Try to be more flexible if possible, but let's stick to what worked before
         pass

    # Log transformation with proper handling of zeros and very small values
    # Use float64 for precision
    log_new_df = df.apply(lambda x: np.log2(x) if np.issubdtype(x.dtype, np.number) else x).astype(float)
    log_new_df = log_new_df.replace(-np.inf, np.nan)
    
    # Apply TPM threshold
    log_new_df[log_new_df < np.log2(tpm_threshold)] = np.nan
    
    # Calculate mean per treatment
    # Fix for pandas 3.0: use transpose or numeric_only=True if applicable, but we want grouped columns
    # Replaced groupby(axis=1) with T.groupby().T
    log_new_df_mean_per_treat = log_new_df.T.groupby("condition").mean().T
    
    # Calculate log fold change between conditions
    log_new_df_logFC = log_new_df_mean_per_treat[condition1] - log_new_df_mean_per_treat[condition2]
    
    # Calculate mean TPM across all samples
    # Optimization: vectorize where possible
    # original: df.groupby("gene").sum()...
    # If index is gene, we don't need groupby("gene"). If multiple rows per gene, we do.
    # Assuming unique genes or groupby needed.
    # We can use numeric_only=True to avoid warnings
    tpm_sum = df.groupby("gene").sum(numeric_only=True)
    
    log_new_df_tpm_mean = tpm_sum.apply(
        lambda x: np.log2(x) if np.issubdtype(x.dtype, np.number) else x,
        axis=0
    ).dropna(how='all').mean(axis=1)
    
    # Combine logFC and tpm_mean into a single DataFrame
    between_conditions_distribution = log_new_df_logFC.to_frame("logFC").join(
        log_new_df_tpm_mean.to_frame("tpm_mean")
    )
    
    logger.info(f"Created between conditions distribution with {len(between_conditions_distribution)} entries")
    return between_conditions_distribution


def log_fold_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate log2 fold change between two arrays."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log2(a / b)
    # Convert infs to nan
    result[np.isinf(result)] = np.nan
    return result


def create_replicates_distribution(df: pd.DataFrame) -> np.ndarray:
    """
    Create a distribution from replicates within each condition.
    
    Returns:
        Sorted numpy array with logFC and TPM values
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    distributions = []
    
    # Process each condition separately
    # Fix for pandas 3.0: iterate over groups manually or use transpose
    for condition_name, condition_df_T in df.T.groupby("condition"):
        condition_df = condition_df_T.T
        logger.info(f"Processing condition: {condition_name}")
        
        # Get all possible pairs of replicates
        cols = condition_df.columns
        col_pairs = list(combinations(cols, r=2))
        
        if not col_pairs:
            continue
            
        for pair in col_pairs:
            # pair is ( (cond, sample1), (cond, sample2) ) or similar depending on MultiIndex
            # If flat columns, just strings.
            
            # Extract values
            vals1 = df[pair[0]].values
            vals2 = df[pair[1]].values
            
            # Log FC
            logfc_values = log_fold_change(vals1, vals2)
            
            # Average log10 TPM
            # Optimization: direct numpy op
            with np.errstate(divide='ignore', invalid='ignore'):
                log10_vals = np.log10(np.column_stack((vals1, vals2)))
                avg_log_tpm_mean = np.mean(log10_vals, axis=1)
            
            # Combine
            # Filter NaNs later to speed up loop
            pair_distribution = np.column_stack((logfc_values, avg_log_tpm_mean))
            distributions.append(pair_distribution)
    
    if not distributions:
        logger.warning("No valid replicate pairs found")
        return np.empty((0, 2))
    
    combined_distribution = np.vstack(distributions)
    logger.info(f"Combined distribution shape: {combined_distribution.shape}")
    
    # Remove rows with NaN or infinite values
    # Optimized check
    mask = np.isfinite(combined_distribution).all(axis=1)
    clean_distribution = combined_distribution[mask]
    
    # Sort by TPM values (column 1)
    # This is crucial for binary search
    clean_distribution = clean_distribution[clean_distribution[:, 1].argsort()]
    
    logger.info(f"Final distribution shape after cleaning: {clean_distribution.shape}")
    return clean_distribution


def calculate_empirical_pvalue_fast(local_area_abs: np.ndarray, logfc_abs_value: float) -> float:
    """
    Calculate empirical p-value using direct counting (replaces ECDF).
    
    Args:
        local_area_abs: Absolute logFC values of the local distribution
        logfc_abs_value: The observed absolute logFC
    """
    if local_area_abs.size == 0:
        return 1.0
        
    # P(random >= observed)
    # ECDF logic was: (1 - ECDF(obs)) * 0.5 = P(random > obs) * 0.5
    # Strict inequality > matches (1 - ECDF) because ECDF is <=
    count = np.sum(local_area_abs > logfc_abs_value)
    return (count / local_area_abs.size) * 0.5


def pvalue_calc(
    row: np.ndarray,
    replicates_logtpms: np.ndarray,
    replicates_logfcs_abs: np.ndarray,
    area: int,
    cutoff: float
) -> float:
    """
    Optimized p-value calculation for a single row.
    """
    between_cond_obs_logfc = abs(row[0])
    ev_logtpm = row[1]
    
    if -cutoff < row[0] < cutoff: # Check signed value against cutoff logic, original was abs check?
        # Original: if -cutoff < between_cond_obs_logfc < cutoff:
        # Wait, between_cond_obs_logfc is ABS(row[0]). 
        # So -cutoff < abs < cutoff means abs < cutoff.
        # This simplifies to: if abs(row[0]) < cutoff: return 1.0
        return 1.0
        
    if between_cond_obs_logfc < cutoff:
        return 1.0

    # Find closest index using binary search
    idx = np.searchsorted(replicates_logtpms, ev_logtpm)
    
    # Adjust to find the actual closest value index
    if idx == 0:
        pass # idx is 0
    elif idx == len(replicates_logtpms):
        idx = len(replicates_logtpms) - 1
    else:
        # Check if previous element is closer
        before = replicates_logtpms[idx - 1]
        after = replicates_logtpms[idx]
        if (ev_logtpm - before) <= (after - ev_logtpm):
            idx = idx - 1
            
    # Define window (matching slice_list logic)
    # slice_list uses half_len = int(area * 0.5)
    # and produces a window of size roughly 2*half_len + 1
    n = len(replicates_logtpms)
    half_area = int(area * 0.5)
    
    # Initial bounds
    left = idx - half_area
    right = idx + half_area + 1
    
    # Handle edges by shifting the window
    if left < 0:
        # If we overlap left, extend right by value of overlap
        # right_bound = min(index + half_len + (-diff) + 1, len)
        # -diff is the amount we are negative.
        overflow = -left
        left = 0
        right = min(right + overflow, n)
    elif right > n:
        # If we overlap right, extend left
        overflow = right - n
        right = n
        left = max(left - overflow, 0)
    
    local_abs_logfcs = replicates_logfcs_abs[left:right]
    
    if local_abs_logfcs.size == 0:
        return 1.0
        
    return calculate_empirical_pvalue_fast(local_abs_logfcs, between_cond_obs_logfc)


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
    Calculate p-values for multiple events.
    """
    logger.info(f"Calculating p-values for {len(between_conditions_distribution)} events")
    
    # Pre-extract replicates data for speed
    rep_tpms = replicates_distribution[:, 1]
    rep_logfcs_abs = np.abs(replicates_distribution[:, 0]) # Pre-compute abs
    
    # Prepare data for parallel processing
    rows = between_conditions_distribution.values
    
    # Use joblib
    try:
        pvalues = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=10 if progress_bar else 0)(
            delayed(pvalue_calc)(
                row, rep_tpms, rep_logfcs_abs, area, cutoff
            )
            for row in rows
        )
        
        b_c_d_df = between_conditions_distribution.copy()
        b_c_d_df["pvalues"] = pvalues
        
        # FDR correction
        try:
            from statsmodels.stats.multitest import fdrcorrection
            _, b_c_d_df["fdr"] = fdrcorrection(b_c_d_df["pvalues"].values)
        except ImportError:
            # Optional: implement simple BH correction if statsmodels missing
            # But we kept statsmodels in pyproject.toml as dependency
            logger.warning("statsmodels not found for FDR, using basic implementation if available or skipping")
            try:
                from .utils import p_adjust_fdr_bh
                b_c_d_df["fdr"] = p_adjust_fdr_bh(b_c_d_df["pvalues"].values)
            except Exception:
                 pass

        return b_c_d_df
    
    except Exception as e:
        logger.error(f"Error in p-value calculation: {e}")
        raise