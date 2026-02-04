
import sys
import os
import time
import numpy as np
import pandas as pd
import logging

# Add the parent directory to sys.path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from empias.empirical_calc import (
    create_replicates_distribution,
    create_between_conditions_distribution,
    calculate_events_pvals_multi
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dummy_data(n_genes=1000, n_samples=3):
    """Generates dummy gene expression data."""
    np.random.seed(42)
    
    conditions = ['wt', 'mutant']
    columns = []
    
    # Create columns with multi-index (condition, sample) similar to what seems to be expected
    # Based on code reading, input df seems to have MultiIndex columns or grouped by condition
    # Looking at create_replicates_distribution: df.groupby("condition", axis=1)
    # This implies columns should key into "condition" level.
    
    # Let's create a DataFrame with simple columns first, then create MultiIndex
    data = {}
    
    for cond in conditions:
        for i in range(1, n_samples + 1):
            col_name = f"{cond}_{i}"
            # Generate random counts
            counts = np.random.exponential(scale=100, size=n_genes)
            # Add some zeros
            counts[np.random.rand(n_genes) < 0.1] = 0
            data[(cond, col_name)] = counts
            
    df = pd.DataFrame(data)
    df.columns.names = ['condition', 'sample']
    df.index = [f"gene_{i}" for i in range(n_genes)]
    df.index.name = 'gene'
    
    return df

def run_benchmark():
    logger.info("Generating dummy data...")
    df = generate_dummy_data(n_genes=2000, n_samples=4)
    
    logger.info("Starting benchmark...")
    start_time = time.time()
    
    # 1. Create replicates distribution
    t0 = time.time()
    rep_dist = create_replicates_distribution(df)
    t1 = time.time()
    logger.info(f"Replicates distribution created in {t1 - t0:.4f}s. Shape: {rep_dist.shape}")
    
    # 2. Create between conditions distribution
    t2 = time.time()
    bet_cond_dist = create_between_conditions_distribution(
        df, tpm_threshold=1.0, condition1='wt', condition2='mutant'
    )
    t3 = time.time()
    logger.info(f"Between conditions distribution created in {t3 - t2:.4f}s. Shape: {bet_cond_dist.shape}")
    
    # 3. Calculate p-values
    t4 = time.time()
    result_df = calculate_events_pvals_multi(
        bet_cond_dist,
        rep_dist,
        area=1000,
        cutoff=0.1, # Low cutoff to ensure we actually calculate p-values for some genes
        n_jobs=-1,
        progress_bar=False
    )
    t5 = time.time()
    logger.info(f"P-values calculated in {t5 - t4:.4f}s. Rows: {len(result_df)}")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.4f}s")
    
    return result_df

if __name__ == "__main__":
    result = run_benchmark()
    
    # Save a summary for verification
    summary_path = os.path.join(os.path.dirname(__file__), 'benchmark_baseline.csv')
    result[['pvalues', 'fdr']].head(20).to_csv(summary_path)
    logger.info(f"Baseline results saved to {summary_path}")
