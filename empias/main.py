"""
Main module for the SALMON analysis package.
"""

import logging
import pandas as pd

# Import empirical calculation functions
from empirical_calc import (
    create_replicates_distribution,
    create_between_conditions_distribution,
    calculate_events_pvals_multi
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_empirical_pvalue(
    df: pd.DataFrame,
    area: int = 1000,
    cutoff: float = 0.5,
    tpm_threshold: float = 1.0,
    n_workers: int = 4,
    progress_bar: bool = False,
    condition1: str = "wt",
    condition2: str = "RS31OX"
) -> pd.DataFrame:
    """
    Calculate empirical p-values for differential expression analysis.
    
    Args:
        df: DataFrame containing gene expression data
        area: The area size for local distribution (default: 1000)
        cutoff: Cutoff value for log fold change (default: 0.5)
        tpm_threshold: Threshold for Transcripts Per Million (default: 1.0)
        n_workers: Number of parallel workers (default: 4)
        progress_bar: Whether to show a progress bar (default: False)
        condition1: First condition name (default: "wt")
        condition2: Second condition name (default: "RS31OX")
    
    Returns:
        DataFrame with calculated p-values
    """
    logger.info("Starting empirical p-value calculation")
    
    try:
        # Create distributions
        logger.info("Creating replicates distribution")
        replicates_distribution = create_replicates_distribution(df)
        
        logger.info("Creating between conditions distribution")
        between_conditions_distribution = create_between_conditions_distribution(
            df, tpm_threshold, condition1, condition2
        )
        
        # Calculate p-values
        logger.info("Calculating p-values")
        pvals_df = calculate_events_pvals_multi(
            between_conditions_distribution,
            replicates_distribution,
            area=area,
            cutoff=cutoff,
            n_jobs=n_workers,
            progress_bar=progress_bar
        )
        
        logger.info("P-value calculation completed successfully")
        return pvals_df
    
    except Exception as e:
        logger.error(f"Error in empirical p-value calculation: {e}")
        raise


def test_print() -> None:
    """Simple test function to verify the package is working."""
    print("Package is properly installed!")


if __name__ == "__main__":
    print("The user interface is under construction")