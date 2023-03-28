import pandas as pd
from .empirical_calc import create_replicates_distribution, create_between_conditions_distribution,\
    calculate_events_pvals_multi

if __name__ is "__main__":
    print("The user interface is under construction")
    pass


def calculate_empirical_pvalue(df: pd.DataFrame, area: int=1000, cutoff: float=0.5,
                               tpm_threshold: int = 1, n_workers: int = 4, progress_bar: bool = False):

    pandarallel.initialize(progress_bar=progress_bar, nb_workers=n_workers)

    replicates_distribution = create_replicates_distribution(df)
    between_conditions_distribution = create_between_conditions_distribution(df, 1)
    pvals_df = calculate_events_pvals_multi(between_conditions_distribution,
                                 replicates_distribution, area,
                                 cutoff)
    return pvals_df

def test_print():
    print("FUNCIONO!")