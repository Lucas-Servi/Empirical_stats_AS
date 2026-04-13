import numpy as np
import pandas as pd
from math import inf
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from .utils import p_adjust_fdr_bh, log2FC_func


class SalmonAnalysis:
    """
    Load and analyse SALMON transcript quantification output.

    Expected input: one ``quant.sf`` (or equivalent) file per sample,
    with columns ``Name`` (transcript ID) and ``TPM``.  Sample names must
    follow the ``treatment_N`` convention (e.g. ``wt_1``, ``wt_2``,
    ``RS31OX_1``) so that replicates can be grouped automatically.

    Typical workflow::

        sa = SalmonAnalysis()
        sa.load_data(file_paths, sample_names)
        sa.merge_replicates()
        sa.gene_isoform_join(transcriptome="AtRTDv2_1_QUASI.LS.parquet")
        sa.create_big_df()
        sa.DESeq2_normalization()
        sa.filter_by_TPM()
        sa.welch_t_test_across_treatments()
        sa.log2FC_across_treatments()
    """

    def __init__(self):
        """
        Initialise an empty SalmonAnalysis instance.

        Attributes:
            data (dict): Maps sample name → raw DataFrame (set by load_data).
            sample_names (list[str]): Ordered list of sample names.
            dfs (list[pd.DataFrame]): Raw DataFrames in load order.
            big_df (pd.DataFrame): Multi-condition DataFrame with a
                (condition, sample) column MultiIndex, built by create_big_df.
            big_df_stats (pd.DataFrame): TPM-filtered copy of big_df with
                appended p-value and log2FC columns.
            treatments (list[str]): Unique condition names derived from
                sample_names (e.g. ``["wt", "RS31OX"]``).
            replicates_dfs (dict): Maps condition name → merged replicate
                DataFrame (transcript × replicate columns).
        """
        self.data = {}
        self.sample_names = []
        self.dfs = []
        self.big_df = pd.DataFrame()
        self.big_df_stats = pd.DataFrame()
        self.treatments = []
        self.replicates_dfs = {}

    def load_data(self, dfs, sample_names, f_format="tsv"):
        self.dfs = [self.open_file(df, f_format) for df in dfs]
        self.sample_names = sample_names
        self.data = dict(zip(self.sample_names, self.dfs))

    def open_file(self, file, f_format="tsv"):
        if f_format == "parquet":
            return pd.read_parquet(file)
        sep = ',' if f_format == "csv" else '\t'
        return pd.read_csv(file, sep=sep)

    def merge_replicates(self):
        """
        Group samples by condition and merge their TPM columns.

        Sample names must follow the ``treatment_N`` convention, where the
        trailing ``_N`` suffix (integer) identifies the replicate number.
        For example ``wt_1``, ``wt_2``, ``RS31OX_1``, ``RS31OX_2``.

        After calling this method ``self.replicates_dfs`` maps each condition
        name to a DataFrame indexed by transcript ``Name`` with columns
        ``sample_1``, ``sample_2``, … (one per replicate).

        Raises:
            ValueError: If any sample name does not end with an integer suffix.
        """
        sorted_names = sorted(self.sample_names, key=lambda x: int(x.split('_')[-1]))
        self.treatments = sorted({name.rsplit('_', 1)[0] for name in sorted_names})

        for treatment in self.treatments:
            replicate_dfs = [self.data[name][['Name', 'TPM']].rename(columns={'TPM': f'sample_{i+1}'})
                             for i, name in enumerate(sorted_names) if name.startswith(treatment)]

            merged_df = replicate_dfs[0]
            for df in replicate_dfs[1:]:
                merged_df = merged_df.merge(df, on='Name', how='outer', validate='one_to_one')

            merged_df.set_index('Name', inplace=True)
            self.replicates_dfs[treatment] = merged_df

    def gene_isoform_join(self, transcriptome=None):
        """
        Annotate each isoform with its parent gene from the reference transcriptome.

        Args:
            transcriptome: Path to a Parquet file with at least an ``isoform``
                column mapping transcript IDs to ``gene`` identifiers
                (e.g. ``AtRTDv2_1_QUASI.LS.parquet``).  No default — must be
                supplied explicitly.

        Raises:
            ValueError: If *transcriptome* is not provided.
        """
        if transcriptome is None:
            raise ValueError(
                "transcriptome path must be provided. "
                "Pass the path to the AtRTDv2 Parquet file, e.g. "
                "gene_isoform_join(transcriptome='AtRTDv2_1_QUASI.LS.parquet')."
            )
        transcriptome_df = pd.read_parquet(transcriptome)
        for treatment, df in self.replicates_dfs.items():
            merged = transcriptome_df.set_index('isoform').join(df, how='inner')
            merged.reset_index().set_index(['gene', 'isoform'], inplace=True)
            self.replicates_dfs[treatment] = merged

    def create_big_df(self):
        self.big_df = pd.concat(self.replicates_dfs, axis=1)
        self.big_df.columns.set_names(["condition", "sample"], inplace=True)

    def DESeq2_normalization(self):
        log_df = np.log(self.big_df.replace(0, np.nan)).dropna()
        log_df_mean = log_df.mean(axis=1)
        log_df_centered = log_df.subtract(log_df_mean, axis=0)
        scaling_factors = np.exp(log_df_centered.median())
        self.big_df = self.big_df.multiply(scaling_factors, axis='columns')

    def filter_by_TPM(self, tpm=1, min_cols=2):
        self.big_df_stats = self.big_df[self.big_df.ge(tpm).sum(axis=1).ge(min_cols)]

    def welch_t_test(self, cond1, cond2):
        col_pvalue = f"pvalue-{cond1}-{cond2}"
        col_fdr = f"fdr-{cond1}-{cond2}"

        if col_pvalue not in self.big_df_stats:
            df = self.big_df_stats[[cond1, cond2]].dropna()
            _, pvalues = stats.ttest_ind(df[cond1], df[cond2], axis=1, equal_var=False)
            df[col_pvalue] = pvalues
            df[col_fdr] = p_adjust_fdr_bh(pvalues)
            self.big_df_stats = self.big_df_stats.join(df[[col_pvalue, col_fdr]])

    def welch_t_test_across_treatments(self):
        conditions = self.treatments[:]
        for i, cond1 in enumerate(conditions[:-1]):
            for cond2 in conditions[i+1:]:
                self.welch_t_test(cond1, cond2)

    def log2FC_across_treatments(self, corrected=True):
        for i, cond1 in enumerate(self.treatments[:-1]):
            for cond2 in self.treatments[i+1:]:
                log2fc_col = f"Log2FC-{cond1}-{cond2}"
                pval_col = f"fdr-{cond1}-{cond2}" if corrected else f"pvalue-{cond1}-{cond2}"
                neg_log_pval_col = f"negative_log_pval-{cond1}-{cond2}"

                self.big_df_stats[log2fc_col] = log2FC_func(self.big_df, cond1, cond2)
                self.big_df_stats[neg_log_pval_col] = -np.log10(self.big_df_stats[pval_col])

    def correlation_pvalue_plot(self, cond1, cond2):
        df = self.big_df
        fig = px.scatter(df, x=df[cond1].mean(axis=1), y=df[cond2].mean(axis=1), trendline="ols")
        fig.show()

    def volcano_plot(self, comparisons):
        fig = go.Figure()
        for label, ratio_col, qval_col in comparisons:
            # ratio_col already contains log2FC values (set by log2FC_across_treatments)
            log2fc = self.big_df_stats[ratio_col]
            neg_log_pval = -np.log10(self.big_df_stats[qval_col])
            fig.add_trace(go.Scatter(x=log2fc, y=neg_log_pval, mode='markers', name=label))
        fig.update_layout(title='Volcano plot')
        fig.show()
