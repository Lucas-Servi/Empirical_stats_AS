import numpy as np
import pandas as pd
from math import inf
import scipy.stats as stats
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src import p_adjust_fdr_bh, log2FC_func


class MyClass:
    """SALMON Handler
    Labels should be ordered with the quant.sf files associated.
    Labels should finish in "_#", where # = sample number (or replicate)
    """

    def __init__(self):
        """
        Initialize main parameters
        """
        self.data = {}
        self.sample_names = []
        self.dfs = []
        self.big_df = pd.DataFrame
        # Should I create a new DF for LogFC, pvalues, etc?
        print("New instance created")

    def load_data(self, dfs, sample_names, f_format="tsv"):
        """
        Generates a python dictionary:
        self.data = {"[Treatment_#]":Pandas.Dataframe, ...}
        
        Generates a python list with treatment_replicate-number:
        self.labels = ["Treatment_#", ...]
        
        Input
        =====
        
        dfs = list of quant.sf files (full path)
        sample_names = list of each "[treatment label]" + "_[replicate number]" (Ordered)
        
        """
        self.dfs = [self.open_file(df, f_format) for df in dfs]
        self.sample_names = sample_names
        print(sample_names)
        self.data = dict(zip(self.sample_names, self.dfs))
        print("Data loaded")

    def open_file(self, file, f_format="tsv"):
        """
        Opens .tsv files (TAB-separated values tables)
        
        Input
        =====
        file = .tsv file full path
        
        Return
        ======
        Pandas DataFrame
        """
        if f_format == "parquet":
            df = pd.read_parquet(file)
        elif f_format == "csv":
            df = pd.read_csv(file)
        elif f_format == "tsv":
            df = pd.read_csv(file, sep='\t')
        else:
            df = pd.read_csv(file, sep='\t')
        return df

    def merge_replicates(self):
        """
        Sorts self.sample_names
        Merges DF from each sample from each treatment
        Creates a new dic replicates_dfs with a DataFrame with all
        replicates from each treatment
        
        """
        self.sample_names = sorted(self.sample_names, key=lambda x: int(x[-1]))
        sample_names = self.sample_names
        self.treatments = []
        self.replicates_dfs = {}
        i = 1
        for name in sample_names:
            n_samples = 1
            if name[-1] == "1":  # Does this for every 1st replicate
                current_name = name[:-2]
                self.treatments.append(current_name)
                df = self.data[name][['Name', 'TPM']].copy()
                for next_name in sample_names[i:]:
                    if next_name[:-2] == current_name:
                        new_df = self.data[next_name][['Name', 'TPM']].copy()
                        df = df.merge(new_df, how='outer', on='Name', validate='one_to_one')
                        n_samples += 1
                i += 1
                colname = ['isoform'] + ['sample_' + str(x) for x in range(1, n_samples + 1)]
                df.columns = colname
                df = df.set_index('isoform')
                self.replicates_dfs[current_name] = df

    def gene_isoform_join(self, transcriptome="AtRTDv2_QUASI.parquet"):
        """
        By default loads AtRTDv2_QUASI transcriptome
        Adds gene and isoform index to each treatment DataFrame
        """
        transcriptome = pd.read_parquet(transcriptome)
        for key, df in self.replicates_dfs.items():
            self.replicates_dfs[key] = pd.concat([transcriptome, df], axis=1).reset_index()
            self.replicates_dfs[key].set_index(["gene", "isoform"], inplace=True)

    def create_big_df(self):
        """
        Concatenates every treatment and index columns by Condition (treatment)
        and sample number as "sample_#" [# = sample number]
        """
        self.big_df = pd.DataFrame()
        self.big_df = pd.concat(self.replicates_dfs, axis=1)
        self.big_df.columns.set_names(["condition", "sample"], inplace=True)

    def DESeq2_normalization(self):
        """
        Applies DESeq2 normalization to the main DataFrame (big_df)
        """
        df_copy = self.big_df.copy
        log_df = np.log(df_copy)
        log_df["average"] = log_df.mean(axis=1, numeric_only=True)
        log_df_clean = log_df.drop(log_df[log_df.average == -inf].index)  # must import inf from math
        log_df_clean.iloc[:, :-1] = log_df_clean.iloc[:, :-1].subtract(log_df_clean['average'].values, axis=0)
        median = log_df_clean.iloc[:, :-1].median()
        scaling_factors = np.power(10, median.values)
        print("Scaling Factors")
        print(scaling_factors)
        df_copy = df_copy.multiply(scaling_factors, axis='columns')
        self.big_df = df_copy  # Overwrites big_df

    def filter_by_TPM(self, tpm=1, min_cols=2):
        self.big_df_tpm = self.big_df[self.big_df.ge(tpm).sum(1).ge(min_cols)]

    def welch_t_test(self, cond1, cond2):
        """
        Perform the Welch's test on two independent samples.
        Welch's test doesn't asume equal variance
        
        Creates a new column with both condition names and the associated p-values corrected 
        using Benjamini-Hochberg p-value correction (FDR)
        
        New colums are labeled as: "###-"+cond1+"-"+cond2 
        where ### is the new value ("pvalue", "fdr")
        """
        # welchs_t_test, only pvalue
        if f"pvalue-{cond1}-{cond2}" not in self.big_df_stats.columns:  # Avoids the duplicate creation of comparisons
            new_df = self.big_df_stats.loc[:, [cond1, cond2]]  # Pandas returns a copy
            df = new_df.dropna()
            _, df[f"pvalue-{cond1}-{cond2}"] = stats.ttest_ind(a=df.loc[:, cond1], b=df.loc[:, cond2],
                                                               alternative='two-sided', axis=1, equal_var=True)
            df2 = df.dropna()
            df2[f"fdr-{cond1}-{cond2}"] = p_adjust_fdr_bh(df2.loc[:, (f"pvalue-{cond1}-{cond2}")])

            # DataFrame update
            self.big_df_stats = self.big_df_stats.join(df2.iloc[:, -2:], on=["gene", "isoform"], lsuffix="_2")

    def welch_t_test_across_treatments(self):
        """
        Performs Welch's Test across multiple conditions,
        starting from the first vs all, second vs remaining, etc
        For example: cond1 vs cond3 will be tested, but not cond3 vs cond1.
        
        If comparisons not made are needed, welch_t_test(self, cond1, cond2)
        function can be called as needed and the DataFrame will be updated.
        """

        self.big_df_stats = self.big_df_tpm.copy(deep=True)
        conditions = self.treatments.copy()
        for i in range(len(conditions) - 1):
            cond1 = conditions.pop(0)
            for cond2 in conditions:
                print(f"Welch's test: {cond1} vs {cond2}")
                self.welch_t_test(cond1, cond2)

    def log2FC_across_treatments(self, corrected=True):
        """
        
        
        """
        df_log2FC = self.big_df_stats.copy
        conditions = self.treatments.copy

        for i in range(len(conditions) - 1):
            cond1 = conditions.pop(0)
            for cond2 in conditions:
                print(f"Log2FC: {cond1} vs {cond2}")
                df_log2FC[f"Log2FC-{cond1}-{cond2}"] = log2FC_func(self.big_df, cond1, cond2)
                if corrected:
                    df_log2FC[f"negative_log_pval-{cond1}-{cond2}"] = np.log10(df_log2FC[f"fdr-{cond1}-{cond2}"]) * (-1)
                else:
                    df_log2FC[f"negative_log_pval-{cond1}-{cond2}"] = np.log10(df_log2FC[f"pvalue-{cond1}-{cond2}"]) * (
                        -1)

        self.big_df_stats = df_log2FC  # DataFrame update

    """
    Plots
    =====
    """

    def correlation_pvalue_plot(self, cond1, cond2):
        df = self.big_df
        fig = px.scatter(df, x=df[cond1].mean(axis=1), y=df[cond2].mean(axis=1), trendline="ols")
        fig.show()

        # big_df_100 = big_df.loc[(big_df.T < 1000).all()]
        # use the function regplot to make a scatterplot
        # sns.regplot(x=big_df_100["C_B"].mean(axis=1), y=big_df_100["C_2"].mean(axis=1))

    def volcano_plot(self):
        df = self.big_df
        # Calculate log2FC and -logP values
        df['log2FC_6h'] = np.log2(df['ratio_6h'])
        df['log2FC_1d'] = np.log2(df['ratio_1d'])
        df['log2FC_3d'] = np.log2(df['ratio_3d'])
        df['log2FC_7d'] = np.log2(df['ratio_7d'])
        df['negative_log_pval_6h'] = np.log10(df['qval_6h']) * (-1)
        df['negative_log_pval_1d'] = np.log10(df['qval_1d']) * (-1)
        df['negative_log_pval_3d'] = np.log10(df['qval_3d']) * (-1)
        df['negative_log_pval_7d'] = np.log10(df['qval_7d']) * (-1)

        # Plot volcano plot
        fig = go.Figure()
        trace1 = go.Scatter(x=df['log2FC_6h'],
                            y=df['negative_log_pval_6h'],
                            mode='markers',
                            name='6hrs',
                            hovertext=list(df.index))
        trace2 = go.Scatter(x=df['log2FC_1d'],
                            y=df['negative_log_pval_1d'],
                            mode='markers',
                            name='day 1',
                            hovertext=list(df.index))
        trace3 = go.Scatter(x=df['log2FC_3d'],
                            y=df['negative_log_pval_3d'],
                            mode='markers',
                            name='day 3',
                            hovertext=list(df.index))
        trace4 = go.Scatter(x=df['log2FC_7d'],
                            y=df['negative_log_pval_7d'],
                            mode='markers',
                            name='day 7',
                            hovertext=list(df.index))
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.add_trace(trace4)
        fig.update_layout(title='Volcano plot for seronegatives')
        fig.show()

        # Plot volcano plot with text
        fig_d1 = px.scatter(df, x='log2FC_1d', y='negative_log_pval_1d', text=df.index)
        fig_d1.update_traces(textposition='top center')
        fig_d1.update_layout(title_text='Volcano plot for seronegatives (day 1)')
        fig_d1.show()
