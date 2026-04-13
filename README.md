# empiAS

**Emp**irical **I**soform **A**nalysis **S**oftware (v1.0.0)

A non-parametric Python package for detecting differential isoform usage (DIU) from transcript-level TPM quantifications. empiAS builds expression-aware null distributions from biological replicates and computes empirical p-values that account for the variance–mean relationship (heteroscedasticity) inherent in RNA-seq data.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- numpy, pandas, scipy, joblib, plotly, statsmodels

Pinned versions for reproducibility are provided in `requirements.txt`.

## Quick Start

For an interactive walkthrough with synthetic data, see **[`quickstart.ipynb`](quickstart.ipynb)** — no external datasets needed.

```python
import pandas as pd
from empias import calculate_empirical_pvalue, SalmonAnalysis

# Option 1: Load from SALMON quant.sf files via SalmonAnalysis
sa = SalmonAnalysis(salmon_dir="path/to/salmon/quant_folders",
                    transcriptome_csv="path/to/isoform_gene_map.csv")
sa.merge_replicates({
    "condition_A": ["sample_1", "sample_2", "sample_3"],
    "condition_B": ["sample_4", "sample_5", "sample_6"],
})
df = sa.data

# Option 2: Use any DataFrame with MultiIndex columns (condition, sample)
# df = pd.read_parquet("expression_matrix.parquet")

# Calculate empirical p-values
results = calculate_empirical_pvalue(
    df,
    condition1="condition_A",
    condition2="condition_B",
    area=1000,          # local window size (default)
    cutoff=0.5,         # minimum |logFC| to test (default)
    tpm_threshold=1.0,  # minimum TPM in at least one sample (default)
    n_workers=-1,       # all available cores (default)
)

print(results.head())
# Columns: gene, isoform, logFC, tpm_mean, pvalues, FDR_BH
```

## Statistical Method

empiAS uses a three-step non-parametric algorithm:

### 1. Null Distribution Construction

For every isoform passing the TPM threshold, log2 fold changes between all pairs of biological replicates within the same condition are computed:

$$
\Delta_{null} = \log_2\left(\frac{TPM_{rep_i}}{TPM_{rep_j}}\right)
$$

This captures the expected technical and biological noise in the absence of a true condition effect.

### 2. Local Background Estimation

Variance in RNA-seq data depends on expression level — low-expressed isoforms are noisier. For each test isoform at expression level *E* (log2 TPM), empiAS selects a local window of *N* nearest-neighbor events (default N = 1,000) from the null distribution by expression level. This is implemented via binary search on sorted arrays (O(log n) per query).

### 3. Empirical P-Value Calculation

The observed log2 fold change between conditions is compared against the local null window. The one-tailed empirical p-value is:

$$
P = \frac{1}{2} \times
\frac{\sum_{i=1}^{N} \mathbb{I}\left(|\Delta_{local,i}| \ge |\Delta_{obs}|\right)}{N}
$$

Where:
- **I** is the indicator function
- **Delta_obs** is the log2 fold change between conditions
- The 1/2 factor converts the magnitude-based (two-tail) count into a one-tailed p-value, given the symmetric null distribution

P-values are corrected for multiple testing using the Benjamini-Hochberg procedure (FDR).

## Input Format

The input DataFrame must have:
- **Rows**: isoform identifiers (index)
- **Columns**: MultiIndex with levels `(condition, sample)` containing TPM values

Example structure:

| | (cond_A, rep1) | (cond_A, rep2) | (cond_A, rep3) | (cond_B, rep1) | (cond_B, rep2) | (cond_B, rep3) |
|---|---|---|---|---|---|---|
| AT3G61860_ID2 | 12.5 | 14.2 | 11.8 | 32.1 | 28.7 | 35.4 |
| AT3G02300_ID3 | 3.1 | 2.8 | 3.5 | 18.2 | 15.6 | 20.1 |

## Output Format

| Column | Description |
|--------|-------------|
| `gene` | Gene identifier |
| `isoform` | Isoform identifier (index) |
| `logFC` | log2 fold change (condition2 / condition1) |
| `tpm_mean` | Mean TPM across all samples |
| `pvalues` | empiAS empirical p-value (one-tailed) |
| `FDR_BH` | Benjamini-Hochberg adjusted p-value |

## API Reference

### `calculate_empirical_pvalue(df, ...)`

Main entry point for differential isoform analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Expression matrix with MultiIndex columns |
| `condition1` | str | `"wt"` | Name of the reference condition |
| `condition2` | str | `"RS31OX"` | Name of the test condition |
| `area` | int | `1000` | Local window size (number of nearest neighbors) |
| `cutoff` | float | `0.5` | Minimum absolute logFC to compute a p-value; isoforms below this get p=1.0 |
| `tpm_threshold` | float | `1.0` | Minimum TPM in at least one sample to include an isoform |
| `n_workers` | int | `-1` | Number of parallel workers (-1 = all cores) |
| `progress_bar` | bool | `False` | Show progress bar during computation |

### `SalmonAnalysis(salmon_dir, transcriptome_csv)`

Helper class to load and merge SALMON `quant.sf` files into the required DataFrame format.

## Testing

```bash
pytest tests/ -v
```

## Citation

If you use empiAS in your research, please cite:

> Servi, L. *et al.* (2026). empiAS: a non-parametric framework for detecting differential isoform usage reveals nuclear-retained lncRNA-mRNA isoforms in *Arabidopsis thaliana*. *(Manuscript in preparation.)*

## License

MIT License. See [LICENSE](LICENSE) for details.
