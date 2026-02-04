# Empias

**Emp**irical **I**sofom **A**nalysis **S**oftware.

A Python package to quantify alternative splicing isoforms logFC expression per isoform per gene using empirical p-value calculation.

## Installation

```bash
pip install .
```

## Usage

```python
import pandas as pd
from empias import calculate_empirical_pvalue

# Load your data
# Data should be a DataFrame with MultiIndex columns (condition, sample) or specific format
df = pd.read_csv("expression_data.csv", index_col=0)

# Calculate p-values
results = calculate_empirical_pvalue(
    df,
    condition1="wt",
    condition2="mutant",
    n_workers=4
)

print(results.head())
```


## Statistical Methods

This package evaluates the significance of differential expression/splicing using a non-parametric empirical approach that accounts for expression-level dependent variability (heteroscedasticity).

---

### 1. Null Distribution Construction

We approximate the null hypothesis (no biological difference) by calculating the Log2 Fold Changes (LogFC) between all pairs of biological replicates within the same condition. This captures the expected technical and biological noise.

$$
\Delta_{null} = \log_2\left(\frac{TPM_{rep_i}}{TPM_{rep_j}}\right)
$$

---

### 2. Local Background Estimation

Variance typically depends on expression abundance (e.g., low-expressed genes are noisier).  
For each query event with expression level $E$ (log TPM), we dynamically select a **local window** of $N$ (default = 1000) events from the null distribution that have the closest expression levels to $E$.

---

### 3. Empirical P-Value Calculation

The p-value is derived by comparing the observed change to the local background noise distribution.  
We calculate the fraction of background events with an absolute LogFC greater than or equal to the observed absolute LogFC.

$$
P = \frac{1}{2} \times
\frac{\sum_{i=1}^{N} \mathbb{I}\left(|\Delta_{local,i}| \ge |\Delta_{obs}|\right)}{N}
$$

Where:

- $\mathbb{I}$ is the indicator function  
- $\Delta_{obs}$ is the LogFC between the two conditions being compared  
- The factor $\frac{1}{2}$ converts the two-tailed probability (magnitude check) into a one-tailed p-value, assuming the noise distribution is symmetric


## Features

- **Empirical P-Value Calculation**: Uses a background distribution of replicates to estimate significance.
- **Local Distribution**: Compares events to genes with similar expression levels (TPM).
- **Parallel Processing**: multi-core support for fast calculation.
