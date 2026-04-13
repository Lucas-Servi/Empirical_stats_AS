"""
Unit tests for empias.empirical_calc.

Run with:  pytest tests/test_empirical_calc.py -v
"""
import numpy as np
import pandas as pd
import pytest

from empias.empirical_calc import (
    calculate_empirical_pvalue_fast,
    create_between_conditions_distribution,
    create_replicates_distribution,
    pvalue_calc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_big_df(n_genes=50, rng=None):
    """Return a minimal MultiIndex DataFrame matching what SalmonAnalysis produces."""
    if rng is None:
        rng = np.random.default_rng(42)
    conditions = ["wt", "RS31OX"]
    replicates = ["sample_1", "sample_2", "sample_3"]
    cols = pd.MultiIndex.from_product([conditions, replicates], names=["condition", "sample"])
    data = rng.exponential(scale=10, size=(n_genes, len(cols))) + 0.01
    genes = [f"AT1G{i:05d}" for i in range(n_genes)]
    df = pd.DataFrame(data, index=pd.Index(genes, name="gene"), columns=cols)
    return df


def _make_replicates_dist(n=500, rng=None):
    """Small synthetic null distribution: sorted by log2 TPM column."""
    if rng is None:
        rng = np.random.default_rng(0)
    logfc = rng.normal(0, 0.3, n)
    logtpm = np.sort(rng.uniform(0, 10, n))
    return np.column_stack((logfc, logtpm))


# ---------------------------------------------------------------------------
# create_replicates_distribution
# ---------------------------------------------------------------------------

class TestCreateReplicatesDistribution:
    def test_returns_array_with_two_columns(self):
        df = _make_big_df()
        dist = create_replicates_distribution(df)
        assert dist.ndim == 2
        assert dist.shape[1] == 2

    def test_sorted_by_logtpm(self):
        df = _make_big_df()
        dist = create_replicates_distribution(df)
        assert np.all(np.diff(dist[:, 1]) >= 0), "Distribution must be sorted by log2 TPM"

    def test_no_nan_or_inf(self):
        df = _make_big_df()
        dist = create_replicates_distribution(df)
        assert np.all(np.isfinite(dist))

    def test_empty_dataframe_returns_empty_array(self):
        conditions = ["wt", "RS31OX"]
        replicates = ["sample_1", "sample_2"]
        cols = pd.MultiIndex.from_product([conditions, replicates], names=["condition", "sample"])
        df = pd.DataFrame(np.empty((0, len(cols))), columns=cols)
        dist = create_replicates_distribution(df)
        assert dist.shape == (0, 2)

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            create_replicates_distribution([[1, 2], [3, 4]])

    def test_single_replicate_per_condition_produces_empty(self):
        """With only one replicate per condition there are no pairs — result must be empty."""
        conditions = ["wt", "RS31OX"]
        cols = pd.MultiIndex.from_tuples(
            [(c, "sample_1") for c in conditions], names=["condition", "sample"]
        )
        df = pd.DataFrame(
            np.ones((10, 2)) * 5,
            index=[f"AT1G{i:05d}" for i in range(10)],
            columns=cols,
        )
        dist = create_replicates_distribution(df)
        assert dist.shape[0] == 0


# ---------------------------------------------------------------------------
# create_between_conditions_distribution
# ---------------------------------------------------------------------------

class TestCreateBetweenConditionsDistribution:
    def test_returns_dataframe_with_logfc_and_tpm_mean(self):
        df = _make_big_df()
        result = create_between_conditions_distribution(df, tpm_threshold=0.5)
        assert isinstance(result, pd.DataFrame)
        assert "logFC" in result.columns
        assert "tpm_mean" in result.columns

    def test_raises_when_condition_missing(self):
        df = _make_big_df()
        with pytest.raises(ValueError, match="not found"):
            create_between_conditions_distribution(df, tpm_threshold=0.5, condition1="wt", condition2="NONEXISTENT")

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            create_between_conditions_distribution("not_a_df", tpm_threshold=1.0)

    def test_tpm_threshold_filters_low_expression(self):
        df = _make_big_df()
        result_loose = create_between_conditions_distribution(df, tpm_threshold=0.001)
        result_strict = create_between_conditions_distribution(df, tpm_threshold=1000.0)
        # Strict threshold removes more rows
        assert len(result_strict.dropna()) <= len(result_loose.dropna())


# ---------------------------------------------------------------------------
# pvalue_calc
# ---------------------------------------------------------------------------

class TestPvalueCalc:
    def setup_method(self):
        self.dist = _make_replicates_dist(n=1000)
        self.rep_tpms = self.dist[:, 1]
        self.rep_logfcs_abs = np.abs(self.dist[:, 0])

    def test_below_cutoff_returns_one(self):
        row = np.array([0.1, 5.0])  # |logFC| = 0.1 < cutoff 0.5
        p = pvalue_calc(row, self.rep_tpms, self.rep_logfcs_abs, area=100, cutoff=0.5)
        assert p == 1.0

    def test_large_logfc_returns_small_pvalue(self):
        row = np.array([10.0, 5.0])  # extreme fold change
        p = pvalue_calc(row, self.rep_tpms, self.rep_logfcs_abs, area=100, cutoff=0.5)
        assert p < 0.05

    def test_pvalue_in_valid_range(self):
        row = np.array([2.0, 5.0])
        p = pvalue_calc(row, self.rep_tpms, self.rep_logfcs_abs, area=100, cutoff=0.5)
        assert 0.0 <= p <= 1.0

    def test_empty_distribution_returns_one(self):
        row = np.array([2.0, 5.0])
        p = pvalue_calc(row, np.array([]), np.array([]), area=100, cutoff=0.5)
        assert p == 1.0

    def test_negative_logfc_treated_as_absolute(self):
        """p-value should be the same for +logFC and -logFC of the same magnitude."""
        row_pos = np.array([2.0, 5.0])
        row_neg = np.array([-2.0, 5.0])
        p_pos = pvalue_calc(row_pos, self.rep_tpms, self.rep_logfcs_abs, area=100, cutoff=0.5)
        p_neg = pvalue_calc(row_neg, self.rep_tpms, self.rep_logfcs_abs, area=100, cutoff=0.5)
        assert p_pos == p_neg


# ---------------------------------------------------------------------------
# calculate_empirical_pvalue_fast
# ---------------------------------------------------------------------------

class TestCalculateEmpiricalPvalueFast:
    def test_all_smaller_returns_zero(self):
        local = np.array([0.1, 0.2, 0.3])
        p = calculate_empirical_pvalue_fast(local, 5.0)
        assert p == 0.0

    def test_all_larger_returns_half(self):
        local = np.array([10.0, 20.0, 30.0])
        p = calculate_empirical_pvalue_fast(local, 1.0)
        assert p == pytest.approx(0.5)

    def test_empty_local_returns_one(self):
        p = calculate_empirical_pvalue_fast(np.array([]), 1.0)
        assert p == 1.0


# ---------------------------------------------------------------------------
# FDR correction (utils.p_adjust_fdr_bh)
# ---------------------------------------------------------------------------
class TestFDRCorrection:
    """Verify Benjamini-Hochberg FDR correction matches statsmodels."""

    def test_bh_matches_statsmodels(self):
        from empias.utils import p_adjust_fdr_bh
        from statsmodels.stats.multitest import fdrcorrection

        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
        _, expected = fdrcorrection(p)
        result = p_adjust_fdr_bh(p)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_bh_monotonicity(self):
        from empias.utils import p_adjust_fdr_bh

        p = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0])
        result = p_adjust_fdr_bh(p)
        # Sorted p-values should give monotonically non-decreasing FDR
        sorted_idx = np.argsort(p)
        assert all(result[sorted_idx][i] <= result[sorted_idx][i + 1]
                    for i in range(len(result) - 1))

    def test_bh_clipped_to_one(self):
        from empias.utils import p_adjust_fdr_bh

        p = np.array([0.8, 0.9, 0.95, 1.0])
        result = p_adjust_fdr_bh(p)
        assert all(result <= 1.0)
