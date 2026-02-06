"""Tests for causal inference models and lift calculator."""

import numpy as np
import pandas as pd
import pytest

from utils.data_generator import generate_marketing_campaign_data, generate_ab_test_data
from models.causal_models import (
    propensity_score_matching,
    inverse_propensity_weighting,
    doubly_robust,
    s_learner,
    t_learner,
)
from models.lift_calculator import (
    compute_basic_lift,
    compute_segment_lift,
    compute_uplift_curve,
    compute_qini_curve,
    compute_lift_by_decile,
)


@pytest.fixture
def marketing_data():
    return generate_marketing_campaign_data(n_samples=1000, treatment_effect=0.10, seed=42)


@pytest.fixture
def ab_data():
    return generate_ab_test_data(n_samples=1000, lift_pct=15.0, seed=42)


@pytest.fixture
def covariates():
    return ["age", "income", "recency", "frequency", "monetary"]


# ── Data generation tests ────────────────────────────────────────────────────


class TestDataGeneration:
    def test_marketing_data_shape(self, marketing_data):
        assert marketing_data.shape[0] == 1000
        assert "treatment" in marketing_data.columns
        assert "conversion" in marketing_data.columns

    def test_marketing_data_treatment_values(self, marketing_data):
        assert set(marketing_data["treatment"].unique()).issubset({0, 1})

    def test_marketing_data_outcome_values(self, marketing_data):
        assert set(marketing_data["conversion"].unique()).issubset({0, 1})

    def test_ab_data_balanced(self, ab_data):
        treatment_rate = ab_data["treatment"].mean()
        assert 0.40 < treatment_rate < 0.60  # roughly 50/50

    def test_ab_data_has_positive_lift(self, ab_data):
        treated = ab_data[ab_data["treatment"] == 1]["conversion"].mean()
        control = ab_data[ab_data["treatment"] == 0]["conversion"].mean()
        assert treated > control


# ── Causal model tests ───────────────────────────────────────────────────────


class TestCausalModels:
    def test_psm_returns_estimate(self, marketing_data, covariates):
        est = propensity_score_matching(
            marketing_data, covariates, "treatment", "conversion"
        )
        assert est.method == "Propensity Score Matching"
        assert est.n_treated > 0
        assert est.ci_lower <= est.ate <= est.ci_upper

    def test_ipw_returns_estimate(self, marketing_data, covariates):
        est = inverse_propensity_weighting(
            marketing_data, covariates, "treatment", "conversion"
        )
        assert est.method == "Inverse Propensity Weighting"
        assert est.ate_se > 0

    def test_doubly_robust_returns_estimate(self, marketing_data, covariates):
        est = doubly_robust(
            marketing_data, covariates, "treatment", "conversion"
        )
        assert est.method == "Doubly Robust (AIPW)"
        assert est.ci_lower < est.ci_upper

    def test_s_learner_has_cate(self, marketing_data, covariates):
        est = s_learner(marketing_data, covariates, "treatment", "conversion")
        assert "cate" in est.details
        assert len(est.details["cate"]) == len(marketing_data)

    def test_t_learner_has_cate(self, marketing_data, covariates):
        est = t_learner(marketing_data, covariates, "treatment", "conversion")
        assert "cate" in est.details
        assert len(est.details["cate"]) == len(marketing_data)

    def test_estimates_in_reasonable_range(self, marketing_data, covariates):
        """All ATE estimates should be in a plausible range for this dataset."""
        for model_fn in [
            propensity_score_matching,
            inverse_propensity_weighting,
            doubly_robust,
            s_learner,
            t_learner,
        ]:
            est = model_fn(marketing_data, covariates, "treatment", "conversion")
            assert -0.5 < est.ate < 0.5, f"{est.method} ATE out of range: {est.ate}"


# ── Lift calculator tests ───────────────────────────────────────────────────


class TestLiftCalculator:
    def test_basic_lift(self, marketing_data):
        lr = compute_basic_lift(marketing_data, "treatment", "conversion")
        assert lr.n_treated + lr.n_control == len(marketing_data)
        assert 0 <= lr.treatment_rate <= 1
        assert 0 <= lr.control_rate <= 1

    def test_segment_lift(self, marketing_data):
        seg_df = compute_segment_lift(
            marketing_data, "treatment", "conversion", "channel"
        )
        assert not seg_df.empty
        assert "absolute_lift" in seg_df.columns

    def test_uplift_curve(self, marketing_data, covariates):
        est = t_learner(marketing_data, covariates, "treatment", "conversion")
        cate = est.details["cate"]
        curve = compute_uplift_curve(
            cate, marketing_data["treatment"].values, marketing_data["conversion"].values
        )
        assert "fraction_targeted" in curve.columns
        assert "cumulative_uplift" in curve.columns
        assert curve["fraction_targeted"].iloc[-1] <= 1.0

    def test_qini_curve(self, marketing_data, covariates):
        est = t_learner(marketing_data, covariates, "treatment", "conversion")
        cate = est.details["cate"]
        qini = compute_qini_curve(
            cate, marketing_data["treatment"].values, marketing_data["conversion"].values
        )
        assert "incremental_gains" in qini.columns
        assert qini["fraction_targeted"].iloc[0] == 0.0

    def test_lift_by_decile(self, marketing_data, covariates):
        est = t_learner(marketing_data, covariates, "treatment", "conversion")
        cate = est.details["cate"]
        dec_df = compute_lift_by_decile(
            cate, marketing_data["treatment"].values, marketing_data["conversion"].values
        )
        assert len(dec_df) > 0
        assert "observed_lift" in dec_df.columns
