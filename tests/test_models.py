"""Tests for causal models and data generation."""

import pytest
import numpy as np
import pandas as pd

from utils.data_generator import generate_outreach_data, filter_by_combo, VALID_COMBOS
from models.causal_models import (
    propensity_score_matching,
    inverse_propensity_weighting,
    doubly_robust,
    s_learner,
    t_learner,
    CausalEstimate,
    CATEResult,
)
from models.lift_calculator import (
    compute_naive_lift,
    compute_segment_lift,
    compute_uplift_curve,
    compute_qini_curve,
    compute_lift_by_decile,
)


@pytest.fixture
def outreach_data():
    return generate_outreach_data(n_reps=30, n_hcps=100, n_records=1000, seed=42)


@pytest.fixture
def covariates():
    return [
        "Rep_Experience_Years",
        "Territory_Size",
        "HCP_Specialty",
        "HCP_Patient_Volume",
        "HCP_Digital_Affinity",
    ]


# ── Data generation tests ───────────────────────────────────────────────────

class TestDataGeneration:
    def test_shape(self, outreach_data):
        assert outreach_data.shape[0] == 1000
        expected_cols = [
            "Rep_ID", "HCP_ID", "Suggestion_Type", "Action_Type",
            "Suggestion_Accepted", "Incremental_Actions",
            "Incremental_TRX", "Incremental_NBRX",
        ]
        for col in expected_cols:
            assert col in outreach_data.columns

    def test_suggestion_types(self, outreach_data):
        valid = {"Email", "Call", "Insights", "All"}
        assert set(outreach_data["Suggestion_Type"].unique()).issubset(valid)

    def test_action_types(self, outreach_data):
        valid = {"Email", "Call", "Insights", "All"}
        assert set(outreach_data["Action_Type"].unique()).issubset(valid)

    def test_binary_treatment(self, outreach_data):
        assert set(outreach_data["Suggestion_Accepted"].unique()).issubset({0, 1})

    def test_filter_by_combo(self, outreach_data):
        filtered = filter_by_combo(outreach_data, "Email", "Email")
        assert all(filtered["Suggestion_Type"] == "Email")
        assert all(filtered["Action_Type"] == "Email")

    def test_filter_all_all(self, outreach_data):
        filtered = filter_by_combo(outreach_data, "All", "All")
        assert len(filtered) == len(outreach_data)


# ── Causal model tests ──────────────────────────────────────────────────────

class TestCausalModels:
    def test_psm(self, outreach_data, covariates):
        est = propensity_score_matching(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        assert isinstance(est, CausalEstimate)
        assert est.ci_lower <= est.ate <= est.ci_upper
        assert est.ate_se >= 0

    def test_ipw(self, outreach_data, covariates):
        est = inverse_propensity_weighting(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates,
            n_bootstrap=50,
        )
        assert isinstance(est, CausalEstimate)
        assert est.ate_se > 0

    def test_doubly_robust(self, outreach_data, covariates):
        est = doubly_robust(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        assert isinstance(est, CausalEstimate)
        assert est.ci_lower <= est.ate <= est.ci_upper

    def test_s_learner(self, outreach_data, covariates):
        est, cate = s_learner(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        assert isinstance(est, CausalEstimate)
        assert isinstance(cate, CATEResult)
        assert len(cate.cate_predictions) == len(outreach_data)

    def test_t_learner(self, outreach_data, covariates):
        est, cate = t_learner(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        assert isinstance(est, CausalEstimate)
        assert isinstance(cate, CATEResult)
        assert len(cate.cate_predictions) == len(outreach_data)


# ── Lift calculator tests ───────────────────────────────────────────────────

class TestLiftCalculator:
    def test_naive_lift(self, outreach_data):
        lift = compute_naive_lift(outreach_data, "Suggestion_Accepted", "Incremental_TRX")
        assert lift.n_treated + lift.n_control == len(outreach_data)
        assert lift.p_value >= 0

    def test_segment_lift(self, outreach_data):
        seg = compute_segment_lift(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", "HCP_Specialty"
        )
        assert len(seg) > 0
        assert "Absolute_Lift" in seg.columns

    def test_uplift_curve(self, outreach_data, covariates):
        _, cate = s_learner(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        uc = compute_uplift_curve(
            outreach_data["Incremental_TRX"].values,
            outreach_data["Suggestion_Accepted"].values,
            cate.cate_predictions,
        )
        assert len(uc) == 20
        assert all(0 < f <= 1 for f in uc["Fraction_Targeted"])

    def test_qini_curve(self, outreach_data, covariates):
        _, cate = s_learner(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        qc = compute_qini_curve(
            outreach_data["Incremental_TRX"].values,
            outreach_data["Suggestion_Accepted"].values,
            cate.cate_predictions,
        )
        assert qc["Incremental_Gain"].iloc[0] == 0.0

    def test_decile_lift(self, outreach_data, covariates):
        _, cate = s_learner(
            outreach_data, "Suggestion_Accepted", "Incremental_TRX", covariates
        )
        dl = compute_lift_by_decile(
            outreach_data["Incremental_TRX"].values,
            outreach_data["Suggestion_Accepted"].values,
            cate.cate_predictions,
        )
        assert "Observed_Lift" in dl.columns
        assert "Predicted_Lift" in dl.columns
