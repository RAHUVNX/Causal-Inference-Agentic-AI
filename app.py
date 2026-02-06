"""
Causal Inference Lift App
=========================
Interactive Streamlit application for estimating the causal lift
(incremental impact) of actions / suggestions using multiple
causal inference methods.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_generator import generate_marketing_campaign_data, generate_ab_test_data
from models.causal_models import (
    CausalEstimate,
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

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Causal Inference Lift App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.85; }
    .metric-card h1 { margin: 5px 0 0 0; font-size: 28px; }
    .positive { color: #00c853; }
    .negative { color: #ff1744; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ─────────────────────────────────────────────────────────


def render_metric_card(label: str, value: str, delta: str = ""):
    """Render a styled metric card."""
    delta_html = f'<p style="margin:2px 0 0 0;font-size:13px;">{delta}</p>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h1>{value}</h1>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical columns for modelling."""
    return pd.get_dummies(df, columns=cols, drop_first=True, dtype=float)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Causal Inference Lift App")
st.sidebar.markdown("---")

data_source = st.sidebar.radio(
    "Data Source",
    ["Generate Synthetic Data", "Upload CSV"],
    index=0,
)

if data_source == "Generate Synthetic Data":
    dataset_type = st.sidebar.selectbox(
        "Dataset Type",
        ["Marketing Campaign (Observational)", "A/B Test (Randomised)"],
    )
    n_samples = st.sidebar.slider("Sample Size", 500, 20000, 5000, step=500)
    true_effect = st.sidebar.slider("True Treatment Effect", 0.01, 0.30, 0.10, step=0.01)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)

    if dataset_type.startswith("Marketing"):
        df = generate_marketing_campaign_data(n_samples, true_effect, seed)
        treatment_col = "treatment"
        outcome_col = "conversion"
        cat_cols = ["channel"]
    else:
        df = generate_ab_test_data(n_samples, true_effect * 100, seed)
        treatment_col = "treatment"
        outcome_col = "conversion"
        cat_cols = []
else:
    uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.info("Please upload a CSV file to get started, or switch to synthetic data.")
        st.stop()

    all_cols = df.columns.tolist()
    treatment_col = st.sidebar.selectbox("Treatment Column", all_cols)
    outcome_col = st.sidebar.selectbox(
        "Outcome Column", [c for c in all_cols if c != treatment_col]
    )
    cat_cols = st.sidebar.multiselect(
        "Categorical Columns (to encode)",
        [c for c in all_cols if c not in [treatment_col, outcome_col]],
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")

methods = st.sidebar.multiselect(
    "Causal Methods to Run",
    [
        "Propensity Score Matching",
        "Inverse Propensity Weighting",
        "Doubly Robust (AIPW)",
        "S-Learner",
        "T-Learner",
    ],
    default=["Propensity Score Matching", "Doubly Robust (AIPW)", "T-Learner"],
)

ps_model_type = st.sidebar.selectbox(
    "Propensity Score Model", ["logistic", "gbm"], index=0
)

# ── Main area ────────────────────────────────────────────────────────────────

st.title("Causal Inference & Lift Analysis")
st.markdown(
    "Estimate the **incremental lift** of actions / suggestions using "
    "multiple causal inference methods. Compare naive vs. causal estimates "
    "and explore heterogeneous treatment effects."
)

# ── 1. Data Overview ─────────────────────────────────────────────────────────

with st.expander("Data Overview", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Features", f"{df.shape[1]}")
    col3.metric("Treated", f"{(df[treatment_col] == 1).sum():,}")
    col4.metric("Control", f"{(df[treatment_col] == 0).sum():,}")

    tab_preview, tab_stats = st.tabs(["Preview", "Statistics"])
    with tab_preview:
        st.dataframe(df.head(100), use_container_width=True)
    with tab_stats:
        st.dataframe(df.describe().round(3), use_container_width=True)

# ── 2. Naive Lift ────────────────────────────────────────────────────────────

st.header("1. Naive (Unadjusted) Lift")

naive_lift = compute_basic_lift(df, treatment_col, outcome_col)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("Absolute Lift", f"{naive_lift.absolute_lift:.4f}")
with col2:
    render_metric_card("Relative Lift", f"{naive_lift.relative_lift_pct:.1f}%")
with col3:
    render_metric_card("Treatment Rate", f"{naive_lift.treatment_rate:.4f}")
with col4:
    render_metric_card("Control Rate", f"{naive_lift.control_rate:.4f}")
with col5:
    sig_text = "Yes (p<0.05)" if naive_lift.statistical_significance else "No"
    render_metric_card("Significant?", sig_text, f"p = {naive_lift.p_value:.4f}")

st.caption(
    "**Naive lift** does not adjust for confounders. In observational data this "
    "can be misleading. Use causal methods below for unbiased estimates."
)

# ── 3. Causal Estimates ─────────────────────────────────────────────────────

st.header("2. Causal Lift Estimates")

if not methods:
    st.warning("Select at least one causal method in the sidebar.")
    st.stop()

# Prepare data
id_cols = [c for c in df.columns if c.endswith("_id")]
exclude = [treatment_col, outcome_col, "true_ite"] + id_cols
covariate_cols = [c for c in df.columns if c not in exclude and c not in cat_cols]

df_encoded = encode_categoricals(df.copy(), cat_cols) if cat_cols else df.copy()
# Update covariate list after encoding
encoded_covariates = [
    c for c in df_encoded.columns if c not in [treatment_col, outcome_col, "true_ite"] + id_cols
]

# Run selected methods
model_map = {
    "Propensity Score Matching": lambda: propensity_score_matching(
        df_encoded, encoded_covariates, treatment_col, outcome_col, ps_model=ps_model_type
    ),
    "Inverse Propensity Weighting": lambda: inverse_propensity_weighting(
        df_encoded, encoded_covariates, treatment_col, outcome_col, ps_model=ps_model_type
    ),
    "Doubly Robust (AIPW)": lambda: doubly_robust(
        df_encoded, encoded_covariates, treatment_col, outcome_col, ps_model=ps_model_type
    ),
    "S-Learner": lambda: s_learner(
        df_encoded, encoded_covariates, treatment_col, outcome_col
    ),
    "T-Learner": lambda: t_learner(
        df_encoded, encoded_covariates, treatment_col, outcome_col
    ),
}

estimates: list[CausalEstimate] = []

progress = st.progress(0, text="Running causal models...")
for i, method in enumerate(methods):
    progress.progress((i + 1) / len(methods), text=f"Running {method}...")
    est = model_map[method]()
    estimates.append(est)
progress.empty()

# Summary table
summary_rows = []
for est in estimates:
    summary_rows.append(
        {
            "Method": est.method,
            "ATE (Lift)": est.ate,
            "Std Error": est.ate_se,
            "95% CI Lower": est.ci_lower,
            "95% CI Upper": est.ci_upper,
            "p-value": est.p_value,
            "Significant": "Yes" if est.p_value < 0.05 else "No",
        }
    )
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Show true effect if available
if "true_ite" in df.columns:
    true_ate = df["true_ite"].mean()
    st.info(f"**True Average Treatment Effect (from data generation):** {true_ate:.4f}")

# ATE comparison chart
fig_ate = go.Figure()
for est in estimates:
    fig_ate.add_trace(
        go.Scatter(
            x=[est.method],
            y=[est.ate],
            error_y=dict(type="data", symmetric=False,
                         array=[est.ci_upper - est.ate],
                         arrayminus=[est.ate - est.ci_lower]),
            mode="markers",
            marker=dict(size=12),
            name=est.method,
        )
    )
# Add naive line
fig_ate.add_hline(
    y=naive_lift.absolute_lift,
    line_dash="dash",
    line_color="red",
    annotation_text="Naive Lift",
)
if "true_ite" in df.columns:
    fig_ate.add_hline(
        y=true_ate,
        line_dash="dot",
        line_color="green",
        annotation_text="True ATE",
    )
fig_ate.update_layout(
    title="ATE Estimates with 95% Confidence Intervals",
    yaxis_title="Average Treatment Effect",
    showlegend=False,
    height=400,
)
st.plotly_chart(fig_ate, use_container_width=True)

# ── 4. Heterogeneous Treatment Effects ──────────────────────────────────────

st.header("3. Heterogeneous Treatment Effects (CATE)")

# Find an estimate with CATE predictions
cate_est = None
for est in estimates:
    if "cate" in est.details:
        cate_est = est
        break

if cate_est is not None:
    cate = cate_est.details["cate"]
    st.markdown(f"*Using CATE predictions from **{cate_est.method}***")

    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_hist = px.histogram(
            x=cate,
            nbins=50,
            title="Distribution of Individual Treatment Effects (CATE)",
            labels={"x": "CATE", "y": "Count"},
            color_discrete_sequence=["#667eea"],
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        # CATE by decile
        decile_df = compute_lift_by_decile(
            cate, df[treatment_col].values, df[outcome_col].values
        )
        fig_dec = go.Figure()
        fig_dec.add_trace(
            go.Bar(
                x=decile_df["decile"],
                y=decile_df["observed_lift"],
                name="Observed Lift",
                marker_color="#764ba2",
            )
        )
        fig_dec.add_trace(
            go.Scatter(
                x=decile_df["decile"],
                y=decile_df["mean_predicted_cate"],
                name="Predicted CATE",
                mode="lines+markers",
                line=dict(color="#00c853", width=2),
            )
        )
        fig_dec.update_layout(
            title="Lift by CATE Decile",
            xaxis_title="Decile (1=highest predicted effect)",
            yaxis_title="Lift",
            height=350,
        )
        st.plotly_chart(fig_dec, use_container_width=True)

    # Uplift & Qini curves
    st.subheader("Uplift & Qini Curves")
    col_up, col_qi = st.columns(2)

    with col_up:
        uplift_df = compute_uplift_curve(
            cate, df[treatment_col].values, df[outcome_col].values
        )
        fig_up = px.line(
            uplift_df,
            x="fraction_targeted",
            y="cumulative_uplift",
            title="Cumulative Uplift Curve",
            labels={"fraction_targeted": "Fraction Targeted", "cumulative_uplift": "Uplift"},
        )
        fig_up.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_up.update_layout(height=350)
        st.plotly_chart(fig_up, use_container_width=True)

    with col_qi:
        qini_df = compute_qini_curve(
            cate, df[treatment_col].values, df[outcome_col].values
        )
        fig_qi = px.line(
            qini_df,
            x="fraction_targeted",
            y="incremental_gains",
            title="Qini Curve (Incremental Gains)",
            labels={
                "fraction_targeted": "Fraction Targeted",
                "incremental_gains": "Incremental Conversions",
            },
        )
        fig_qi.update_layout(height=350)
        st.plotly_chart(fig_qi, use_container_width=True)

    # Feature importance for CATE
    if "true_ite" in df.columns:
        st.subheader("CATE vs True ITE")
        scatter_df = pd.DataFrame({"Predicted CATE": cate, "True ITE": df["true_ite"].values})
        fig_scatter = px.scatter(
            scatter_df,
            x="True ITE",
            y="Predicted CATE",
            title="Predicted CATE vs True Individual Treatment Effect",
            opacity=0.3,
            color_discrete_sequence=["#667eea"],
        )
        fig_scatter.add_shape(
            type="line", x0=scatter_df["True ITE"].min(), x1=scatter_df["True ITE"].max(),
            y0=scatter_df["True ITE"].min(), y1=scatter_df["True ITE"].max(),
            line=dict(dash="dash", color="red"),
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("Select **S-Learner** or **T-Learner** to see heterogeneous treatment effect analysis.")

# ── 5. Segment Analysis ─────────────────────────────────────────────────────

st.header("4. Segment-Level Lift")

segment_candidates = [c for c in df.columns if c not in [treatment_col, outcome_col, "true_ite"] + id_cols]
if segment_candidates:
    seg_col = st.selectbox("Segment By", segment_candidates)

    # Bin numeric columns for segment analysis
    if df[seg_col].dtype in ["float64", "int64"] and df[seg_col].nunique() > 10:
        df["_segment_bin"] = pd.qcut(df[seg_col], 5, duplicates="drop")
        seg_df = compute_segment_lift(df, treatment_col, outcome_col, "_segment_bin")
        df.drop(columns=["_segment_bin"], inplace=True)
    else:
        seg_df = compute_segment_lift(df, treatment_col, outcome_col, seg_col)

    if not seg_df.empty:
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

        fig_seg = go.Figure()
        colors = ["#00c853" if s else "#ff1744" for s in seg_df["significant"]]
        fig_seg.add_trace(
            go.Bar(
                x=seg_df["segment"].astype(str),
                y=seg_df["absolute_lift"],
                marker_color=colors,
                text=seg_df["relative_lift_pct"].apply(lambda v: f"{v:+.1f}%"),
                textposition="outside",
            )
        )
        fig_seg.update_layout(
            title=f"Absolute Lift by {seg_col}",
            xaxis_title=seg_col,
            yaxis_title="Absolute Lift",
            height=400,
        )
        st.plotly_chart(fig_seg, use_container_width=True)
    else:
        st.warning("Not enough data in segments for lift calculation.")

# ── 6. Propensity Score Diagnostics ─────────────────────────────────────────

st.header("5. Propensity Score Diagnostics")

ps_est = None
for est in estimates:
    if "propensity_scores" in est.details:
        ps_est = est
        break

if ps_est is not None:
    ps = ps_est.details["propensity_scores"]
    ps_df = pd.DataFrame({"propensity_score": ps, "treatment": df[treatment_col].values})

    col_ps1, col_ps2 = st.columns(2)
    with col_ps1:
        fig_ps = px.histogram(
            ps_df,
            x="propensity_score",
            color="treatment",
            barmode="overlay",
            nbins=50,
            title="Propensity Score Distribution by Treatment Group",
            labels={"propensity_score": "Propensity Score", "treatment": "Treatment"},
            color_discrete_map={0: "#ff6b6b", 1: "#51cf66"},
            opacity=0.6,
        )
        fig_ps.update_layout(height=400)
        st.plotly_chart(fig_ps, use_container_width=True)

    with col_ps2:
        fig_ps_box = px.box(
            ps_df,
            x="treatment",
            y="propensity_score",
            color="treatment",
            title="Propensity Score Box Plot",
            color_discrete_map={0: "#ff6b6b", 1: "#51cf66"},
        )
        fig_ps_box.update_layout(height=400)
        st.plotly_chart(fig_ps_box, use_container_width=True)

    # Overlap statistics
    ps_treated = ps[df[treatment_col].values == 1]
    ps_control = ps[df[treatment_col].values == 0]
    overlap_min = max(ps_treated.min(), ps_control.min())
    overlap_max = min(ps_treated.max(), ps_control.max())
    st.markdown(
        f"**Overlap region:** [{overlap_min:.3f}, {overlap_max:.3f}] | "
        f"Treated PS range: [{ps_treated.min():.3f}, {ps_treated.max():.3f}] | "
        f"Control PS range: [{ps_control.min():.3f}, {ps_control.max():.3f}]"
    )
else:
    st.info("Run a propensity-score-based method (PSM, IPW, or Doubly Robust) to see diagnostics.")

# ── 7. Recommendations ──────────────────────────────────────────────────────

st.header("6. Actionable Recommendations")

if estimates:
    best_est = min(estimates, key=lambda e: e.p_value)

    if best_est.p_value < 0.05 and best_est.ate > 0:
        st.success(
            f"**The action has a statistically significant positive causal effect.**\n\n"
            f"- Best estimate ({best_est.method}): **ATE = {best_est.ate:.4f}** "
            f"(95% CI: [{best_est.ci_lower:.4f}, {best_est.ci_upper:.4f}])\n"
            f"- This means the suggestion/action causally increases the outcome by "
            f"**{best_est.ate:.4f}** on average per individual."
        )

        if cate_est is not None:
            pct_positive = (cate > 0).mean() * 100
            st.markdown(
                f"- **{pct_positive:.1f}%** of individuals are predicted to benefit from the action.\n"
                f"- Consider **targeting the top deciles** (highest predicted CATE) for maximum ROI."
            )
    elif best_est.p_value < 0.05 and best_est.ate < 0:
        st.error(
            f"**The action has a statistically significant NEGATIVE causal effect.**\n\n"
            f"- Best estimate ({best_est.method}): ATE = {best_est.ate:.4f}\n"
            f"- Consider **stopping or redesigning** this action."
        )
    else:
        st.warning(
            f"**No statistically significant causal effect detected** (p = {best_est.p_value:.4f}).\n\n"
            f"- The estimated ATE is {best_est.ate:.4f} but we cannot rule out zero effect.\n"
            f"- Consider collecting more data or redesigning the intervention."
        )

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:12px;'>"
    "Causal Inference Lift App | Built with Streamlit, scikit-learn, and Plotly"
    "</div>",
    unsafe_allow_html=True,
)
