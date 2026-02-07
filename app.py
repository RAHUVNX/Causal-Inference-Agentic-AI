"""
Causal Inference Outreach Model - Interactive Web Application

A Streamlit application for analyzing the causal impact of Rep outreach
suggestions and actions on HCP prescription behavior (TRX / NBRX).

Two-stage analysis:
  Stage 1: Suggestions -> Incremental Actions
  Stage 2: Actions -> Incremental Outcomes (TRX or NBRX)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_generator import generate_outreach_data, filter_by_combo, VALID_COMBOS
from models.causal_models import (
    propensity_score_matching,
    inverse_propensity_weighting,
    doubly_robust,
    s_learner,
    t_learner,
    encode_features,
    estimate_propensity_scores,
)
from models.lift_calculator import (
    compute_naive_lift,
    compute_segment_lift,
    compute_uplift_curve,
    compute_qini_curve,
    compute_lift_by_decile,
)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Causal Inference - HCP Outreach Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1B2A4A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6B8A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 0.3rem 0 0 0;
        font-size: 1.8rem;
    }
    .metric-blue {
        background: linear-gradient(135deg, #2196F3 0%, #1565C0 100%);
    }
    .metric-green {
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
    }
    .metric-orange {
        background: linear-gradient(135deg, #FF9800 0%, #E65100 100%);
    }
    .metric-red {
        background: linear-gradient(135deg, #ef5350 0%, #c62828 100%);
    }
    .metric-teal {
        background: linear-gradient(135deg, #26A69A 0%, #00695C 100%);
    }
    .combo-badge {
        display: inline-block;
        background: #E3F2FD;
        color: #1565C0;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    .significance-yes {
        color: #2E7D32;
        font-weight: 700;
    }
    .significance-no {
        color: #C62828;
        font-weight: 700;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #e8ecf4 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric_card(title: str, value: str, css_class: str = "metric-card"):
    st.markdown(
        f'<div class="{css_class}"><h3>{title}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("## Configuration")

# Data source
st.sidebar.markdown("### Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Synthetic Data", "Upload CSV"],
    index=0,
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload outreach CSV", type=["csv"])
else:
    uploaded_file = None

st.sidebar.markdown("---")

# Suggestion-Action combo selector
st.sidebar.markdown("### Suggestion-Action Combination")
combo_labels = [f"{s} - {a}" for s, a in VALID_COMBOS]
selected_combo_label = st.sidebar.selectbox(
    "Select combination to analyze",
    combo_labels,
    index=0,
    help="Filter data by Suggestion Type and Action Type pair",
)
selected_combo_idx = combo_labels.index(selected_combo_label)
selected_suggestion, selected_action = VALID_COMBOS[selected_combo_idx]

st.sidebar.markdown("---")

# Outcome selector
st.sidebar.markdown("### Outcome Variable")
outcome_choice = st.sidebar.radio(
    "Incremental outcome to analyze",
    ["TRX", "NBRX"],
    index=0,
    help="TRX = Total Prescriptions, NBRX = New-to-Brand Prescriptions",
)
outcome_col = f"Incremental_{outcome_choice}"

st.sidebar.markdown("---")

# Analysis stage
st.sidebar.markdown("### Analysis Stage")
analysis_stage = st.sidebar.radio(
    "What to estimate",
    ["Incremental Actions (Suggestions -> Actions)", "Incremental Outcomes (Actions -> Outcome)"],
    index=1,
)

st.sidebar.markdown("---")

# Model configuration
st.sidebar.markdown("### Model Settings")
causal_methods = st.sidebar.multiselect(
    "Causal methods",
    ["PSM", "IPW", "Doubly Robust", "S-Learner", "T-Learner"],
    default=["PSM", "IPW", "Doubly Robust"],
)

ps_model_type = st.sidebar.selectbox(
    "Propensity score model",
    ["logistic", "gbm"],
    index=0,
)

if data_source == "Synthetic Data":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Synthetic Data Settings")
    n_records = st.sidebar.slider("Number of records", 2000, 20000, 8000, 1000)
    seed = st.sidebar.number_input("Random seed", 1, 9999, 42)


# ── Load Data ────────────────────────────────────────────────────────────────

@st.cache_data
def load_synthetic(n_records, seed):
    return generate_outreach_data(n_records=n_records, seed=seed)


if data_source == "Upload CSV" and uploaded_file is not None:
    df_full = pd.read_csv(uploaded_file)
elif data_source == "Synthetic Data":
    df_full = load_synthetic(n_records, seed)
else:
    df_full = load_synthetic(8000, 42)

# Apply combo filter
df = filter_by_combo(df_full, selected_suggestion, selected_action)

# Determine treatment and covariates based on analysis stage
if "Incremental Actions" in analysis_stage:
    treatment_col = "Suggestion_Accepted"
    outcome_display = "Incremental_Actions"
    covariate_cols = [
        "Rep_Experience_Years",
        "Territory_Size",
        "HCP_Specialty",
        "HCP_Patient_Volume",
        "HCP_Digital_Affinity",
    ]
    stage_label = "Stage 1: Suggestions -> Incremental Actions"
    stage_description = "Estimating the causal effect of accepting a suggestion on the number of incremental outreach actions."
else:
    treatment_col = "Suggestion_Accepted"
    outcome_display = outcome_col
    covariate_cols = [
        "Rep_Experience_Years",
        "Territory_Size",
        "HCP_Specialty",
        "HCP_Patient_Volume",
        "HCP_Digital_Affinity",
        "Incremental_Actions",
    ]
    stage_label = f"Stage 2: Actions -> Incremental {outcome_choice}"
    stage_description = f"Estimating the causal effect of acting on suggestions on incremental {outcome_choice} outcomes."

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">Causal Inference - HCP Outreach Model</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">{stage_label} &nbsp;|&nbsp; '
    f'Combo: <span class="combo-badge">{selected_suggestion} - {selected_action}</span></p>',
    unsafe_allow_html=True,
)

# ── Safety check ─────────────────────────────────────────────────────────────

if len(df) < 50:
    st.error(
        f"Only {len(df)} records match the selected combination "
        f"({selected_suggestion} - {selected_action}). "
        "Need at least 50 records for reliable estimation. "
        "Try a broader combination or increase sample size."
    )
    st.stop()

if df[treatment_col].nunique() < 2:
    st.error("The filtered data has only one treatment group. Cannot estimate causal effects.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Overview",
    "Naive Lift",
    "Causal Estimates",
    "Heterogeneous Effects (CATE)",
    "Segment Analysis",
    "Propensity Diagnostics",
])


# ── TAB 1: Data Overview ────────────────────────────────────────────────────

with tab1:
    st.markdown(f"### Data Overview  —  {selected_suggestion} / {selected_action}")
    st.markdown(stage_description)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Total Records", f"{len(df):,}", "metric-card metric-blue")
    with col2:
        n_treated = int(df[treatment_col].sum())
        render_metric_card("Accepted Suggestions", f"{n_treated:,}", "metric-card metric-green")
    with col3:
        n_control = int((df[treatment_col] == 0).sum())
        render_metric_card("Not Accepted", f"{n_control:,}", "metric-card metric-orange")
    with col4:
        accept_rate = n_treated / len(df) * 100
        render_metric_card("Acceptance Rate", f"{accept_rate:.1f}%", "metric-card metric-teal")
    with col5:
        mean_outcome = df[outcome_display].mean()
        render_metric_card(f"Mean {outcome_display}", f"{mean_outcome:.2f}", "metric-card metric-red")

    st.markdown("---")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Suggestion-Action Distribution (Full Dataset)")
        cross = pd.crosstab(df_full["Suggestion_Type"], df_full["Action_Type"], margins=True)
        st.dataframe(cross, use_container_width=True)

    with col_right:
        st.markdown("#### Filtered Data Preview")
        st.dataframe(df.head(20), use_container_width=True, height=300)

    st.markdown("---")

    # Distribution plots
    col_a, col_b = st.columns(2)
    with col_a:
        fig_out = px.histogram(
            df,
            x=outcome_display,
            color=treatment_col,
            barmode="overlay",
            nbins=40,
            title=f"Distribution of {outcome_display} by Treatment",
            color_discrete_map={0: "#EF5350", 1: "#42A5F5"},
            opacity=0.7,
        )
        fig_out.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_out, use_container_width=True)

    with col_b:
        fig_spec = px.histogram(
            df,
            x="HCP_Specialty",
            color=treatment_col,
            barmode="group",
            title="Treatment Distribution by HCP Specialty",
            color_discrete_map={0: "#EF5350", 1: "#42A5F5"},
        )
        fig_spec.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("#### Descriptive Statistics")
    desc_cols = [
        "Rep_Experience_Years", "Territory_Size", "HCP_Patient_Volume",
        "HCP_Digital_Affinity", "Incremental_Actions",
        "Incremental_TRX", "Incremental_NBRX",
    ]
    st.dataframe(df[desc_cols].describe().round(3), use_container_width=True)


# ── TAB 2: Naive Lift ───────────────────────────────────────────────────────

with tab2:
    st.markdown(f"### Naive (Unadjusted) Lift  —  {outcome_display}")

    lift = compute_naive_lift(df, treatment_col, outcome_display)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Absolute Lift", f"{lift.absolute_lift:.3f}", "metric-card metric-green")
    with col2:
        render_metric_card("Relative Lift", f"{lift.relative_lift_pct:.1f}%", "metric-card metric-blue")
    with col3:
        sig_label = "Yes" if lift.statistical_significance else "No"
        css = "metric-card metric-green" if lift.statistical_significance else "metric-card metric-red"
        render_metric_card("Statistically Significant", sig_label, css)
    with col4:
        render_metric_card("P-Value", f"{lift.p_value:.4f}", "metric-card metric-teal")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        fig_means = go.Figure()
        fig_means.add_trace(go.Bar(
            x=["Control (Not Accepted)", "Treatment (Accepted)"],
            y=[lift.control_mean, lift.treatment_mean],
            marker_color=["#EF5350", "#42A5F5"],
            text=[f"{lift.control_mean:.3f}", f"{lift.treatment_mean:.3f}"],
            textposition="outside",
        ))
        fig_means.update_layout(
            title=f"Mean {outcome_display}: Treated vs Control",
            yaxis_title=outcome_display,
            height=400,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_means, use_container_width=True)

    with col_r:
        st.markdown("#### Lift Summary")
        summary_data = {
            "Metric": [
                "Treatment Mean", "Control Mean", "Absolute Lift",
                "Relative Lift (%)", "95% CI Lower", "95% CI Upper",
                "P-Value", "N Treated", "N Control",
            ],
            "Value": [
                f"{lift.treatment_mean:.4f}", f"{lift.control_mean:.4f}",
                f"{lift.absolute_lift:.4f}", f"{lift.relative_lift_pct:.2f}%",
                f"{lift.ci_lower:.4f}", f"{lift.ci_upper:.4f}",
                f"{lift.p_value:.4f}", f"{lift.n_treated:,}", f"{lift.n_control:,}",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.info(
        "**Note:** Naive lift does not adjust for confounders. Reps who accept suggestions "
        "may systematically differ from those who don't. Use the Causal Estimates tab for "
        "adjusted estimates."
    )


# ── TAB 3: Causal Estimates ─────────────────────────────────────────────────

with tab3:
    st.markdown(f"### Causal Lift Estimates  —  {outcome_display}")
    st.markdown(stage_description)

    if not causal_methods:
        st.warning("Select at least one causal method in the sidebar.")
        st.stop()

    estimates = []
    cate_results = {}

    with st.spinner("Running causal estimation methods..."):
        method_map = {
            "PSM": ("Propensity Score Matching", lambda: propensity_score_matching(
                df, treatment_col, outcome_display, covariate_cols, ps_model_type
            )),
            "IPW": ("Inverse Propensity Weighting", lambda: inverse_propensity_weighting(
                df, treatment_col, outcome_display, covariate_cols, ps_model_type
            )),
            "Doubly Robust": ("Doubly Robust (AIPW)", lambda: doubly_robust(
                df, treatment_col, outcome_display, covariate_cols, ps_model_type
            )),
            "S-Learner": ("S-Learner", lambda: s_learner(
                df, treatment_col, outcome_display, covariate_cols
            )),
            "T-Learner": ("T-Learner", lambda: t_learner(
                df, treatment_col, outcome_display, covariate_cols
            )),
        }

        for method_key in causal_methods:
            label, fn = method_map[method_key]
            result = fn()
            if isinstance(result, tuple):
                est, cate = result
                estimates.append(est)
                cate_results[method_key] = cate
            else:
                estimates.append(result)

    # Summary table
    est_rows = []
    for e in estimates:
        est_rows.append({
            "Method": e.method,
            "ATE": round(e.ate, 4),
            "Std Error": round(e.ate_se, 4),
            "95% CI Lower": round(e.ci_lower, 4),
            "95% CI Upper": round(e.ci_upper, 4),
            "P-Value": round(e.p_value, 4),
            "Significant": "Yes" if e.p_value < 0.05 else "No",
            "N Treated": e.n_treated,
            "N Control": e.n_control,
        })
    est_df = pd.DataFrame(est_rows)
    st.dataframe(est_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Forest plot
    fig_forest = go.Figure()
    for i, e in enumerate(estimates):
        color = "#2E7D32" if e.p_value < 0.05 else "#C62828"
        fig_forest.add_trace(go.Scatter(
            x=[e.ci_lower, e.ate, e.ci_upper],
            y=[e.method] * 3,
            mode="lines+markers",
            marker=dict(size=[8, 14, 8], color=color),
            line=dict(color=color, width=3),
            name=e.method,
            showlegend=False,
            hovertemplate=f"<b>{e.method}</b><br>ATE: {e.ate:.4f}<br>CI: [{e.ci_lower:.4f}, {e.ci_upper:.4f}]<br>p: {e.p_value:.4f}<extra></extra>",
        ))

    fig_forest.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_forest.update_layout(
        title="Forest Plot: Causal Effect Estimates (95% CI)",
        xaxis_title=f"Average Treatment Effect on {outcome_display}",
        height=max(250, 80 * len(estimates)),
        margin=dict(l=200, t=40, b=40),
    )
    st.plotly_chart(fig_forest, use_container_width=True)


# ── TAB 4: Heterogeneous Effects (CATE) ─────────────────────────────────────

with tab4:
    st.markdown(f"### Heterogeneous Treatment Effects  —  {outcome_display}")

    if not cate_results:
        st.info(
            "CATE analysis requires S-Learner or T-Learner. "
            "Enable them in the sidebar under Causal Methods."
        )
    else:
        cate_method = st.selectbox(
            "Select CATE method",
            list(cate_results.keys()),
        )
        cate = cate_results[cate_method]
        preds = cate.cate_predictions

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Mean CATE", f"{np.mean(preds):.4f}", "metric-card metric-blue")
        with col2:
            render_metric_card("Median CATE", f"{np.median(preds):.4f}", "metric-card metric-green")
        with col3:
            pct_positive = (preds > 0).mean() * 100
            render_metric_card("% Positive CATE", f"{pct_positive:.1f}%", "metric-card metric-teal")

        st.markdown("---")

        # CATE distribution
        col_l, col_r = st.columns(2)
        with col_l:
            fig_cate = px.histogram(
                x=preds, nbins=50,
                title=f"CATE Distribution ({cate_method})",
                labels={"x": f"Predicted Effect on {outcome_display}", "y": "Count"},
                color_discrete_sequence=["#42A5F5"],
            )
            fig_cate.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.7)
            fig_cate.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig_cate, use_container_width=True)

        with col_r:
            # Uplift curve
            uplift_df = compute_uplift_curve(
                df[outcome_display].values,
                df[treatment_col].values,
                preds,
            )
            fig_uplift = px.line(
                uplift_df, x="Fraction_Targeted", y="Cumulative_Uplift",
                title="Uplift Curve (Cumulative Gain)",
                markers=True,
            )
            fig_uplift.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig_uplift, use_container_width=True)

        st.markdown("---")

        # Qini curve and decile analysis
        col_q, col_d = st.columns(2)
        with col_q:
            qini_df = compute_qini_curve(
                df[outcome_display].values,
                df[treatment_col].values,
                preds,
            )
            fig_qini = px.line(
                qini_df, x="Fraction_Targeted", y="Incremental_Gain",
                title="Qini Curve (Incremental Gains)",
                markers=True,
                color_discrete_sequence=["#FF7043"],
            )
            fig_qini.update_layout(height=400, margin=dict(t=40, b=20))
            st.plotly_chart(fig_qini, use_container_width=True)

        with col_d:
            decile_df = compute_lift_by_decile(
                df[outcome_display].values,
                df[treatment_col].values,
                preds,
            )
            fig_dec = go.Figure()
            fig_dec.add_trace(go.Bar(
                x=decile_df["Decile"], y=decile_df["Observed_Lift"],
                name="Observed Lift", marker_color="#42A5F5",
            ))
            fig_dec.add_trace(go.Bar(
                x=decile_df["Decile"], y=decile_df["Predicted_Lift"],
                name="Predicted Lift", marker_color="#FF7043",
            ))
            fig_dec.update_layout(
                title="Lift by CATE Decile",
                xaxis_title="Decile",
                yaxis_title="Lift",
                barmode="group",
                height=400,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_dec, use_container_width=True)

        st.markdown("#### Decile Details")
        st.dataframe(decile_df, use_container_width=True, hide_index=True)


# ── TAB 5: Segment Analysis ─────────────────────────────────────────────────

with tab5:
    st.markdown(f"### Segment-Level Lift  —  {outcome_display}")

    segment_options = ["HCP_Specialty", "Suggestion_Type", "Action_Type"]
    # Only show segment options that have >1 unique value in filtered data
    segment_options = [s for s in segment_options if df[s].nunique() > 1]

    if not segment_options:
        st.info("No segmentation variables available with multiple values in the filtered data.")
    else:
        segment_col = st.selectbox("Segment by", segment_options)

        seg_df = compute_segment_lift(df, treatment_col, outcome_display, segment_col)
        if seg_df.empty:
            st.warning("Not enough data in each segment for lift computation.")
        else:
            st.dataframe(seg_df, use_container_width=True, hide_index=True)

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_seg = go.Figure()
                fig_seg.add_trace(go.Bar(
                    x=seg_df["Segment"],
                    y=seg_df["Absolute_Lift"],
                    marker_color=[
                        "#43A047" if sig else "#EF5350"
                        for sig in seg_df["Significant"]
                    ],
                    text=[f"{v:.3f}" for v in seg_df["Absolute_Lift"]],
                    textposition="outside",
                ))
                fig_seg.update_layout(
                    title=f"Absolute Lift by {segment_col}",
                    yaxis_title="Absolute Lift",
                    height=400,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_seg, use_container_width=True)

            with col_s2:
                fig_seg2 = go.Figure()
                fig_seg2.add_trace(go.Bar(
                    x=seg_df["Segment"],
                    y=seg_df["Treatment_Mean"],
                    name="Treatment",
                    marker_color="#42A5F5",
                ))
                fig_seg2.add_trace(go.Bar(
                    x=seg_df["Segment"],
                    y=seg_df["Control_Mean"],
                    name="Control",
                    marker_color="#EF5350",
                ))
                fig_seg2.update_layout(
                    title=f"Outcome Means by {segment_col}",
                    yaxis_title=f"Mean {outcome_display}",
                    barmode="group",
                    height=400,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_seg2, use_container_width=True)

    # Cross-combo comparison
    st.markdown("---")
    st.markdown("### Cross-Combination Comparison")
    st.markdown("Compare lift across all valid suggestion-action combinations.")

    cross_results = []
    for sug, act in VALID_COMBOS:
        sub = filter_by_combo(df_full, sug, act)
        if len(sub) >= 30 and sub[treatment_col].nunique() == 2:
            lift_r = compute_naive_lift(sub, treatment_col, outcome_display)
            cross_results.append({
                "Combination": f"{sug} - {act}",
                "N": len(sub),
                "Absolute_Lift": round(lift_r.absolute_lift, 4),
                "Relative_Lift_Pct": round(lift_r.relative_lift_pct, 2),
                "P_Value": round(lift_r.p_value, 4),
                "Significant": lift_r.statistical_significance,
            })

    if cross_results:
        cross_df = pd.DataFrame(cross_results)
        st.dataframe(cross_df, use_container_width=True, hide_index=True)

        fig_cross = go.Figure()
        fig_cross.add_trace(go.Bar(
            x=cross_df["Combination"],
            y=cross_df["Absolute_Lift"],
            marker_color=[
                "#43A047" if sig else "#EF5350"
                for sig in cross_df["Significant"]
            ],
            text=[f"{v:.3f}" for v in cross_df["Absolute_Lift"]],
            textposition="outside",
        ))
        fig_cross.update_layout(
            title=f"Naive Lift Across All Combinations ({outcome_display})",
            yaxis_title="Absolute Lift",
            height=450,
            margin=dict(t=40, b=80),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_cross, use_container_width=True)


# ── TAB 6: Propensity Diagnostics ───────────────────────────────────────────

with tab6:
    st.markdown("### Propensity Score Diagnostics")

    X = encode_features(df, covariate_cols)
    treatment = df[treatment_col].values

    with st.spinner("Estimating propensity scores..."):
        ps = estimate_propensity_scores(X, treatment, ps_model_type)

    col1, col2 = st.columns(2)
    with col1:
        fig_ps = go.Figure()
        fig_ps.add_trace(go.Histogram(
            x=ps[treatment == 1],
            name="Treated",
            marker_color="#42A5F5",
            opacity=0.7,
            nbinsx=40,
        ))
        fig_ps.add_trace(go.Histogram(
            x=ps[treatment == 0],
            name="Control",
            marker_color="#EF5350",
            opacity=0.7,
            nbinsx=40,
        ))
        fig_ps.update_layout(
            title="Propensity Score Distribution",
            xaxis_title="Propensity Score",
            yaxis_title="Count",
            barmode="overlay",
            height=400,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_ps, use_container_width=True)

    with col2:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            x=["Control"] * int(np.sum(treatment == 0)) + ["Treated"] * int(np.sum(treatment == 1)),
            y=np.concatenate([ps[treatment == 0], ps[treatment == 1]]),
            marker_color="#667eea",
        ))
        fig_box.update_layout(
            title="Propensity Score Box Plot",
            yaxis_title="Propensity Score",
            height=400,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Overlap statistics
    st.markdown("#### Overlap Statistics")
    overlap_data = {
        "Group": ["Treated", "Control"],
        "N": [int(np.sum(treatment == 1)), int(np.sum(treatment == 0))],
        "Mean PS": [round(np.mean(ps[treatment == 1]), 4), round(np.mean(ps[treatment == 0]), 4)],
        "Median PS": [round(np.median(ps[treatment == 1]), 4), round(np.median(ps[treatment == 0]), 4)],
        "Min PS": [round(np.min(ps[treatment == 1]), 4), round(np.min(ps[treatment == 0]), 4)],
        "Max PS": [round(np.max(ps[treatment == 1]), 4), round(np.max(ps[treatment == 0]), 4)],
    }
    st.dataframe(pd.DataFrame(overlap_data), use_container_width=True, hide_index=True)

    # Standardized mean differences
    st.markdown("#### Covariate Balance (Standardized Mean Differences)")
    X_numeric = encode_features(df, covariate_cols)
    smd_rows = []
    feat_names = []
    for col in covariate_cols:
        if df[col].dtype == object or df[col].dtype.name == "category":
            for val in sorted(df[col].unique())[1:]:
                feat_names.append(f"{col}_{val}")
        else:
            feat_names.append(col)

    for i, fname in enumerate(feat_names):
        if i < X_numeric.shape[1]:
            t_vals = X_numeric[treatment == 1, i]
            c_vals = X_numeric[treatment == 0, i]
            pooled_sd = np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2)
            smd = (np.mean(t_vals) - np.mean(c_vals)) / pooled_sd if pooled_sd > 0 else 0
            smd_rows.append({"Covariate": fname, "SMD": round(abs(smd), 4)})

    if smd_rows:
        smd_df = pd.DataFrame(smd_rows)
        fig_smd = go.Figure()
        colors = ["#43A047" if v < 0.1 else "#FF9800" if v < 0.2 else "#EF5350" for v in smd_df["SMD"]]
        fig_smd.add_trace(go.Bar(
            x=smd_df["SMD"],
            y=smd_df["Covariate"],
            orientation="h",
            marker_color=colors,
        ))
        fig_smd.add_vline(x=0.1, line_dash="dash", line_color="orange", annotation_text="0.1 threshold")
        fig_smd.update_layout(
            title="Standardized Mean Differences (|SMD|)",
            xaxis_title="|SMD|",
            height=max(300, 40 * len(smd_rows)),
            margin=dict(l=200, t=40, b=20),
        )
        st.plotly_chart(fig_smd, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Causal Inference HCP Outreach Model &nbsp;|&nbsp; "
    f"Data: {len(df):,} records &nbsp;|&nbsp; "
    f"Combination: {selected_suggestion}-{selected_action} &nbsp;|&nbsp; "
    f"Outcome: {outcome_choice}"
    "</div>",
    unsafe_allow_html=True,
)
