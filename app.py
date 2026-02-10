"""
Causal Inference Outreach Model - Interactive Web Application

A Streamlit application for analyzing the causal impact of Rep outreach
suggestions and actions on HCP prescription behavior (TRX / NBRX).

Supports Excel (.xlsx / .xls) and CSV data uploads with dynamic column mapping.

Two-stage analysis:
  Stage 1: Suggestions -> Incremental Actions
  Stage 2: Actions -> Incremental Outcomes (TRX or NBRX)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #e8ecf4 100%);
    }
    .upload-box {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin-bottom: 1rem;
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


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## Configuration")

# ── Data Source ──────────────────────────────────────────────────────────────

st.sidebar.markdown("### 1. Upload Data")
data_source = st.sidebar.radio(
    "Data source",
    ["Upload File (Excel / CSV)", "Demo with Synthetic Data"],
    index=0,
)

uploaded_file = None
if data_source == "Upload File (Excel / CSV)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your outreach data",
        type=["xlsx", "xls", "csv"],
        help="Supported formats: .xlsx, .xls, .csv",
    )

# ── Load raw data ────────────────────────────────────────────────────────────


@st.cache_data
def load_uploaded_file(file_data, file_name):
    """Load uploaded file (Excel or CSV) into DataFrame."""
    if file_name.endswith((".xlsx", ".xls")):
        # Read all sheets, let user pick
        xls = pd.ExcelFile(file_data)
        return xls
    else:
        return pd.read_csv(file_data)


@st.cache_data
def load_synthetic(n_records, seed):
    return generate_outreach_data(n_records=n_records, seed=seed)


# Determine whether we have data yet
df_raw = None
using_synthetic = False

if data_source == "Upload File (Excel / CSV)" and uploaded_file is not None:
    if uploaded_file.name.endswith((".xlsx", ".xls")):
        xls = load_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        sheet_names = xls.sheet_names
        if len(sheet_names) > 1:
            selected_sheet = st.sidebar.selectbox(
                "Select sheet",
                sheet_names,
                help="Your Excel file has multiple sheets. Pick the one with outreach data.",
            )
        else:
            selected_sheet = sheet_names[0]
        df_raw = pd.read_excel(xls, sheet_name=selected_sheet)
    else:
        df_raw = load_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)

elif data_source == "Demo with Synthetic Data":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Synthetic Data Settings")
    n_records = st.sidebar.slider("Number of records", 2000, 20000, 8000, 1000)
    seed = st.sidebar.number_input("Random seed", 1, 9999, 42)
    df_raw = load_synthetic(n_records, seed)
    using_synthetic = True

# ── If no data yet, show landing page ────────────────────────────────────────

if df_raw is None:
    st.markdown('<p class="main-header">Causal Inference - HCP Outreach Model</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload your Excel / CSV outreach data to get started</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-box">
            <h3>How it works</h3>
            <p>1. Upload your Excel (.xlsx) or CSV file using the sidebar</p>
            <p>2. Map your columns to the required fields (Rep ID, HCP ID, Suggestion Type, etc.)</p>
            <p>3. Select a Suggestion-Action combination (e.g. All-All, Call-Call, Email-Email)</p>
            <p>4. Choose the outcome to analyze (TRX or NBRX)</p>
            <p>5. Run causal inference models to estimate incremental impact</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Expected Data Columns")
    st.markdown(
        """
        Your data should contain columns for:
        - **Rep ID** - Sales representative identifier
        - **HCP ID** - Healthcare professional identifier
        - **Suggestion Type** - Type of suggestion given (Email, Call, Insights, All)
        - **Action Type** - Type of action the rep took (Email, Call, Insights, All)
        - **Treatment / Accepted** - Binary column: did the rep accept the suggestion? (1/0 or Yes/No)
        - **Outcome columns** - Incremental TRX, NBRX, or action counts
        - **Covariates** (optional) - Any additional numeric/categorical columns for confounding adjustment
        """
    )

    st.info("Use the sidebar to upload your file or switch to 'Demo with Synthetic Data' to explore the tool.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN MAPPING
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Map Your Columns")

all_cols = list(df_raw.columns)
none_option = ["-- Not Available --"]


def smart_default(options, keywords):
    """Find best default column match from a list of keywords."""
    for kw in keywords:
        for i, col in enumerate(options):
            if isinstance(col, str) and kw.lower() in col.lower():
                return i
    return 0


# Required columns
if using_synthetic:
    # For synthetic data, columns are already named correctly - skip mapping
    col_rep_id = "Rep_ID"
    col_hcp_id = "HCP_ID"
    col_suggestion_type = "Suggestion_Type"
    col_action_type = "Action_Type"
    col_treatment = "Suggestion_Accepted"
    col_incremental_actions = "Incremental_Actions"
    col_trx = "Incremental_TRX"
    col_nbrx = "Incremental_NBRX"
    # Covariates
    available_covariates = [
        "Rep_Experience_Years", "Territory_Size", "HCP_Specialty",
        "HCP_Patient_Volume", "HCP_Digital_Affinity",
    ]
else:
    col_rep_id = st.sidebar.selectbox(
        "Rep ID column",
        all_cols,
        index=smart_default(all_cols, ["rep_id", "rep", "representative", "sales_rep"]),
    )
    col_hcp_id = st.sidebar.selectbox(
        "HCP ID column",
        all_cols,
        index=smart_default(all_cols, ["hcp_id", "hcp", "physician", "doctor", "npi"]),
    )
    col_suggestion_type = st.sidebar.selectbox(
        "Suggestion Type column",
        all_cols,
        index=smart_default(all_cols, ["suggestion_type", "suggestion", "sug_type", "channel_suggested"]),
    )
    col_action_type = st.sidebar.selectbox(
        "Action Type column",
        all_cols,
        index=smart_default(all_cols, ["action_type", "action", "act_type", "channel_action"]),
    )
    col_treatment = st.sidebar.selectbox(
        "Treatment column (Suggestion Accepted: 1/0)",
        all_cols,
        index=smart_default(all_cols, [
            "suggestion_accepted", "accepted", "treatment", "treated", "is_accepted",
        ]),
        help="Binary column indicating if the rep accepted the suggestion (1=Yes, 0=No)",
    )

    # Outcome columns
    st.sidebar.markdown("#### Outcome Columns")

    outcome_cols_available = none_option + all_cols
    col_incremental_actions = st.sidebar.selectbox(
        "Incremental Actions column",
        outcome_cols_available,
        index=smart_default(outcome_cols_available, [
            "incremental_actions", "incr_actions", "action_count", "num_actions",
        ]),
    )
    if col_incremental_actions == none_option[0]:
        col_incremental_actions = None

    col_trx = st.sidebar.selectbox(
        "Incremental TRX column",
        outcome_cols_available,
        index=smart_default(outcome_cols_available, [
            "incremental_trx", "incr_trx", "delta_trx", "trx_lift", "trx",
        ]),
    )
    if col_trx == none_option[0]:
        col_trx = None

    col_nbrx = st.sidebar.selectbox(
        "Incremental NBRX column",
        outcome_cols_available,
        index=smart_default(outcome_cols_available, [
            "incremental_nbrx", "incr_nbrx", "delta_nbrx", "nbrx_lift", "nbrx",
        ]),
    )
    if col_nbrx == none_option[0]:
        col_nbrx = None

    # Covariates
    st.sidebar.markdown("#### Covariates (for confounding adjustment)")
    excluded = {col_rep_id, col_hcp_id, col_suggestion_type, col_action_type,
                col_treatment, col_incremental_actions, col_trx, col_nbrx}
    covariate_options = [c for c in all_cols if c not in excluded and c is not None]

    available_covariates = st.sidebar.multiselect(
        "Select covariate columns",
        covariate_options,
        default=covariate_options[:min(5, len(covariate_options))],
        help="Numeric or categorical columns to control for confounding",
    )


# ── Standardize column names ────────────────────────────────────────────────

if not using_synthetic:
    rename_map = {
        col_rep_id: "Rep_ID",
        col_hcp_id: "HCP_ID",
        col_suggestion_type: "Suggestion_Type",
        col_action_type: "Action_Type",
        col_treatment: "Suggestion_Accepted",
    }
    if col_incremental_actions:
        rename_map[col_incremental_actions] = "Incremental_Actions"
    if col_trx:
        rename_map[col_trx] = "Incremental_TRX"
    if col_nbrx:
        rename_map[col_nbrx] = "Incremental_NBRX"

    # Rename covariates to keep their original names (no rename needed)
    # but rename the core columns for internal consistency
    covariate_rename = {}
    for cov in available_covariates:
        if cov in rename_map:
            covariate_rename[cov] = rename_map[cov]

    df_full = df_raw.rename(columns=rename_map)

    # Update covariate names to match renamed columns
    available_covariates = [rename_map.get(c, c) for c in available_covariates]
else:
    df_full = df_raw.copy()


# ── Validate treatment column ────────────────────────────────────────────────

if "Suggestion_Accepted" in df_full.columns:
    # Convert Yes/No, True/False, Y/N to 1/0
    treat_col = df_full["Suggestion_Accepted"]
    if treat_col.dtype == object:
        mapping = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        df_full["Suggestion_Accepted"] = treat_col.str.strip().str.lower().map(mapping)
    df_full["Suggestion_Accepted"] = pd.to_numeric(df_full["Suggestion_Accepted"], errors="coerce")
    df_full = df_full.dropna(subset=["Suggestion_Accepted"])
    df_full["Suggestion_Accepted"] = df_full["Suggestion_Accepted"].astype(int)


# ── Normalize suggestion/action type values ──────────────────────────────────

def normalize_type_values(series):
    """Standardize suggestion/action type values to title case."""
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "email": "Email", "emails": "Email", "e-mail": "Email",
        "call": "Call", "calls": "Call", "phone": "Call",
        "insights": "Insights", "insight": "Insights",
        "all": "All", "all channels": "All", "all_channels": "All",
    }
    return s.map(lambda x: mapping.get(x, x.title()))


if "Suggestion_Type" in df_full.columns:
    df_full["Suggestion_Type"] = normalize_type_values(df_full["Suggestion_Type"])
if "Action_Type" in df_full.columns:
    df_full["Action_Type"] = normalize_type_values(df_full["Action_Type"])

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Analysis Settings")

# Build dynamic combo list from actual data values
sug_values = sorted(df_full["Suggestion_Type"].dropna().unique().tolist())
act_values = sorted(df_full["Action_Type"].dropna().unique().tolist())

# Build all possible combos from the data, plus an "All - All" (no filter) option
dynamic_combos = [("All", "All")]
for s in sug_values:
    for a in act_values:
        if (s, a) != ("All", "All"):
            sub = df_full[(df_full["Suggestion_Type"] == s) & (df_full["Action_Type"] == a)]
            if len(sub) >= 10:
                dynamic_combos.append((s, a))
# Also add "All-<action>" and "<suggestion>-All" combos
for a in act_values:
    if ("All", a) not in dynamic_combos:
        dynamic_combos.append(("All", a))
for s in sug_values:
    if (s, "All") not in dynamic_combos:
        dynamic_combos.append((s, "All"))

combo_labels = [f"{s} - {a}" for s, a in dynamic_combos]
selected_combo_label = st.sidebar.selectbox(
    "Suggestion - Action combination",
    combo_labels,
    index=0,
    help="Filter data by Suggestion Type and Action Type pair. 'All' means no filter on that dimension.",
)
selected_combo_idx = combo_labels.index(selected_combo_label)
selected_suggestion, selected_action = dynamic_combos[selected_combo_idx]

# Outcome selector - dynamically based on available columns
outcome_options = []
if "Incremental_TRX" in df_full.columns:
    outcome_options.append("TRX")
if "Incremental_NBRX" in df_full.columns:
    outcome_options.append("NBRX")
if "Incremental_Actions" in df_full.columns:
    outcome_options.append("Actions")

if not outcome_options:
    st.error(
        "No outcome columns found. Please map at least one of: "
        "Incremental TRX, Incremental NBRX, or Incremental Actions."
    )
    st.stop()

outcome_choice = st.sidebar.radio(
    "Incremental outcome to analyze",
    outcome_options,
    index=0,
    help="TRX = Total Prescriptions, NBRX = New-to-Brand Prescriptions, Actions = Outreach action count",
)

if outcome_choice == "Actions":
    outcome_col = "Incremental_Actions"
else:
    outcome_col = f"Incremental_{outcome_choice}"

# Analysis stage
st.sidebar.markdown("---")
analysis_stage = st.sidebar.radio(
    "Analysis stage",
    ["Incremental Actions (Suggestions -> Actions)", "Incremental Outcomes (Actions -> Outcome)"],
    index=1,
)

# Model configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 4. Model Settings")
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

# ══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS AND SET UP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

df = filter_by_combo(df_full, selected_suggestion, selected_action)

# Determine treatment and covariates based on analysis stage
treatment_col = "Suggestion_Accepted"

if "Incremental Actions" in analysis_stage:
    if "Incremental_Actions" not in df.columns:
        st.error("Incremental Actions column is required for Stage 1 analysis. Please map it in the sidebar.")
        st.stop()
    outcome_display = "Incremental_Actions"
    covariate_cols = [c for c in available_covariates if c != "Incremental_Actions"]
    stage_label = "Stage 1: Suggestions -> Incremental Actions"
    stage_description = "Estimating the causal effect of accepting a suggestion on the number of incremental outreach actions."
else:
    outcome_display = outcome_col
    if outcome_display not in df.columns:
        st.error(f"Column '{outcome_display}' not found. Please map the {outcome_choice} column in the sidebar.")
        st.stop()
    covariate_cols = available_covariates.copy()
    if "Incremental_Actions" in df.columns and "Incremental_Actions" not in covariate_cols:
        covariate_cols.append("Incremental_Actions")
    stage_label = f"Stage 2: Actions -> Incremental {outcome_choice}"
    stage_description = f"Estimating the causal effect of acting on suggestions on incremental {outcome_choice} outcomes."

# Ensure outcome column is numeric
df[outcome_display] = pd.to_numeric(df[outcome_display], errors="coerce")
df = df.dropna(subset=[outcome_display, treatment_col])

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
    st.markdown(f"### Data Overview  --  {selected_suggestion} / {selected_action}")
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
        # Pick a useful categorical column for the second chart
        cat_col = None
        for candidate in ["HCP_Specialty", "Suggestion_Type", "Action_Type"]:
            if candidate in df.columns and df[candidate].nunique() > 1:
                cat_col = candidate
                break
        if cat_col is None:
            for c in available_covariates:
                if c in df.columns and (df[c].dtype == object or df[c].dtype.name == "category"):
                    cat_col = c
                    break

        if cat_col:
            fig_spec = px.histogram(
                df,
                x=cat_col,
                color=treatment_col,
                barmode="group",
                title=f"Treatment Distribution by {cat_col}",
                color_discrete_map={0: "#EF5350", 1: "#42A5F5"},
            )
            fig_spec.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("#### Descriptive Statistics")
    numeric_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)


# ── TAB 2: Naive Lift ───────────────────────────────────────────────────────

with tab2:
    st.markdown(f"### Naive (Unadjusted) Lift  --  {outcome_display}")

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
        "may systematically differ from those who don't. Use the **Causal Estimates** tab for "
        "adjusted estimates."
    )


# ── TAB 3: Causal Estimates ─────────────────────────────────────────────────

with tab3:
    st.markdown(f"### Causal Lift Estimates  --  {outcome_display}")
    st.markdown(stage_description)

    if not causal_methods:
        st.warning("Select at least one causal method in the sidebar.")
    elif not covariate_cols:
        st.warning(
            "No covariates selected for confounding adjustment. "
            "Please select covariate columns in the sidebar (Step 2)."
        )
    else:
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
                hovertemplate=(
                    f"<b>{e.method}</b><br>ATE: {e.ate:.4f}<br>"
                    f"CI: [{e.ci_lower:.4f}, {e.ci_upper:.4f}]<br>"
                    f"p: {e.p_value:.4f}<extra></extra>"
                ),
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
    st.markdown(f"### Heterogeneous Treatment Effects  --  {outcome_display}")

    # cate_results may not be defined if tab3 had warnings
    _cate_results = cate_results if "cate_results" in dir() else {}

    if not _cate_results:
        st.info(
            "CATE analysis requires **S-Learner** or **T-Learner**. "
            "Enable them in the sidebar under Causal Methods, and ensure covariates are selected."
        )
    else:
        cate_method = st.selectbox(
            "Select CATE method",
            list(_cate_results.keys()),
        )
        cate = _cate_results[cate_method]
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
    st.markdown(f"### Segment-Level Lift  --  {outcome_display}")

    # Build segment options from categorical columns in the data
    segment_options = []
    for c in ["HCP_Specialty", "Suggestion_Type", "Action_Type"] + available_covariates:
        if c in df.columns and (df[c].dtype == object or df[c].dtype.name == "category"):
            if df[c].nunique() > 1 and c not in segment_options:
                segment_options.append(c)

    if not segment_options:
        st.info("No categorical segmentation variables found in the data.")
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
    st.markdown("Compare naive lift across all valid suggestion-action combinations in your data.")

    cross_results = []
    for sug, act in dynamic_combos:
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

    if not covariate_cols:
        st.warning("Select covariates in the sidebar to run propensity score diagnostics.")
    else:
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
                x=(["Control"] * int(np.sum(treatment == 0))
                   + ["Treated"] * int(np.sum(treatment == 1))),
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
        for c in covariate_cols:
            if df[c].dtype == object or df[c].dtype.name == "category":
                for val in sorted(df[c].unique())[1:]:
                    feat_names.append(f"{c}_{val}")
            else:
                feat_names.append(c)

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
            colors = [
                "#43A047" if v < 0.1 else "#FF9800" if v < 0.2 else "#EF5350"
                for v in smd_df["SMD"]
            ]
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
data_label = "Uploaded" if not using_synthetic else "Synthetic"
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    f"Causal Inference HCP Outreach Model &nbsp;|&nbsp; "
    f"Data: {data_label} ({len(df):,} records) &nbsp;|&nbsp; "
    f"Combo: {selected_suggestion}-{selected_action} &nbsp;|&nbsp; "
    f"Outcome: {outcome_choice}"
    "</div>",
    unsafe_allow_html=True,
)
