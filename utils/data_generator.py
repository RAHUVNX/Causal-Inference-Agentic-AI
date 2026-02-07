"""
Synthetic data generator for Rep-HCP outreach causal inference model.

Generates data at the granularity of: Rep_ID, HCP_ID, Suggestion_Type, Action_Type
with two outcome stages:
  1. Incremental Actions: Did the Rep act on the suggestion?
  2. Incremental Outcomes: What was the outcome (TRX / NBRX) from the action?
"""

import numpy as np
import pandas as pd


SUGGESTION_TYPES = ["Email", "Call", "Insights", "All"]
ACTION_TYPES = ["Email", "Call", "Insights", "All"]

# Valid suggestion-action combinations for filtering
VALID_COMBOS = [
    ("All", "All"),
    ("All", "Call"),
    ("All", "Email"),
    ("All", "Insights"),
    ("Call", "Call"),
    ("Email", "Email"),
    ("Insights", "Insights"),
]


def generate_outreach_data(
    n_reps: int = 80,
    n_hcps: int = 500,
    n_records: int = 8000,
    base_action_rate: float = 0.45,
    base_trx_effect: float = 3.5,
    base_nbrx_effect: float = 1.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic Rep-HCP outreach data.

    Parameters
    ----------
    n_reps : int
        Number of unique sales representatives.
    n_hcps : int
        Number of unique healthcare professionals.
    base_action_rate : float
        Baseline probability a rep acts on a suggestion.
    base_trx_effect : float
        Average incremental TRX uplift when action is taken.
    base_nbrx_effect : float
        Average incremental NBRX uplift when action is taken.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic outreach dataset.
    """
    rng = np.random.default_rng(seed)

    rep_ids = rng.choice(range(1, n_reps + 1), size=n_records, replace=True)
    hcp_ids = rng.choice(range(1, n_hcps + 1), size=n_records, replace=True)

    # Rep-level features
    rep_experience = rng.uniform(1, 20, size=n_reps + 1)  # years
    rep_territory_size = rng.integers(20, 200, size=n_reps + 1)

    # HCP-level features
    hcp_specialty_idx = rng.integers(0, 4, size=n_hcps + 1)
    specialties = ["Cardiology", "Oncology", "Neurology", "Primary Care"]
    hcp_patient_volume = rng.poisson(80, size=n_hcps + 1).clip(10, 500)
    hcp_digital_affinity = rng.beta(2, 3, size=n_hcps + 1)  # 0-1 score

    # Assign suggestion types with distribution reflecting real scenarios
    suggestion_probs = [0.25, 0.25, 0.20, 0.30]  # Email, Call, Insights, All
    suggestion_types = rng.choice(SUGGESTION_TYPES, size=n_records, p=suggestion_probs)

    # Determine action type based on suggestion type
    action_types = []
    for sug in suggestion_types:
        if sug == "All":
            # When "All" is suggested, rep may do any action or all
            action_types.append(rng.choice(ACTION_TYPES, p=[0.20, 0.25, 0.15, 0.40]))
        else:
            # When specific suggestion, rep mostly follows or does nothing
            if rng.random() < 0.75:
                action_types.append(sug)
            else:
                action_types.append(rng.choice(ACTION_TYPES, p=[0.25, 0.25, 0.25, 0.25]))
    action_types = np.array(action_types)

    # Build feature vectors
    experience = rep_experience[rep_ids]
    territory = rep_territory_size[rep_ids]
    specialty = np.array([specialties[hcp_specialty_idx[h]] for h in hcp_ids])
    patient_vol = hcp_patient_volume[hcp_ids]
    digital_aff = hcp_digital_affinity[hcp_ids]

    # --- Stage 1: Suggestion Accepted (Action Taken) ---
    # Whether the rep acted on the suggestion at all
    logit_action = (
        -0.5
        + 0.04 * experience
        + 0.003 * patient_vol
        + 1.2 * digital_aff
        - 0.002 * territory
        + np.where(suggestion_types == "All", 0.3, 0.0)
        + np.where(suggestion_types == "Email", 0.1, 0.0)
        + np.where(suggestion_types == "Call", -0.05, 0.0)
        + rng.normal(0, 0.3, size=n_records)
    )
    action_prob = 1 / (1 + np.exp(-logit_action))
    suggestion_accepted = (rng.random(n_records) < action_prob).astype(int)

    # --- Stage 2: Incremental Actions count ---
    # Number of incremental outreach actions taken (0 if not accepted)
    base_actions = np.where(
        suggestion_accepted == 1,
        rng.poisson(
            np.where(action_types == "All", 4.0,
            np.where(action_types == "Call", 2.5,
            np.where(action_types == "Email", 3.0, 1.8))),
        ),
        0,
    )
    incremental_actions = base_actions.clip(0, 20)

    # --- Stage 3: Incremental Outcomes (TRX and NBRX) ---
    # TRX uplift depends on action intensity, HCP patient volume, suggestion-action alignment
    alignment_bonus = np.where(
        (suggestion_types == action_types) | (suggestion_types == "All"),
        1.3,
        0.8,
    )

    trx_noise = rng.normal(0, 1.5, size=n_records)
    incremental_trx = np.where(
        suggestion_accepted == 1,
        (
            base_trx_effect
            * alignment_bonus
            * (1 + 0.01 * patient_vol)
            * (0.5 + 0.5 * incremental_actions / 4.0)
            + trx_noise
        ).clip(0, None),
        rng.exponential(0.5, size=n_records).clip(0, 3),  # small baseline even without action
    )
    incremental_trx = np.round(incremental_trx, 2)

    nbrx_noise = rng.normal(0, 0.6, size=n_records)
    incremental_nbrx = np.where(
        suggestion_accepted == 1,
        (
            base_nbrx_effect
            * alignment_bonus
            * (1 + 0.005 * patient_vol)
            * (0.4 + 0.6 * incremental_actions / 4.0)
            + nbrx_noise
        ).clip(0, None),
        rng.exponential(0.2, size=n_records).clip(0, 1.5),
    )
    incremental_nbrx = np.round(incremental_nbrx, 2)

    # Pre/post period baseline metrics
    baseline_trx = rng.poisson(15, size=n_records).clip(0, 100).astype(float)
    baseline_nbrx = rng.poisson(4, size=n_records).clip(0, 30).astype(float)
    post_trx = np.round(baseline_trx + incremental_trx, 2)
    post_nbrx = np.round(baseline_nbrx + incremental_nbrx, 2)

    df = pd.DataFrame(
        {
            "Rep_ID": rep_ids,
            "HCP_ID": hcp_ids,
            "Rep_Experience_Years": np.round(experience, 1),
            "Territory_Size": territory,
            "HCP_Specialty": specialty,
            "HCP_Patient_Volume": patient_vol,
            "HCP_Digital_Affinity": np.round(digital_aff, 3),
            "Suggestion_Type": suggestion_types,
            "Action_Type": action_types,
            "Suggestion_Accepted": suggestion_accepted,
            "Incremental_Actions": incremental_actions,
            "Baseline_TRX": baseline_trx,
            "Post_TRX": post_trx,
            "Incremental_TRX": incremental_trx,
            "Baseline_NBRX": baseline_nbrx,
            "Post_NBRX": post_nbrx,
            "Incremental_NBRX": incremental_nbrx,
        }
    )

    return df


def filter_by_combo(df: pd.DataFrame, suggestion_type: str, action_type: str) -> pd.DataFrame:
    """
    Filter dataset by suggestion-action combination.

    Parameters
    ----------
    df : pd.DataFrame
        Full outreach dataset.
    suggestion_type : str
        One of: 'Email', 'Call', 'Insights', 'All'.
    action_type : str
        One of: 'Email', 'Call', 'Insights', 'All'.

    Returns
    -------
    pd.DataFrame
        Filtered subset.
    """
    mask = pd.Series(True, index=df.index)

    if suggestion_type != "All":
        mask &= df["Suggestion_Type"] == suggestion_type
    if action_type != "All":
        mask &= df["Action_Type"] == action_type

    return df[mask].copy()
