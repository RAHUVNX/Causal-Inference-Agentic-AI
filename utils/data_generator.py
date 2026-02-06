"""
Synthetic data generator for causal inference scenarios.

Generates realistic marketing campaign / intervention datasets with
known treatment effects so users can validate their causal models.
"""

import numpy as np
import pandas as pd


def generate_marketing_campaign_data(
    n_samples: int = 5000,
    treatment_effect: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic marketing campaign dataset with known causal effect.

    Simulates a scenario where a company sends promotional offers (treatment)
    to customers and measures conversion (outcome). Confounders include
    customer demographics and past behaviour.

    Args:
        n_samples: Number of observations.
        treatment_effect: True average treatment effect on conversion probability.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
            customer_id, age, income, recency, frequency, monetary,
            channel (email/sms/push), treatment (0/1), conversion (0/1),
            revenue, true_ite (individual treatment effect used in generation).
    """
    rng = np.random.RandomState(seed)

    age = rng.normal(40, 12, n_samples).clip(18, 80).astype(int)
    income = (rng.lognormal(10.5, 0.6, n_samples)).astype(int)
    recency = rng.exponential(30, n_samples).clip(1, 365).astype(int)
    frequency = rng.poisson(5, n_samples).clip(0, 50)
    monetary = (rng.gamma(2, 50, n_samples)).round(2)
    channel = rng.choice(["email", "sms", "push"], n_samples, p=[0.5, 0.3, 0.2])

    # Treatment assignment is confounded: higher-income, more-frequent
    # buyers are more likely to receive the promotion.
    propensity_logit = (
        -1.0
        + 0.3 * (income / 100000)
        + 0.1 * (frequency / 10)
        - 0.005 * recency
        + 0.2 * (channel == "email").astype(float)
    )
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = rng.binomial(1, propensity)

    # Heterogeneous treatment effect: younger, higher-income customers
    # respond more to the promotion.
    ite = treatment_effect * (
        1.0
        + 0.5 * ((income - income.mean()) / income.std())
        + 0.3 * ((age.mean() - age) / age.std())
    )
    ite = ite.clip(0, 0.5)

    # Base conversion probability (without treatment).
    base_logit = (
        -2.0
        + 0.4 * (income / 100000)
        + 0.15 * (frequency / 10)
        - 0.003 * recency
        + 0.1 * (monetary / 100)
    )
    base_prob = 1 / (1 + np.exp(-base_logit))

    conversion_prob = base_prob + treatment * ite
    conversion_prob = conversion_prob.clip(0, 1)
    conversion = rng.binomial(1, conversion_prob)

    # Revenue conditional on conversion.
    revenue = conversion * rng.lognormal(3.5, 0.8, n_samples).round(2)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_samples + 1),
            "age": age,
            "income": income,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "channel": channel,
            "treatment": treatment,
            "conversion": conversion,
            "revenue": revenue,
            "true_ite": ite.round(4),
        }
    )
    return df


def generate_ab_test_data(
    n_samples: int = 3000,
    lift_pct: float = 12.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a clean A/B test dataset (randomised assignment).

    Args:
        n_samples: Number of observations.
        lift_pct: Percentage lift of treatment over control conversion rate.
        seed: Random seed.

    Returns:
        DataFrame with treatment, covariates, and outcome.
    """
    rng = np.random.RandomState(seed)

    age = rng.normal(35, 10, n_samples).clip(18, 70).astype(int)
    days_since_signup = rng.exponential(180, n_samples).clip(1, 1000).astype(int)
    page_views = rng.poisson(8, n_samples)

    treatment = rng.binomial(1, 0.5, n_samples)

    base_rate = 0.15
    lift = lift_pct / 100.0
    conversion_prob = base_rate + treatment * (base_rate * lift)
    conversion = rng.binomial(1, conversion_prob)

    return pd.DataFrame(
        {
            "user_id": np.arange(1, n_samples + 1),
            "age": age,
            "days_since_signup": days_since_signup,
            "page_views": page_views,
            "treatment": treatment,
            "conversion": conversion,
        }
    )
