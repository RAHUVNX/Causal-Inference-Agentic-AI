"""
Metrics computation for trained models.

Computes:
  - R^2 (coefficient of determination)
  - MAE (mean absolute error)
  - RMSE (root mean squared error)
  - Accuracy, Precision, Recall, F1 (binarized lift classification)
"""

import numpy as np


def compute_r_squared(actual: list[float], predicted: list[float]) -> float:
    a = np.array(actual)
    p = np.array(predicted)
    if len(a) == 0:
        return 0.0
    mean = a.mean()
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_mae(actual: list[float], predicted: list[float]) -> float:
    a = np.array(actual)
    p = np.array(predicted)
    if len(a) == 0:
        return 0.0
    return float(np.mean(np.abs(a - p)))


def compute_rmse(actual: list[float], predicted: list[float]) -> float:
    a = np.array(actual)
    p = np.array(predicted)
    if len(a) == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - p) ** 2)))


def compute_classification_metrics(
    observed: list[float], incremental: list[float]
) -> dict:
    """Binary classification metrics: positive lift vs no lift."""
    n = len(observed)
    if n == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    tp = fp = fn = tn = 0
    for i in range(n):
        actual_pos = observed[i] > 0
        pred_pos = incremental[i] > 0
        if pred_pos and actual_pos:
            tp += 1
        elif pred_pos and not actual_pos:
            fp += 1
        elif not pred_pos and actual_pos:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_full_metrics(lift_results: list[dict]) -> dict:
    """Compute full metrics report for both Path A and Path B."""
    observed_actions = [r["observedAction"] for r in lift_results]
    predicted_actions = [r["predictedAction"] for r in lift_results]
    incremental_actions = [r["incrementalAction"] for r in lift_results]

    observed_outcomes = [r["observedOutcome"] for r in lift_results]
    predicted_outcomes = [r["predictedOutcome"] for r in lift_results]
    incremental_outcomes = [r["incrementalOutcome"] for r in lift_results]

    path_a_class = compute_classification_metrics(observed_actions, incremental_actions)
    path_b_class = compute_classification_metrics(
        observed_outcomes, incremental_outcomes
    )

    return {
        "pathA": {
            "rSquared": compute_r_squared(observed_actions, predicted_actions),
            "mae": compute_mae(observed_actions, predicted_actions),
            "rmse": compute_rmse(observed_actions, predicted_actions),
            **path_a_class,
        },
        "pathB": {
            "rSquared": compute_r_squared(observed_outcomes, predicted_outcomes),
            "mae": compute_mae(observed_outcomes, predicted_outcomes),
            "rmse": compute_rmse(observed_outcomes, predicted_outcomes),
            **path_b_class,
        },
    }
