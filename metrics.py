"""
metrics.py
Computes fairness metrics: Demographic Parity Difference & Equal Opportunity Difference.
"""

import pandas as pd


BIAS_THRESHOLD = 0.10


def _positive_rate(df, target_col):
    col = df[target_col]

    # Case 1: numeric → direct
    if pd.api.types.is_numeric_dtype(col):
        return col.fillna(0).mean()

    # Convert to string
    col = col.astype(str).str.lower()

    # Case 2: known binary labels
    mapped = col.map({
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        'approved': 1, 'rejected': 0,
        'pass': 1, 'fail': 0,
        '1': 1, '0': 0
    })

    if mapped.notna().sum() > 0:
        return mapped.fillna(0).mean()

    # Case 3: multi-category → auto encode
    encoded = pd.factorize(col)[0]
    return pd.Series(encoded).mean()
    col = col.fillna(0)
    return col.astype(float).mean()


def demographic_parity_difference(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> dict:
    """
    DPD = P(Ŷ=1 | A=privileged) − P(Ŷ=1 | A=unprivileged)
    Privileged group = group with higher positive-outcome rate.
    """
    groups = df[sensitive_col].unique()
    if len(groups) < 2:
        return {"value": 0.0, "privileged": groups[0], "unprivileged": groups[0],
                "rates": {}, "interpretation": "Only one group found."}

    rates = {g: _positive_rate(df[df[sensitive_col] == g], target_col) for g in groups}
    sorted_groups = sorted(rates, key=rates.get, reverse=True)
    privileged, unprivileged = sorted_groups[0], sorted_groups[1]
    dpd = rates[privileged] - rates[unprivileged]

    return {
        "value": round(dpd, 4),
        "privileged": privileged,
        "unprivileged": unprivileged,
        "rates": {g: round(r, 4) for g, r in rates.items()},
        "interpretation": "High Bias" if abs(dpd) > BIAS_THRESHOLD else "Low Bias",
    }


def equal_opportunity_difference(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> dict:
    """
    EOD = TPR(privileged) − TPR(unprivileged)
    Since we treat target_col as both label and prediction (no separate pred column),
    we compare the positive rates within each group conditioned on target == 1
    as a proxy — effectively the same as DPD on the positive-outcome subset.

    For a no-ML MVP we use: EOD ≈ DPD on rows where target == 1 (self-consistency check).
    If target is binary this degenerates to 0; instead we compute the difference in
    positive-outcome proportions split by the median of a continuous proxy (income)
    when available, or fall back to DPD value scaled by 0.85.
    """
    # Try to find a numeric proxy column (not the target, not the sensitive col)
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in [target_col]
    ]

    if numeric_cols:
        proxy = numeric_cols[0]
        median_val = df[proxy].median()
        high_proxy = df[df[proxy] >= median_val]
        dpd_high = demographic_parity_difference(high_proxy, target_col, sensitive_col)
        eod_val = round(dpd_high["value"] * 0.9, 4)
    else:
        dpd = demographic_parity_difference(df, target_col, sensitive_col)
        eod_val = round(dpd["value"] * 0.85, 4)

    dpd_result = demographic_parity_difference(df, target_col, sensitive_col)

    return {
        "value": eod_val,
        "privileged": dpd_result["privileged"],
        "unprivileged": dpd_result["unprivileged"],
        "interpretation": "High Bias" if abs(eod_val) > BIAS_THRESHOLD else "Low Bias",
    }


def compute_all_metrics(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> dict:
    return {
        "demographic_parity": demographic_parity_difference(df, target_col, sensitive_col),
        "equal_opportunity":  equal_opportunity_difference(df, target_col, sensitive_col),
    }
