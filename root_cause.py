"""
root_cause.py
Finds features most correlated with the sensitive attribute — potential bias sources.
"""

import pandas as pd
import numpy as np


def _encode_if_needed(series: pd.Series) -> pd.Series:
    """Label-encode a categorical/string series to numeric for correlation."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    # Covers object, string, category dtypes
    return series.astype("category").cat.codes.astype(float)


def correlation_with_sensitive(
    df: pd.DataFrame,
    sensitive_col: str,
    target_col: str,
    top_n: int = 2,
) -> list[dict]:
    """
    Returns the top-N features (excluding target & sensitive) most correlated
    with the sensitive attribute.
    """
    sensitive_encoded = _encode_if_needed(df[sensitive_col])

    results = []
    for col in df.columns:
        if col in [sensitive_col, target_col]:
            continue
        try:
            col_encoded = _encode_if_needed(df[col])
            corr = col_encoded.corr(sensitive_encoded)
            if not np.isnan(corr):
                results.append({"feature": col, "correlation": round(corr, 4)})
        except Exception:
            continue

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    top = results[:top_n]

    for item in top:
        abs_corr = abs(item["correlation"])
        if abs_corr >= 0.5:
            item["risk"] = "High"
        elif abs_corr >= 0.2:
            item["risk"] = "Medium"
        else:
            item["risk"] = "Low"

    return top


def class_imbalance_check(df: pd.DataFrame, sensitive_col: str) -> dict:
    """Checks whether the sensitive attribute groups are imbalanced (>60/40 split)."""
    counts = df[sensitive_col].value_counts(normalize=True)
    max_ratio = counts.max()
    return {
        "distribution": counts.round(4).to_dict(),
        "is_imbalanced": bool(max_ratio > 0.60),
        "dominant_group": counts.idxmax(),
        "dominant_ratio": round(max_ratio, 4),
    }
