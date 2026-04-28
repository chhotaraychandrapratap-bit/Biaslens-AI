"""
fix_suggestions.py
Rule-based fix recommendations based on bias metrics and root-cause findings.
"""


THRESHOLD = 0.10


def generate_fixes(
    metrics: dict,
    correlated_features: list[dict],
    imbalance_info: dict,
) -> list[dict]:
    """
    Returns a prioritised list of fix suggestions with title, detail, and priority.
    """
    fixes = []

    dpd = abs(metrics["demographic_parity"]["value"])
    eod = abs(metrics["equal_opportunity"]["value"])

    # ── Imbalance fix ────────────────────────────────────────────────────────
    if imbalance_info["is_imbalanced"]:
        fixes.append({
            "priority": "High",
            "title": "Resampling / Re-weighting",
            "detail": (
                f"The dataset is imbalanced: {imbalance_info['dominant_group']} "
                f"makes up {imbalance_info['dominant_ratio']*100:.1f}% of records. "
                "Apply oversampling (SMOTE) on the minority group, or use class "
                "weights during model training to compensate."
            ),
            "icon": "⚖️",
        })

    # ── Correlated feature removal ────────────────────────────────────────────
    high_risk = [f for f in correlated_features if f["risk"] == "High"]
    if high_risk:
        names = ", ".join(f["feature"] for f in high_risk)
        fixes.append({
            "priority": "High",
            "title": f"Remove or De-correlate Feature(s): {names}",
            "detail": (
                f"Feature(s) [{names}] are strongly correlated with the sensitive "
                "attribute and act as proxies for it. Consider removing them or "
                "applying PCA / adversarial de-biasing to reduce this proxy effect."
            ),
            "icon": "✂️",
        })

    medium_risk = [f for f in correlated_features if f["risk"] == "Medium"]
    if medium_risk:
        names = ", ".join(f["feature"] for f in medium_risk)
        fixes.append({
            "priority": "Medium",
            "title": f"Audit Feature(s): {names}",
            "detail": (
                f"Feature(s) [{names}] show moderate correlation with the sensitive "
                "attribute. Monitor them and consider whether they introduce indirect "
                "discrimination via proxy relationships."
            ),
            "icon": "🔍",
        })

    # ── Demographic parity fix ────────────────────────────────────────────────
    if dpd > THRESHOLD:
        fixes.append({
            "priority": "High",
            "title": "Fairness-Aware Training (Demographic Parity)",
            "detail": (
                f"Demographic Parity Difference is {dpd:.3f} (> {THRESHOLD}). "
                "Apply a fairness constraint during model training (e.g., "
                "Reductions approach in Fairlearn, or post-processing "
                "threshold optimisation) to equalise positive-prediction rates."
            ),
            "icon": "🎯",
        })

    # ── Equal opportunity fix ─────────────────────────────────────────────────
    if eod > THRESHOLD:
        fixes.append({
            "priority": "High" if eod > 0.20 else "Medium",
            "title": "Equal Opportunity Calibration",
            "detail": (
                f"Equal Opportunity Difference is {eod:.3f}. "
                "Adjust decision thresholds per group to equalise true-positive "
                "rates, or use an in-processing fairness constraint that directly "
                "targets equalised opportunity."
            ),
            "icon": "🔧",
        })

    # ── Baseline good practice ────────────────────────────────────────────────
    fixes.append({
        "priority": "Low",
        "title": "Ongoing Fairness Monitoring",
        "detail": (
            "Even after applying fixes, re-evaluate fairness metrics every time "
            "the model or dataset changes. Set up automated bias dashboards in "
            "production (e.g., Evidently AI, WhyLabs) for continuous monitoring."
        ),
        "icon": "📊",
    })

    return fixes
