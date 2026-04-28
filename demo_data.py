"""
demo_data.py
Generates a realistic biased loan dataset for demo purposes.
"""

import pandas as pd
import numpy as np


def generate_demo_dataset(n=300, seed=42) -> pd.DataFrame:
    """
    Creates a synthetic loan-approval dataset with intentional gender bias.
    Men are approved at ~75%, women at ~35%.
    """
    rng = np.random.default_rng(seed)

    gender      = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    age         = rng.integers(22, 60, size=n)
    income      = np.where(
        gender == "Male",
        rng.integers(40000, 120000, size=n),
        rng.integers(30000, 100000, size=n),
    )
    credit_score = rng.integers(550, 850, size=n)
    loan_amount  = rng.integers(5000, 50000, size=n)
    employment_years = rng.integers(0, 20, size=n)

    # Biased approval logic
    base_prob = (credit_score - 550) / 300           # 0‥1 from credit score
    income_boost = (income - 30000) / 90000 * 0.3    # small income boost
    gender_bias  = np.where(gender == "Male", 0.25, -0.15)  # hard gender thumb

    approval_prob = np.clip(base_prob + income_boost + gender_bias, 0.05, 0.95)
    loan_approved = (rng.random(size=n) < approval_prob).astype(int)

    return pd.DataFrame({
        "gender":           gender,
        "age":              age,
        "income":           income,
        "credit_score":     credit_score,
        "loan_amount":      loan_amount,
        "employment_years": employment_years,
        "loan_approved":    loan_approved,
    })
