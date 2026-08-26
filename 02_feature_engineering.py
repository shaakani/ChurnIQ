"""
ChurnIQ - Step 2: Feature engineering

Derives modelling features from the raw customer dataset:
- tenure_days: signup to today (or to last_active for churned users' context)
- watch_per_fee_ratio: engagement relative to what they're paying
- is_heavy_complainer: 2+ complaints flag
- days_since_active: recency signal
- one-hot encoding for gender, country, subscription_type
"""

import pandas as pd
from datetime import datetime

TODAY = datetime(2026, 8, 19)

df = pd.read_csv("churniq_customer_data.csv", parse_dates=["signup_date", "last_active_date"])

df["tenure_days"] = (TODAY - df["signup_date"]).dt.days
df["watch_per_fee_ratio"] = df["average_watch_hours"] / df["monthly_fee"]
df["is_heavy_complainer"] = (df["complaints_raised"] >= 2).astype(int)
df["is_new_customer"] = (df["tenure_days"] < 90).astype(int)

# NOTE: last_active_date / days_since_active is deliberately EXCLUDED as a model
# feature. In this dataset (and in most real churn setups), "days since last
# active" is a direct symptom of having already churned, not a predictor
# available before the fact - including it causes data leakage (label leakage),
# which is why an early version of this pipeline produced a suspicious 1.00 AUC.
# Only genuinely "before the fact" behavioural signals are used for modelling.

# one-hot encode categoricals
df_encoded = pd.get_dummies(
    df,
    columns=["gender", "country", "subscription_type"],
    drop_first=True,
)

# drop raw date columns and ID (not model features)
model_df = df_encoded.drop(columns=["signup_date", "last_active_date", "user_id"])

model_df.to_csv("churniq_features.csv", index=False)

print(f"Engineered dataset shape: {model_df.shape}")
print(f"Feature columns: {[c for c in model_df.columns if c != 'is_churned']}")
print(f"Missing values: {model_df.isna().sum().sum()}")
