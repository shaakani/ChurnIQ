"""
ChurnIQ - Step 1: Generate synthetic customer dataset

Builds a synthetic streaming-service customer base using the same column
structure as a real customer dataset (user_id, age, gender, signup_date,
last_active_date, country, subscription_type, average_watch_hours,
mobile_app_usage_pct, complaints_raised, received_promotions,
referred_by_friend, is_churned, monthly_fee).

Unlike a toy dataset, churn here is generated from an underlying logistic
function of several features PLUS random noise - so there is genuine signal,
but not perfect separation. The actual AUC a model achieves on this data is
not hardcoded or targeted; it falls out of whatever the training script
finds, same as any real modelling exercise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

N = 7500

COUNTRIES = ["USA", "UK", "Canada", "Ireland", "Germany", "Australia", "India", "France"]
SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]
SUB_FEE = {"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}
GENDERS = ["Male", "Female", "Other"]

# --- base attributes -------------------------------------------------
user_id = np.arange(1, N + 1)
age = np.random.randint(18, 75, size=N)
gender = np.random.choice(GENDERS, size=N, p=[0.48, 0.48, 0.04])
country = np.random.choice(COUNTRIES, size=N)
subscription_type = np.random.choice(SUBSCRIPTION_TYPES, size=N, p=[0.4, 0.4, 0.2])

# signup dates spread over the last ~3 years
today = datetime(2026, 8, 19)
signup_days_ago = np.random.randint(30, 1100, size=N)
signup_date = [today - timedelta(days=int(d)) for d in signup_days_ago]

# engagement + behaviour signals
average_watch_hours = np.clip(np.random.gamma(shape=2.2, scale=4.0, size=N), 0, 60)
mobile_app_usage_pct = np.clip(np.random.beta(2, 2, size=N) * 100, 0, 100)
complaints_raised = np.random.poisson(lam=0.6, size=N)
received_promotions = np.random.binomial(1, 0.35, size=N)
referred_by_friend = np.random.binomial(1, 0.22, size=N)
monthly_fee = np.array([SUB_FEE[s] for s in subscription_type]) + np.random.normal(0, 0.4, N).round(2)
monthly_fee = np.round(np.clip(monthly_fee, 5, None), 2)

tenure_days = signup_days_ago  # helper for churn model, not a final column yet

# --- churn probability: real logistic structure + noise --------------
# Standardise inputs for a clean logit combination
z_watch = (average_watch_hours - average_watch_hours.mean()) / average_watch_hours.std()
z_mobile = (mobile_app_usage_pct - mobile_app_usage_pct.mean()) / mobile_app_usage_pct.std()
z_tenure = (tenure_days - tenure_days.mean()) / tenure_days.std()
z_fee = (monthly_fee - monthly_fee.mean()) / monthly_fee.std()

logit = (
    -0.9                      # baseline
    - 0.85 * z_watch          # more watch hours -> less churn
    - 0.55 * z_mobile         # more app usage -> less churn
    - 0.65 * z_tenure         # longer tenure -> less churn
    + 0.45 * z_fee            # higher fee -> more churn
    + 0.35 * complaints_raised
    - 0.50 * received_promotions
    - 0.30 * referred_by_friend
    + np.random.normal(0, 0.9, N)   # irreducible noise - keeps this from being trivially separable
)

churn_prob = 1 / (1 + np.exp(-logit))
is_churned = np.random.binomial(1, churn_prob)

# last_active_date: churned users went inactive earlier
last_active_days_ago = np.where(
    is_churned == 1,
    np.random.randint(15, 120, size=N),
    np.random.randint(0, 10, size=N),
)
last_active_days_ago = np.minimum(last_active_days_ago, tenure_days)
last_active_date = [today - timedelta(days=int(d)) for d in last_active_days_ago]

df = pd.DataFrame({
    "user_id": user_id,
    "age": age,
    "gender": gender,
    "signup_date": [d.strftime("%Y-%m-%d") for d in signup_date],
    "last_active_date": [d.strftime("%Y-%m-%d") for d in last_active_date],
    "country": country,
    "subscription_type": subscription_type,
    "average_watch_hours": average_watch_hours.round(2),
    "mobile_app_usage_pct": mobile_app_usage_pct.round(2),
    "complaints_raised": complaints_raised,
    "received_promotions": received_promotions,
    "referred_by_friend": referred_by_friend,
    "is_churned": is_churned,
    "monthly_fee": monthly_fee,
})

df.to_csv("churniq_customer_data.csv", index=False)

print(f"Generated {len(df):,} rows")
print(f"Churn rate: {df['is_churned'].mean():.2%}")
print(f"Columns: {list(df.columns)}")
print(df.head(3).to_string())
