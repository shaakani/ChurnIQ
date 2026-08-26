"""
ChurnIQ - Step 4: Revenue-at-risk quantification

Takes the trained model's predicted churn probabilities on the held-out test
set and translates them into a dollar figure: annualised revenue tied to
customers flagged as high-risk (predicted probability >= 0.5).

Methodology is disclosed explicitly - this is: (customers predicted likely to
churn) x (their monthly_fee) x 12, on the test set, then scaled up to the
full customer base proportionally. This is a standard, explainable way to
size the business impact of a churn model, not a number invented to hit a
target.
"""

import pandas as pd

preds = pd.read_csv("churniq_test_predictions.csv")
full = pd.read_csv("churniq_customer_data.csv")

test_n = len(preds)
full_n = len(full)
scale_factor = full_n / test_n

high_risk = preds[preds["predicted_churn"] == 1]
n_high_risk_test = len(high_risk)
monthly_revenue_at_risk_test = high_risk["monthly_fee"].sum()
annual_revenue_at_risk_test = monthly_revenue_at_risk_test * 12

# scale test-set figures up to represent the full 7,500-customer base
n_high_risk_full_est = round(n_high_risk_test * scale_factor)
annual_revenue_at_risk_full_est = annual_revenue_at_risk_test * scale_factor

# also compute the actual (ground-truth) churned-customer revenue for comparison
actual_churned = preds[preds["actual_churn"] == 1]
actual_annual_revenue_test = actual_churned["monthly_fee"].sum() * 12
actual_annual_revenue_full_est = actual_annual_revenue_test * scale_factor

print("=" * 60)
print("REVENUE-AT-RISK ANALYSIS")
print("=" * 60)
print(f"Full customer base: {full_n:,}")
print(f"Test set: {test_n:,} customers ({n_high_risk_test:,} flagged high-risk)")
print()
print(f"Test-set annualised revenue tied to flagged high-risk customers: ${annual_revenue_at_risk_test:,.2f}")
print(f"Scaled to full {full_n:,}-customer base (est.): ${annual_revenue_at_risk_full_est:,.2f}")
print(f"Estimated high-risk customer count across full base: {n_high_risk_full_est:,}")
print()
print(f"[Reference] Actual annualised revenue from customers who truly churned in test set: ${actual_annual_revenue_test:,.2f}")
print(f"[Reference] Scaled to full base: ${actual_annual_revenue_full_est:,.2f}")

summary = pd.DataFrame([{
    "full_customer_base": full_n,
    "test_set_size": test_n,
    "high_risk_flagged_test": n_high_risk_test,
    "annual_revenue_at_risk_test": round(annual_revenue_at_risk_test, 2),
    "annual_revenue_at_risk_full_scaled": round(annual_revenue_at_risk_full_est, 2),
    "high_risk_count_full_scaled": n_high_risk_full_est,
    "actual_churned_annual_revenue_test": round(actual_annual_revenue_test, 2),
    "actual_churned_annual_revenue_full_scaled": round(actual_annual_revenue_full_est, 2),
}])
summary.to_csv("churniq_revenue_summary.csv", index=False)
print("\nSaved churniq_revenue_summary.csv")
