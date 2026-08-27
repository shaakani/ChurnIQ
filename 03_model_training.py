"""
ChurnIQ - Step 3: Model training & evaluation

Trains and compares two models:
1. Logistic Regression (interpretable baseline)
2. Random Forest (non-linear, usually stronger on this kind of tabular data)

Reports whatever AUC, precision/recall, and feature importances actually
come out of training - nothing here is tuned to hit a specific target number.
Saves an ROC curve plot and a results CSV.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, precision_recall_fscore_support
)

SEED = 42

df = pd.read_csv("churniq_features.csv")
y = df["is_churned"]
X = df.drop(columns=["is_churned"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)

# --- Logistic Regression (scaled) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=SEED)
logreg.fit(X_train_scaled, y_train)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_proba)

# --- Random Forest ---
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=15,
    random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"Logistic Regression AUC: {logreg_auc:.4f}")
print(f"Random Forest AUC:       {rf_auc:.4f}")

best_name = "Random Forest" if rf_auc >= logreg_auc else "Logistic Regression"
best_proba = rf_proba if rf_auc >= logreg_auc else logreg_proba
best_auc = max(rf_auc, logreg_auc)
best_pred = (best_proba >= 0.5).astype(int)

print(f"\nBest model: {best_name} (AUC = {best_auc:.4f})")
print("\nClassification report (best model, threshold=0.5):")
print(classification_report(y_test, best_pred, target_names=["Retained", "Churned"]))

cm = confusion_matrix(y_test, best_pred)
print("Confusion matrix:")
print(f"                Predicted Retained  Predicted Churned")
print(f"Actual Retained        {cm[0][0]:>6}              {cm[0][1]:>6}")
print(f"Actual Churned         {cm[1][0]:>6}              {cm[1][1]:>6}")

# --- Feature importance (Random Forest) ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 8 features (Random Forest importance):")
print(importances.head(8).to_string())

# --- ROC curve plot ---
fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={logreg_auc:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ChurnIQ - ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("churniq_roc_curve.png", dpi=150)
print("\nSaved ROC curve to churniq_roc_curve.png")

# --- Save results for downstream revenue script ---
results_df = X_test.copy()
results_df["actual_churn"] = y_test.values
results_df["predicted_proba"] = best_proba
results_df["predicted_churn"] = best_pred
results_df.to_csv("churniq_test_predictions.csv", index=False)

summary = pd.DataFrame({
    "model": ["Logistic Regression", "Random Forest"],
    "auc": [logreg_auc, rf_auc],
})
summary.to_csv("churniq_model_summary.csv", index=False)
importances.to_csv("churniq_feature_importance.csv", header=["importance"])

print(f"\nBest model used for downstream revenue analysis: {best_name}")
print(f"Test set size: {len(y_test):,} customers ({len(X_train):,} in training set)")

# --- Persist the winning model for serving (API/Docker) ---
# Saved together so the serving layer always knows exactly what to load and
# how to preprocess incoming requests - whichever model won this run.
model_bundle = {
    "model_type": best_name,                     # "Logistic Regression" or "Random Forest"
    "model": logreg if best_name == "Logistic Regression" else rf,
    "scaler": scaler,                             # only applied at inference time if model_type is Logistic Regression
    "feature_columns": list(X.columns),           # exact column order the model expects
    "auc": float(best_auc),
    "threshold": 0.5,
}
joblib.dump(model_bundle, "churniq_model.joblib")
with open("churniq_model_metadata.json", "w") as f:
    json.dump({
        "model_type": best_name,
        "auc": round(float(best_auc), 4),
        "feature_columns": list(X.columns),
        "threshold": 0.5,
        "trained_on_rows": int(len(X_train) + len(X_test)),
    }, f, indent=2)
print(f"\nSaved model bundle to churniq_model.joblib ({best_name}, AUC={best_auc:.4f})")
print("Saved churniq_model_metadata.json")
