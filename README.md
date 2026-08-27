# ChurnIQ — Customer Churn Prediction & Revenue Impact

A churn prediction pipeline for a simulated subscription streaming service: synthetic data generation with genuine (not guaranteed) predictive signal, feature engineering, model comparison, and business-facing revenue-at-risk quantification.

Built August 2026 as a personal project. This is separate from and does not represent any employer's data or deliverables — it uses the same general column structure as a real customer dataset I worked with previously, rebuilt as new synthetic data to practise and demonstrate the full churn-modelling workflow end to end.

## What it does

1. **`01_generate_data.py`** — Generates 7,500 synthetic customer records (age, gender, country, subscription type, watch hours, app usage, complaints, promotions, referrals, monthly fee, churn status) with churn probability driven by a logistic combination of engagement, tenure, and pricing signals plus random noise — so there's real structure to learn, but not perfect separation.
2. **`02_feature_engineering.py`** — Derives tenure, watch-hours-per-fee-dollar ratio, complaint flags, and one-hot encodes categoricals. Deliberately excludes `days_since_active`/`last_active_date` as a feature (see Data Leakage note below).
3. **`03_model_training.py`** — Trains and compares Logistic Regression and Random Forest classifiers on a stratified 75/25 train/test split; reports AUC, classification metrics, confusion matrix, and feature importances. Persists the winning model to `churniq_model.joblib` for serving.
4. **`04_revenue_impact.py`** — Translates the model's high-risk predictions into an annualised revenue-at-risk dollar figure, with the calculation method fully shown.
5. **`app.py`** — FastAPI service that loads `churniq_model.joblib` and serves real-time predictions over HTTP.

## Real results (this run)

- **7,500 customer records**, 33.9% churn rate
- **Logistic Regression AUC: 0.756** (Random Forest: 0.752 — logistic regression won on this data)
- Classification report at 0.5 threshold: 74% accuracy, 65% precision / 48% recall on the churned class
- Top predictive features: `watch_per_fee_ratio`, `tenure_days`, `average_watch_hours`, `mobile_app_usage_pct`
- **Revenue at risk: $320,576/year** (estimated across the full 7,500-customer base, scaled from 472 high-risk customers flagged in the 1,875-customer test set)

## A data leakage catch worth mentioning

The first version of this pipeline included `days_since_active` as a feature and produced a **1.00 AUC** — a dead giveaway of a broken model, not a good one. The root cause: in the generated data, "days since last active" is itself a direct consequence of having churned (churned customers were inactive 15-120 days; retained customers within 10 days), so the model was effectively being handed the answer. Removing that feature dropped AUC from a meaningless 1.00 to a genuine, defensible 0.756. This is a real and common failure mode in churn modelling — using a signal that's only known *after* the outcome you're trying to predict.

## Revenue calculation methodology

Annualised revenue at risk = (customers flagged high-risk by the model, i.e. predicted churn probability ≥ 0.5) × (their monthly subscription fee) × 12, computed on the held-out test set and scaled proportionally to the full customer base. This is disclosed explicitly rather than presented as a single unexplained number — the underlying `monthly_fee` values and flagged customers are all in `churniq_test_predictions.csv`.

## Honest limitations

- Data is synthetic, not from a real company — explicitly disclosed, same as ScaleSpark's simulated 5M-row dataset.
- 48% recall on the churned class means the model misses over half of actual churners at the default 0.5 threshold — a real limitation worth discussing, not glossing over. Recall could be improved by lowering the threshold at the cost of more false positives; that trade-off is a legitimate follow-up analysis.
- Revenue-at-risk figure depends on the 0.5 threshold choice and is an estimate scaled from a test sample, not a company's actual financials.

## Serving the model (API + Docker)

The trained model isn't just left in a notebook — it's wrapped in a FastAPI service (`app.py`) and containerised, so it can actually be called over HTTP rather than only re-run as a script.

**Run locally:**
```
pip install -r requirements-api.txt
uvicorn app:app --reload --port 8000
```
Interactive docs at `http://127.0.0.1:8000/docs`.

**Run in Docker:**
```
docker build -t churniq-api .
docker run -p 8000:8000 churniq-api
```

**Endpoints:** `GET /health` (liveness + model metadata), `POST /predict` (single customer), `POST /predict/batch` (list of customers).

**Real example (this run):**
```
$ curl http://127.0.0.1:8000/health
{"status":"ok","model_type":"Logistic Regression","model_auc":0.7559,"trained_on_rows":7500}

$ curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "age": 29, "average_watch_hours": 1.5, "mobile_app_usage_pct": 8.0,
  "complaints_raised": 3, "received_promotions": 0, "referred_by_friend": 0,
  "monthly_fee": 17.99, "tenure_days": 20, "gender": "Male",
  "country": "USA", "subscription_type": "Premium"
}'
{"churn_probability":0.9724,"predicted_churn":true,"risk_tier":"high","model_type":"Logistic Regression","model_auc":0.7559}
```

A new, low-engagement, short-tenure customer with complaints scores 97.2% churn probability; a long-tenure, highly-engaged customer with a referral and a promo scores 1% — directionally consistent with the feature importances found during training (`watch_per_fee_ratio`, `tenure_days`, `average_watch_hours` are the top drivers).

## Stack

Python · SQL · FastAPI · Docker · pandas · NumPy · scikit-learn · Matplotlib
