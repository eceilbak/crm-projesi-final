# grafikleri_ml.py
# Purpose:
# - Read E Commerce Dataset.csv (sep=";")
# - Train an ML model (Logistic Regression + StandardScaler)
# - Use predict_proba outputs as churn_probability
# - Produce the same 4 charts, but based on ML predictions (not heuristic scoring)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


print("grafikleri_ml.py başladı")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "E Commerce Dataset.csv")

# -----------------------------
# 1) READ DATA
# -----------------------------
df = pd.read_csv(csv_path, sep=";")
df.columns = [c.strip() for c in df.columns]
print("CSV okundu:", len(df), "satır,", len(df.columns), "kolon")

# -----------------------------
# 2) ML CONFIG
# -----------------------------
TARGET = "Churn"
FEATURES = [
    "Tenure",
    "SatisfactionScore",
    "DaySinceLastOrder",
    "OrderCount",
    "CouponUsed",
    "Complain",
]

missing = [c for c in [TARGET] + FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Eksik kolon(lar): {missing}")

# numeric cast + imputation (median)
for c in [TARGET] + FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in FEATURES:
    df[c] = df[c].fillna(df[c].median())

# target cleaning
df[TARGET] = df[TARGET].fillna(df[TARGET].median()).clip(0, 1).astype(int)

X = df[FEATURES].copy()
y = df[TARGET].copy()

# -----------------------------
# 3) TRAIN MODEL (LR baseline)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ]
)

model.fit(X_train, y_train)

test_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, test_prob)
print("Model trained. Test ROC-AUC:", round(auc, 4))

# -----------------------------
# 4) PREDICT PROBABILITIES FOR ALL ROWS
# -----------------------------
df["churn_probability"] = model.predict_proba(df[FEATURES])[:, 1]

# -----------------------------
# 5) GRAPH 1 — Probability Distribution (ML-based)
# -----------------------------
plt.figure()
plt.hist(df["churn_probability"], bins=30)
plt.title("Predicted Churn Probability Distribution (ML)")
plt.savefig(os.path.join(BASE_DIR, "fig1_probability_distribution.png"))
plt.close()

# -----------------------------
# 6) Risk Segment (threshold-based)
# NOTE: thresholds are business choices; keep your current ones or revise later.
# -----------------------------
def risk_segment(p: float) -> str:
    if p >= 0.66:
        return "High"
    elif p >= 0.33:
        return "Medium"
    else:
        return "Low"

df["RiskSegment"] = df["churn_probability"].apply(risk_segment)

# -----------------------------
# 7) GRAPH 2 — Boxplot by Segment
# -----------------------------
plt.figure()
df.boxplot(column="churn_probability", by="RiskSegment")
plt.suptitle("")
plt.title("Risk Segment vs Predicted Probability (ML)")
plt.savefig(os.path.join(BASE_DIR, "fig2_risk_segment_boxplot.png"))
plt.close()

# -----------------------------
# 8) GRAPH 3 — Feature Impact (Model-based for Logistic Regression)
# - Better than correlation for “model effect” in LR
# - Uses absolute coefficient magnitude (on standardized scale)
# -----------------------------
clf = model.named_steps["clf"]
coefs = pd.Series(np.abs(clf.coef_.reshape(-1)), index=FEATURES).sort_values()

plt.figure()
coefs.plot(kind="barh")
plt.title("Feature Impact on Churn (|LogReg Coefs|, standardized)")
plt.savefig(os.path.join(BASE_DIR, "fig3_feature_impact.png"))
plt.close()

# -----------------------------
# 9) GRAPH 4 — CRM Action Coverage by Segment
# -----------------------------
def crm_action(seg: str) -> str:
    if seg == "High":
        return "Win-back Campaign"
    elif seg == "Medium":
        return "Personalized Offer"
    else:
        return "Regular Campaign"

df["CRM_Action"] = df["RiskSegment"].apply(crm_action)

df.groupby(["RiskSegment", "CRM_Action"]).size().unstack(fill_value=0).plot(
    kind="bar", stacked=True
)
plt.title("CRM Action Coverage by Risk Segment (ML)")
plt.savefig(os.path.join(BASE_DIR, "fig4_crm_action_coverage.png"))
plt.close()

print("TÜM ML TABANLI GRAFİKLER KAYDEDİLDİ:")
print("- fig1_probability_distribution.png")
print("- fig2_risk_segment_boxplot.png")
print("- fig3_feature_impact.png")
print("- fig4_crm_action_coverage.png")
