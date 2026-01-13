import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

print("grafikler.py başladı")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "E Commerce Dataset.csv")

df = pd.read_csv(csv_path, sep=";")
df.columns = [c.strip() for c in df.columns]

print("CSV okundu:", len(df))

np.random.seed(42)
df["churn_probability"] = (
    0.4 * (5 - df["SatisfactionScore"]) / 4 +
    0.3 * df["Complain"] +
    0.3 * df["DaySinceLastOrder"] / df["DaySinceLastOrder"].max()
)

# -----------------------------
# 1️⃣ Probability Distribution
# -----------------------------
plt.figure()
plt.hist(df["churn_probability"], bins=30)
plt.title("Predicted Churn Probability Distribution")
plt.savefig("fig1_probability_distribution.png")
plt.close()

# -----------------------------
# Risk Segment
# -----------------------------
def risk_segment(p):
    if p >= 0.66:
        return "High"
    elif p >= 0.33:
        return "Medium"
    else:
        return "Low"

df["RiskSegment"] = df["churn_probability"].apply(risk_segment)

# -----------------------------
# 2️⃣ Boxplot
# -----------------------------
plt.figure()
df.boxplot(column="churn_probability", by="RiskSegment")
plt.suptitle("")
plt.title("Risk Segment vs Predicted Probability")
plt.savefig("fig2_risk_segment_boxplot.png")
plt.close()

# -----------------------------
# 3️⃣ Feature Impact (Correlation)
# -----------------------------
FEATURES = [
    "SatisfactionScore",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder"
]

corr = df[FEATURES + ["churn_probability"]].corr()["churn_probability"].drop("churn_probability")

plt.figure()
corr.abs().sort_values().plot(kind="barh")
plt.title("Feature Impact on Churn (Correlation Based)")
plt.savefig("fig3_feature_impact.png")
plt.close()

# -----------------------------
# 4️⃣ CRM Action Coverage
# -----------------------------
def crm_action(seg):
    if seg == "High":
        return "Win-back Campaign"
    elif seg == "Medium":
        return "Personalized Offer"
    else:
        return "Regular Campaign"

df["CRM_Action"] = df["RiskSegment"].apply(crm_action)

df.groupby(["RiskSegment", "CRM_Action"]).size().unstack().plot(
    kind="bar", stacked=True
)
plt.title("CRM Action Coverage by Risk Segment")
plt.savefig("fig4_crm_action_coverage.png")
plt.close()

print("TÜM GRAFİKLER KAYDEDİLDİ")

