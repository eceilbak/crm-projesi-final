# data.py
# Purpose:
# - Load the e-commerce churn dataset used in the paper
# - Understand how churn=1 and churn=0 customers differ
# - Support the business rules used in the CRM decision engine

import pandas as pd
FILE_NAME = "E Commerce Dataset.csv"
SEP = ";"
df = pd.read_csv("E Commerce Dataset.csv", sep=";")
df.columns = [c.strip() for c in df.columns]

print(df.head())


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(FILE_NAME, sep=SEP)
df.columns = [c.strip() for c in df.columns]

print("Dataset loaded")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

# -----------------------------
# BASIC CHECKS
# -----------------------------
print("Target (Churn) distribution:")
print(df["Churn"].value_counts())
print()

# -----------------------------
# FACTORS USED IN CRM RULES
# -----------------------------
FACTORS = [
    "SatisfactionScore",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder"
]

print("Summary statistics for selected factors:")
print(df[FACTORS].describe())
print()

# -----------------------------
# COMPARISON: Churn=1 vs Churn=0
# -----------------------------
print("Average values by churn group:")
group_means = df.groupby("Churn")[FACTORS].mean()
print(group_means)
print()

# -----------------------------
# OPTIONAL: MEDIAN COMPARISON
# -----------------------------
print("Median values by churn group:")
group_medians = df.groupby("Churn")[FACTORS].median()
print(group_medians)
print()

# -----------------------------
# SIMPLE INSIGHT PRINTS
# -----------------------------
print("Quick insights:")
if group_means.loc[1, "SatisfactionScore"] < group_means.loc[0, "SatisfactionScore"]:
    print("- Churned customers tend to have lower SatisfactionScore.")

if group_means.loc[1, "DaySinceLastOrder"] > group_means.loc[0, "DaySinceLastOrder"]:
    print("- Churned customers tend to have higher DaySinceLastOrder.")

if group_means.loc[1, "Complain"] > group_means.loc[0, "Complain"]:
    print("- Complaints are more common among churned customers.")

print("\nData analysis finished.")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(BASE_DIR, "crm_ai_output.csv")
df.to_csv(out_path, index=False)
print("Saved to:", out_path)