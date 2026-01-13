import streamlit as st
import pandas as pd
import os
from datetime import datetime



st.set_page_config(page_title="AI-based CRM Decision System", layout="centered")

st.title("AI-based CRM Churn & Action Decision System")
st.write("Rule-based AI CRM engine using business logic and customer permissions.")

# -----------------------------
# INPUTS
# -----------------------------
st.subheader("Customer Factors")

tenure = st.number_input("Tenure (months)", 0, 60, value=6)
satisfaction = st.slider("Satisfaction Score (1–5)", 1, 5, value=3)
days_since = st.number_input("Days Since Last Order", 0, 365, value=10)
order_count = st.number_input("Order Count", 0, 50, value=3)
coupon_used = st.number_input("Coupon Used (count)", 0, 20, value=1)
complain = st.selectbox("Complain (0 = No, 1 = Yes)", [0, 1])
order_amount_hike = st.number_input("Order Amount Hike From Last Year (%)", 0, 100, value=10)

st.subheader("Customer Communication Permissions")
allow_email = st.selectbox("Email Permission", [0, 1])
allow_sms = st.selectbox("SMS Permission", [0, 1])
allow_push = st.selectbox("Push Notification Permission", [0, 1])
preferred_device = st.selectbox("Preferred Device", ["Mobile", "Computer"])

# -----------------------------
# SCORING LOGIC
# -----------------------------
def score_factors():
    factors = {}

    # Satisfaction
    if satisfaction <= 2:
        factors["SatisfactionScore"] = {"score": 3, "used": True}
    elif satisfaction == 3:
        factors["SatisfactionScore"] = {"score": 2, "used": True}
    else:
        factors["SatisfactionScore"] = {"score": 1, "used": True}

    # Complain
    factors["Complain"] = {"score": 3 if complain == 1 else 1, "used": True}

    # Order Amount Hike
    if order_amount_hike == 0:
        factors["OrderAmountHike"] = {"score": 3, "used": True}
    elif order_amount_hike <= 14:
        factors["OrderAmountHike"] = {"score": 2, "used": True}
    else:
        factors["OrderAmountHike"] = {"score": 1, "used": True}

    # Coupon Used
    if coupon_used <= 2:
        factors["CouponUsed"] = {"score": 3, "used": True}
    elif coupon_used <= 6:
        factors["CouponUsed"] = {"score": 2, "used": True}
    else:
        factors["CouponUsed"] = {"score": 1, "used": True}

    # Order Count
    if order_count <= 5:
        factors["OrderCount"] = {"score": 3, "used": True}
    elif order_count <= 9:
        factors["OrderCount"] = {"score": 2, "used": True}
    else:
        factors["OrderCount"] = {"score": 1, "used": True}

    # Days Since Last Order
    if days_since >= 15:
        factors["DaysSinceLastOrder"] = {"score": 3, "used": True}
    elif days_since >= 7:
        factors["DaysSinceLastOrder"] = {"score": 2, "used": True}
    else:
        factors["DaysSinceLastOrder"] = {"score": 1, "used": True}

    return factors


def overall_segment_from_avg(factor_segments):
    scores = []
    used = []

    for k, v in factor_segments.items():
        if v["used"]:
            scores.append(v["score"])
            used.append(k)

    avg = sum(scores) / len(scores)

    if avg >= 2.34:
        seg = "High"
    elif avg >= 1.67:
        seg = "Medium"
    else:
        seg = "Low"

    return seg, avg, used


# -----------------------------
# CRM ACTION LOGIC
# -----------------------------
def crm_action(segment):
    channels = []

    if segment == "High":
        if allow_email:
            channels.append("Email")
        if allow_sms:
            channels.append("SMS")
        if allow_push:
            channels.append("Push")

        action = "Send discount + win-back campaign"

    elif segment == "Medium":
        if preferred_device == "Mobile" and allow_push:
            channels.append("Push")
        elif allow_email:
            channels.append("Email")

        action = "Reminder + personalized offer"

    else:
        if allow_push:
            channels.append("Push")
        elif allow_email:
            channels.append("Email")

        action = "Regular campaign notification"

    if not channels:
        return action, "No communication sent (no permission)"

    return action, ", ".join(channels)


# -----------------------------
# RUN
# -----------------------------
if st.button("Generate CRM Recommendation"):
    factors = score_factors()
    segment, avg_score, used_factors = overall_segment_from_avg(factors)
    action, channel = crm_action(segment)

    st.subheader("AI Decision Result")
    st.write(f"**Risk Segment:** {segment}")
    st.write(f"**Average Risk Score:** {round(avg_score, 2)}")
    st.write(f"**Used Factors:** {', '.join(used_factors)}")

    st.subheader("Recommended CRM Action")
    st.write(f"**Action:** {action}")
    st.write(f"**Channel:** {channel}")
    # -----------------------------
    # SAVE TO CSV
    # -----------------------------
    output = {
        "Tenure": tenure,
        "SatisfactionScore": satisfaction,
        "DaysSinceLastOrder": days_since,
        "OrderCount": order_count,
        "CouponUsed": coupon_used,
        "Complain": complain,
        "OrderAmountHike": order_amount_hike,
        "RiskSegment": segment,
        "AverageRiskScore": round(avg_score, 2),
        "Action": action,
        "Channel": channel,
        "Timestamp": datetime.now(),
        "AllowEmail": allow_email,
        "AllowSMS": allow_sms,
        "AllowPush": allow_push,
        "PreferredDevice": preferred_device

    }

    output_df = pd.DataFrame([output])

    FILE_NAME = "crm_app_outputs.csv"

    if os.path.exists(FILE_NAME):
        output_df.to_csv(FILE_NAME, mode="a", header=False, index=False)
    else:
        output_df.to_csv(FILE_NAME, index=False)

    st.success("Result saved to crm_app_outputs.csv")


    if "No communication" in channel:
        st.warning("Customer has no permission. Action cannot be executed.")


