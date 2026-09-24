import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))

from search import get_recommendation
from explain import get_explanation
from clean_data import load_and_clean

st.set_page_config(page_title="SpotWise AI", layout="wide")

st.title("SpotWise AI – Intelligent Cloud Spot Instance Bidding Advisor")
st.write("Get an AI-recommended bid for your cloud spot instance job.")

st.divider()

# ---------------- INPUT FORM ----------------
st.subheader("Job Details")

col1, col2, col3 = st.columns(3)

with col1:
    provider = st.selectbox("Cloud Provider", ["AWS"])  # only AWS data available for now
    instance_type = st.selectbox(
        "Instance Type",
        ["m4.xlarge", "c3.4xlarge", "r3.large"]  # match your TARGET_INSTANCES from train_model.py
    )
    availability_zone = st.selectbox(
        "Availability Zone",
        ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1e"]
    )

with col2:
    operating_system = st.selectbox(
        "Operating System",
        ["Linux/UNIX", "Windows", "SUSE Linux"]
    )
    runtime_hours = st.number_input("Runtime (hours)", min_value=1, max_value=48, value=4)
    deadline_hours = st.number_input("Deadline (hours from now)", min_value=1, max_value=72, value=6)

with col3:
    budget = st.number_input("Budget ($)", min_value=0.1, value=5.0, step=0.5)
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Balanced", "Aggressive"])
    hour = st.slider("Hour of Day (24h)", 0, 23, 14)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

st.divider()

# ---------------- RUN BUTTON ----------------
if st.button("Get Recommendation", type="primary"):
    with st.spinner("Running Expectiminimax search..."):
        recommendation = get_recommendation(
            instance_type=instance_type,
            availability_zone=availability_zone,
            operating_system=operating_system,
            hour=hour,
            day_of_week=day_of_week,
            runtime_hours=runtime_hours,
            budget=budget,
            risk_profile=risk_profile
        )

        explanation_text, facts = get_explanation(
            recommendation=recommendation,
            budget=budget,
            runtime_hours=runtime_hours,
            deadline_hours=deadline_hours,
            risk_profile=risk_profile
        )

    st.success("Recommendation ready!")

    # ---------------- OUTPUT DISPLAY ----------------
    st.subheader("Recommendation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended Bid", f"${recommendation['recommended_bid']}/hr")
    m2.metric("Expected Cost", f"${recommendation['expected_cost']}")
    m3.metric("Completion Probability", f"{recommendation['completion_probability']}%")
    m4.metric("Risk Level", recommendation['risk_level'])

    m5, m6, m7 = st.columns(3)
    m5.metric("Interruption Probability", f"{recommendation['interruption_probability']}%")
    m6.metric("Estimated Savings", f"${recommendation['estimated_savings']} ({recommendation['estimated_savings_pct']}%)")
    m7.metric("Predicted Market Price", f"${recommendation['predicted_price']}/hr")

    st.subheader("Explainable Reasoning")
    st.info(explanation_text)

st.divider()
st.caption("SpotWise AI — built with Expectiminimax search, ML price prediction, and rule-based explanation.")