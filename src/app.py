import streamlit as st
import sys
import os
import plotly.graph_objects as go
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from search import get_recommendation, compare_strategies
from explain import get_explanation
from clean_data import get_price_history

st.set_page_config(page_title="SpotWise AI", layout="wide")

st.title("SpotWise AI – Intelligent Cloud Spot Instance Bidding Advisor")
st.write("Get an AI-recommended bid for your cloud spot instance job.")

st.divider()

# ---------------- INPUT FORM ----------------
st.subheader("Job Details")

col1, col2, col3 = st.columns(3)

with col1:
    provider = st.selectbox("Cloud Provider", ["AWS"])
    instance_type = st.selectbox(
        "Instance Type",
        ["m4.xlarge", "c3.4xlarge", "r3.large"]
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

        strategies, predicted_price = compare_strategies(
            instance_type=instance_type,
            availability_zone=availability_zone,
            operating_system=operating_system,
            hour=hour,
            day_of_week=day_of_week,
            runtime_hours=runtime_hours,
            budget=budget,
            risk_profile=risk_profile
        )

        price_history = get_price_history(instance_type, availability_zone)

    st.success("Recommendation ready!")

    # ---------------- METRICS ----------------
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

    # ---------------- EXPLANATION ----------------
    st.subheader("Explainable Reasoning")
    st.info(explanation_text)

    # ---------------- PRICE TREND CHART ----------------
    st.subheader("Historical Price Trend")

    if len(price_history) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_history["Timestamp"],
            y=price_history["SpotPrice"],
            mode="lines",
            name="Historical Price",
            line=dict(color="steelblue")
        ))
        fig.add_hline(
            y=recommendation["recommended_bid"],
            line_dash="dash",
            line_color="green",
            annotation_text="Recommended Bid"
        )
        fig.add_hline(
            y=predicted_price,
            line_dash="dot",
            line_color="orange",
            annotation_text="Predicted Price"
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Spot Price ($/hr)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No historical data available for this instance type / zone combination.")

    # ---------------- STRATEGY COMPARISON TABLE ----------------
    st.subheader("Strategy Comparison")
    comparison_df = pd.DataFrame(strategies)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("SpotWise AI — built with Expectiminimax search, ML price prediction, and rule-based explanation.")