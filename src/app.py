import streamlit as st
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from search import get_recommendation, compare_strategies
from explain import get_explanation
import base64

BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.svg")

def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_B64 = img_to_b64(LOGO_PATH)
LOGO_MIME = "image/svg+xml"   # use "image/png" if your logo is a PNG

st.set_page_config(
    page_title="SpotWise AI",
    page_icon=os.path.join(BASE_DIR, "assets", "logo.png"),
    layout="wide"
)
# ---------------- STYLING (UI only) ----------------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #312e81 55%, #4f46e5 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(49, 46, 129, 0.25);
    }
    .hero h1 {color: #ffffff; font-size: 2.2rem; margin: 0 0 .4rem 0; font-weight: 800; letter-spacing: -0.5px;}
    .hero p {color: #c7d2fe; font-size: 1.05rem; margin: 0;}
    .hero .badge {
        display: inline-block; background: rgba(255,255,255,0.15); color: #e0e7ff;
        padding: 3px 12px; border-radius: 999px; font-size: .78rem; margin-bottom: .8rem;
        letter-spacing: .5px; font-weight: 600;
    }

    /* Section headings */
    .section-title {font-size: 1.35rem; font-weight: 700; margin: 1.2rem 0 .2rem 0;}
    .section-sub {opacity: .65; font-size: .92rem; margin-bottom: .8rem;}
    .card-label {
        font-size: .78rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: #6366f1; margin-bottom: .6rem;
    }

    /* Bordered containers as cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] p {font-size: .85rem; opacity: .75; font-weight: 600;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 700;}

    /* Primary button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        border: none; color: white; font-weight: 700; font-size: 1.05rem;
        padding: .7rem 1rem; border-radius: 12px;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(79, 70, 229, 0.45);
    }

    /* Dataframe + alerts */
    div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    .hero-title {display: flex; align-items: center; gap: 16px; margin-bottom: .4rem;}
.hero-title img {height: 60px; width: 60px; border-radius: 14px;}
.hero-title h1 {margin: 0 !important;}
</style>
""", unsafe_allow_html=True)


def section(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


# ---------------- HERO ----------------
st.markdown(f"""
<div class="hero">
    <div class="badge">ALPHA-BETA PRUNING · CSP · FORWARD CHAINING</div>
    <div class="hero-title">
        <img src="data:{LOGO_MIME};base64,{LOGO_B64}">
        <h1>SpotWise AI</h1>
    </div>
    <p>Intelligent Cloud Spot Instance Bidding Advisor. Find the best spot bid with
    explainable, search-based reasoning.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------
section("Job Details", "Describe your workload and constraints, then let the search find the optimal bid.")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown('<div class="card-label"> Infrastructure</div>', unsafe_allow_html=True)
        provider = st.selectbox("Cloud Provider", ["AWS"])
        instance_type = st.selectbox("Instance Type", ["m4.xlarge", "c3.4xlarge", "r3.large"])
        availability_zone = st.selectbox(
            "Availability Zone",
            ["us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1e"]
        )

with col2:
    with st.container(border=True):
        st.markdown('<div class="card-label"> Workload & Timing</div>', unsafe_allow_html=True)
        operating_system = st.selectbox("Operating System", ["Linux/UNIX", "Windows", "SUSE Linux"])
        runtime_hours = st.number_input("Runtime (hours)", min_value=1, max_value=48, value=4)
        deadline_hours = st.number_input("Deadline (hours from now)", min_value=1, max_value=72, value=6)

with col3:
    with st.container(border=True):
        st.markdown('<div class="card-label"> Budget & Risk</div>', unsafe_allow_html=True)
        budget = st.number_input("Budget ($)", min_value=0.1, value=5.0, step=0.5)
        risk_profile = st.selectbox("Risk Profile", ["Conservative", "Balanced", "Aggressive"])
        hour = st.slider("Hour of Day (24h)", 0, 23, 14)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

st.write("")
# ---------------- RUN BUTTON ----------------
run_clicked = st.button(" Get Recommendation", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running alpha-beta search (first run loads the dataset, please wait)..."):
        recommendation = get_recommendation(
            instance_type=instance_type,
            availability_zone=availability_zone,
            operating_system=operating_system,
            hour=hour,
            day_of_week=day_of_week,
            runtime_hours=runtime_hours,
            budget=budget,
            risk_profile=risk_profile,
            deadline_hours=deadline_hours
        )

        if recommendation.get("infeasible"):
            st.error(recommendation["reason"])
            st.stop()

        explanation_text, facts = get_explanation(
            recommendation=recommendation,
            budget=budget,
            runtime_hours=runtime_hours,
            deadline_hours=deadline_hours,
            risk_profile=risk_profile
        )

        strategies, median_price = compare_strategies(
            instance_type=instance_type,
            availability_zone=availability_zone,
            operating_system=operating_system,
            hour=hour,
            day_of_week=day_of_week,
            runtime_hours=runtime_hours,
            budget=budget,
            risk_profile=risk_profile
        )

    st.success("Recommendation ready!")

    # ---------------- METRICS ----------------
    st.divider()
    section("Recommendation", "The optimal bid found by the search, with its cost and risk profile.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended Bid", f"${recommendation['recommended_bid']}/hr")
    m2.metric("Expected Cost", f"${recommendation['expected_cost']}")
    m3.metric("Completion Probability", f"{recommendation['completion_probability']}%")
    m4.metric("Risk Level", recommendation['risk_level'])

    st.write("")

    m5, m6, m7 = st.columns(3)
    m5.metric("Interruption Probability", f"{recommendation['interruption_probability']}%")
    m6.metric("Estimated Savings", f"${recommendation['estimated_savings']} ({recommendation['estimated_savings_pct']}%)")
    m7.metric("Typical (Median) Price", f"${recommendation['predicted_price']}/hr")

    st.write("")

    with st.container(border=True):
        st.markdown('<div class="card-label"> Search Statistics</div>', unsafe_allow_html=True)
        st.caption(
            f"Alpha-beta search: {recommendation['nodes_visited']} nodes visited "
            f"(plain minimax would visit {recommendation['nodes_without_pruning']}), "
            f"{recommendation['pruning_cutoffs']} pruning cutoffs."
        )
        st.caption(f"Max spend (bid ceiling): ${recommendation['max_cost']}")

    # ---------------- EXPLANATION ----------------
    st.divider()
    section("Explainable Reasoning", "Why this bid was chosen, based on the rules that fired.")

    st.info(explanation_text)
    st.caption("Forward-chaining rules fired: " + ", ".join(facts["rules_fired"]))

    # ---------------- STRATEGY COMPARISON ----------------
    st.divider()
    section("Strategy Comparison", "How the recommended bid stacks up against alternative bidding strategies.")

    st.dataframe(pd.DataFrame(strategies), use_container_width=True, hide_index=True)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<div style='text-align:center; opacity:.6; font-size:.85rem; padding:.5rem 0 1rem 0;'>"
    "☁️ <b>SpotWise AI</b> · alpha-beta pruning search, CSP constraints, and forward-chaining explanations."
    "</div>",
    unsafe_allow_html=True
)