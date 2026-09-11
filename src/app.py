import streamlit as st
import pandas as pd
from clean_data import load_and_clean

st.set_page_config(page_title="SpotWise AI", layout="wide")
st.title("SpotWise AI – Intelligent Cloud Spot Instance Bidding Advisor")
st.write("Setup check: app is running ✅")

st.subheader("Dataset Preview (us-east-1)")

column_names = ["Timestamp", "InstanceType", "OperatingSystem", "AvailabilityZone", "SpotPrice"]

df = load_and_clean()
st.write("Shape:", df.shape)
st.write("Columns:", df.columns.tolist())
st.dataframe(df.head(20))