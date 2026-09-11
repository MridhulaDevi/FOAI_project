import streamlit as st
import pandas as pd

st.set_page_config(page_title="SpotWise AI", layout="wide")
st.title("SpotWise AI – Intelligent Cloud Spot Instance Bidding Advisor")
st.write("Setup check: app is running ✅")

st.subheader("Dataset Preview (us-east-1)")

column_names = ["Timestamp", "InstanceType", "OperatingSystem", "AvailabilityZone", "SpotPrice"]
df = pd.read_csv("data/us-east-1.csv", header=None, names=column_names)

st.write("Shape:", df.shape)
st.write("Columns:", df.columns.tolist())
st.dataframe(df.head(20))