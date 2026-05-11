from pathlib import Path

import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
driver_df = pd.read_csv(BASE_DIR / "data" / "driver_profile_dashboard.csv")
trip_df = pd.read_csv(BASE_DIR / "data" / "telematics_trip_summary.csv")

st.set_page_config(page_title="Driver Profile Dashboard", layout="wide")
st.title("Week 4 Driver Profile Dashboard")
st.caption("Built from the driving behavior motion dataset using pseudo trips and pseudo drivers.")

col1, col2, col3 = st.columns(3)
col1.metric("Drivers", int(driver_df["driver_id"].nunique()))
col2.metric("Trips", int(trip_df["trip_id"].nunique()))
col3.metric("Average Rating Proxy", round(driver_df["avg_driver_rating"].mean(), 2))

st.subheader("Driver Risk Overview")
st.dataframe(driver_df, use_container_width=True)

st.subheader("Top Drivers by Violation Score")
st.bar_chart(driver_df.set_index("driver_id")["avg_violation_score"])

st.subheader("Trips by Driver")
trip_mix = driver_df.set_index("driver_id")[["aggressive_trips", "normal_trips", "slow_trips"]]
st.bar_chart(trip_mix)

st.subheader("Ratings vs Violation Score")
scatter_df = driver_df[["driver_id", "avg_driver_rating", "avg_violation_score"]].copy()
st.dataframe(scatter_df, use_container_width=True)
