import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, sidebar_filters

st.set_page_config(page_title="Weather Impact", page_icon="\U0001F327", layout="wide")
st.title("Weather Impact on Accident Severity")

try:
    params = sidebar_filters()
    rows = get("/accidents/weather-impact", params=params)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

df = pd.DataFrame(rows)
if df.empty:
    st.info("No data for these filters -- widen them or run the ETL pipeline.")
    st.stop()

fig = px.bar(df, x="weather_conditions", y="accident_count", color="severity_label",
             barmode="stack", labels={"weather_conditions": "Weather condition", "accident_count": "Accidents"})
fig.update_xaxes(categoryorder="total descending")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Average temperature / precipitation by weather condition")
agg = df.groupby("weather_conditions")[["avg_temperature_c", "avg_precipitation_mm"]].mean().reset_index()
fig2 = px.bar(agg, x="weather_conditions", y=["avg_temperature_c", "avg_precipitation_mm"], barmode="group")
st.plotly_chart(fig2, use_container_width=True)
