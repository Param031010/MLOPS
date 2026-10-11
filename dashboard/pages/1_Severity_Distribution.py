import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, sidebar_filters

st.set_page_config(page_title="Severity Distribution", page_icon="\U0001F4CA", layout="wide")
st.title("Severity Distribution")

try:
    params = sidebar_filters()
    # monthly-trend carries the year dimension, so totals respect the year filter
    trend = pd.DataFrame(get("/accidents/monthly-trend", params=params))
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if trend.empty:
    st.info("No data for these filters -- widen them or run the ETL pipeline.")
    st.stop()

df = (trend.groupby(["severity_id", "severity_label"], as_index=False)
      [["accident_count", "total_casualties"]].sum()
      .sort_values("severity_id"))

chart_type = st.sidebar.radio("Chart type", ["Bar", "Donut"])
color_map = {"Fatal": "#d62728", "Serious": "#ff7f0e", "Slight": "#1f77b4", "Data missing": "#7f7f7f"}

st.metric("Accidents in selection", f"{int(df['accident_count'].sum()):,}")
if chart_type == "Bar":
    fig = px.bar(df, x="severity_label", y="accident_count", color="severity_label",
                 color_discrete_map=color_map, text="accident_count",
                 labels={"severity_label": "Severity", "accident_count": "Accidents"})
else:
    fig = px.pie(df, names="severity_label", values="accident_count", hole=0.5,
                 color="severity_label", color_discrete_map=color_map)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(df, use_container_width=True, hide_index=True)
