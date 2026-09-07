import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Time Patterns", page_icon="\U0001F550", layout="wide")
st.title("Time Patterns")

try:
    hour_rows = get("/accidents/by-hour")
    monthly_rows = get("/accidents/monthly-trend")
except ApiError as exc:
    st.error(str(exc))
    st.stop()

hour_df = pd.DataFrame(hour_rows)
monthly_df = pd.DataFrame(monthly_rows)
if hour_df.empty:
    st.info("No data yet -- run the ETL pipeline.")
    st.stop()

severities = sorted(hour_df["severity_label"].unique())
selected = st.sidebar.multiselect("Severity", severities, default=severities)

hour_df = hour_df[hour_df["severity_label"].isin(selected)]
monthly_df = monthly_df[monthly_df["severity_label"].isin(selected)]

tab1, tab2 = st.tabs(["Hour x Weekday", "Monthly Trend"])

with tab1:
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    pivot = (
        hour_df.groupby(["day_name", "hour"])["accident_count"].sum()
        .reset_index()
        .pivot(index="day_name", columns="hour", values="accident_count")
        .reindex(day_names)
        .fillna(0)
    )
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                     labels={"x": "Hour of day", "y": "Day of week", "color": "Accidents"})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    monthly_df = monthly_df.sort_values(["year", "month"])
    monthly_df["period"] = monthly_df["year"].astype(str) + "-" + monthly_df["month"].astype(str).str.zfill(2)
    fig2 = px.line(monthly_df, x="period", y="accident_count", color="severity_label",
                    markers=True, labels={"period": "Month", "accident_count": "Accidents"})
    st.plotly_chart(fig2, use_container_width=True)
