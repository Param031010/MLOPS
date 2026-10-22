import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, sidebar_filters

st.set_page_config(page_title="Traffic Accident Severity Analytics", page_icon="\U0001F6A6", layout="wide")

st.title("Traffic Accident Severity Analytics")
st.caption("UK road accidents (STATS19) enriched with weather and road context -- Postgres/PostGIS warehouse via FastAPI")

SEVERITY_COLORS = {"Fatal": "#d62728", "Serious": "#ff7f0e", "Slight": "#1f77b4", "Data missing": "#7f7f7f"}
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

try:
    overview = get("/accidents/overview")
    if overview["total_accidents"] == 0:
        st.warning("No data loaded yet. Run the ETL pipeline (`extract.py` -> `transform.py` -> `load.py`), "
                   "or trigger it via `POST /admin/run-etl`.")
        st.stop()
    params = sidebar_filters()
    trend = pd.DataFrame(get("/accidents/monthly-trend", params=params))
    by_hour = pd.DataFrame(get("/accidents/by-hour", params=params))
except ApiError as exc:
    st.error(f"Could not reach the API: {exc}")
    st.stop()

if trend.empty:
    st.info("No data for these filters -- widen them in the sidebar.")
    st.stop()

by_severity = (trend.groupby(["severity_id", "severity_label"], as_index=False)
               [["accident_count", "total_casualties"]].sum().sort_values("severity_id"))


def count_for(label: str) -> int:
    row = by_severity.loc[by_severity["severity_label"] == label, "accident_count"]
    return int(row.iloc[0]) if len(row) else 0


k1, k2, k3, k4 = st.columns(4)
k1.metric("Accidents", f"{int(by_severity['accident_count'].sum()):,}")
k2.metric("Casualties", f"{int(by_severity['total_casualties'].sum()):,}")
k3.metric("Fatal", f"{count_for('Fatal'):,}")
k4.metric("Serious", f"{count_for('Serious'):,}")

left, right = st.columns([1, 2])
with left:
    st.subheader("Severity mix")
    fig = px.pie(by_severity, names="severity_label", values="accident_count", hole=0.55,
                 color="severity_label", color_discrete_map=SEVERITY_COLORS)
    fig.update_layout(margin=dict(t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("When accidents happen")
    pivot = (by_hour.groupby(["day_name", "hour"])["accident_count"].sum().reset_index()
             .pivot(index="day_name", columns="hour", values="accident_count")
             .reindex(DAY_NAMES).fillna(0))
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                    labels={"x": "Hour of day", "y": "", "color": "Accidents"})
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly trend")
monthly = trend.sort_values(["year", "month"]).copy()
monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
fig = px.line(monthly, x="period", y="accident_count", color="severity_label",
              color_discrete_map=SEVERITY_COLORS, labels={"period": "Month", "accident_count": "Accidents"})
st.plotly_chart(fig, use_container_width=True)

with st.expander("Data coverage"):
    c1, c2 = st.columns(2)
    c1.metric("Weather (Open-Meteo)", f"{overview['weather_coverage_pct']:.1f}%")
    c2.metric("Road context (TomTom)", f"{overview['road_context_coverage_pct']:.2f}%")
    st.caption("TomTom's free tier can't cover every coordinate -- this is the measured coverage, not hidden.")

st.divider()
n1, n2, n3 = st.columns(3)
n1.page_link("pages/1_Risk_Factors.py", label="Risk factors: weather, roads, people", icon="\U0001F327")
n2.page_link("pages/2_Hotspot_Map.py", label="Hotspot map", icon="\U0001F5FA")
n3.page_link("pages/3_Severity_Model.py", label="Severity predictor & monitoring", icon="\U0001F52E")
