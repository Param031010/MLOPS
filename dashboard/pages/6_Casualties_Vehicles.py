import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Casualties & Vehicles", page_icon="\U0001F695", layout="wide")
st.title("Casualties & Vehicles")

try:
    by_type = pd.DataFrame(get("/casualties/by-type"))
    demographics = pd.DataFrame(get("/casualties/demographics"))
    vehicles = pd.DataFrame(get("/vehicles/by-type"))
    yearly = pd.DataFrame(get("/accidents/casualties"))
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if by_type.empty:
    st.info("No data yet -- run the ETL pipeline.")
    st.stop()

severities = sorted(by_type["severity_label"].unique())
selected = st.sidebar.multiselect("Severity", severities, default=severities)

tab1, tab2, tab3, tab4 = st.tabs(["Casualty type", "Demographics", "Vehicle type", "Yearly totals"])

with tab1:
    d = by_type[by_type["severity_label"].isin(selected)]
    fig = px.bar(d, x="casualty_type", y="casualty_count", color="severity_label", barmode="stack")
    fig.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    d = demographics[demographics["severity_label"].isin(selected)]
    fig = px.density_heatmap(d, x="age_band_of_casualty", y="sex_of_casualty", z="casualty_count",
                              color_continuous_scale="Reds", histfunc="sum")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    d = vehicles[vehicles["severity_label"].isin(selected)]
    fig = px.bar(d, x="vehicle_type", y="vehicle_count", color="severity_label", barmode="stack")
    fig.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.bar(yearly, x="year", y=["total_accidents", "total_casualties", "total_vehicles"],
                 barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(yearly, use_container_width=True, hide_index=True)
