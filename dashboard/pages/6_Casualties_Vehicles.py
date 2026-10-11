import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, sidebar_filters

st.set_page_config(page_title="Casualties & Vehicles", page_icon="\U0001F695", layout="wide")
st.title("Casualties & Vehicles")

try:
    params = sidebar_filters()
    by_type = pd.DataFrame(get("/casualties/by-type", params=params))
    demographics = pd.DataFrame(get("/casualties/demographics", params=params))
    vehicles = pd.DataFrame(get("/vehicles/by-type", params=params))
    yearly = pd.DataFrame(get("/accidents/casualties"))
    if "year_from" in params:
        yearly = yearly[yearly["year"].between(params["year_from"], params["year_to"])]
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if by_type.empty:
    st.info("No data for these filters -- widen them or run the ETL pipeline.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Casualty type", "Demographics", "Vehicle type", "Yearly totals"])

with tab1:
    fig = px.bar(by_type, x="casualty_type", y="casualty_count", color="severity_label", barmode="stack")
    fig.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.density_heatmap(demographics, x="age_band_of_casualty", y="sex_of_casualty",
                              z="casualty_count", color_continuous_scale="Reds", histfunc="sum")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.bar(vehicles, x="vehicle_type", y="vehicle_count", color="severity_label", barmode="stack")
    fig.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.bar(yearly, x="year", y=["total_accidents", "total_casualties", "total_vehicles"],
                 barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(yearly, use_container_width=True, hide_index=True)
