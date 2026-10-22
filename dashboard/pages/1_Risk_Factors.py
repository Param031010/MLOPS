import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, sidebar_filters

st.set_page_config(page_title="Risk Factors", page_icon="\U0001F327", layout="wide")
st.title("Risk Factors")
st.caption("What conditions, roads and people are associated with more severe accidents.")

try:
    params = sidebar_filters()
    weather = pd.DataFrame(get("/accidents/weather-impact", params=params))
    roads = pd.DataFrame(get("/accidents/road-type-severity", params=params))
    demographics = pd.DataFrame(get("/casualties/demographics", params=params))
    vehicles = pd.DataFrame(get("/vehicles/by-type", params=params))
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if weather.empty and roads.empty:
    st.info("No data for these filters -- widen them in the sidebar.")
    st.stop()

weather_tab, road_tab, people_tab = st.tabs(["Weather", "Road type", "People & vehicles"])

with weather_tab:
    share = st.toggle("Show as % within weather condition", key="weather_pct")
    df = weather.copy()
    y = "accident_count"
    if share:
        df["pct"] = df["accident_count"] / df.groupby("weather_conditions")["accident_count"].transform("sum") * 100
        y = "pct"
    fig = px.bar(df, x="weather_conditions", y=y, color="severity_label", barmode="stack",
                 labels={"weather_conditions": "", "accident_count": "Accidents", "pct": "% of accidents"})
    if not share:
        fig.update_xaxes(categoryorder="total descending")
    st.plotly_chart(fig, use_container_width=True)

with road_tab:
    share = st.toggle("Show as % within road type", value=True, key="road_pct")
    df = roads.copy()
    y = "accident_count"
    if share:
        df["pct"] = df["accident_count"] / df.groupby("road_type")["accident_count"].transform("sum") * 100
        y = "pct"
    fig = px.bar(df, x="road_type", y=y, color="severity_label", barmode="stack",
                 labels={"road_type": "", "accident_count": "Accidents", "pct": "% of accidents"})
    st.plotly_chart(fig, use_container_width=True)

with people_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Casualties by age and sex")
        fig = px.density_heatmap(demographics, x="age_band_of_casualty", y="sex_of_casualty",
                                 z="casualty_count", histfunc="sum", color_continuous_scale="Reds",
                                 labels={"age_band_of_casualty": "Age band", "sex_of_casualty": ""})
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Vehicles involved")
        fig = px.bar(vehicles, x="vehicle_count", y="vehicle_type", color="severity_label",
                     orientation="h", labels={"vehicle_count": "Vehicles", "vehicle_type": ""})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
