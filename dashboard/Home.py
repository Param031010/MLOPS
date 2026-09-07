import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Traffic Accident Severity Analytics", page_icon="\U0001F6A6", layout="wide")

st.title("Traffic Accident Severity Analytics")
st.caption("UK road accidents, weather, and road context -- from a Postgres/PostGIS warehouse via FastAPI")

try:
    overview = get("/accidents/overview")
except ApiError as exc:
    st.error(f"Could not reach the API: {exc}")
    st.stop()

if overview["total_accidents"] == 0:
    st.warning("No data loaded yet. Run the ETL pipeline (`extract.py` -> `transform.py` -> `load.py`), "
               "or trigger it via `POST /admin/run-etl`.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total accidents", f"{overview['total_accidents']:,}")
col2.metric("Total casualties", f"{overview['total_casualties']:,}")
col3.metric("Total vehicles involved", f"{overview['total_vehicles']:,}")
col4.metric("Years covered", ", ".join(str(y) for y in overview["years_covered"]))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Fatal", f"{overview['fatal_count']:,}")
col6.metric("Serious", f"{overview['serious_count']:,}")
col7.metric("Slight", f"{overview['slight_count']:,}")

st.divider()
st.subheader("Enrichment coverage")
cov1, cov2 = st.columns(2)
cov1.metric("Road context (TomTom) coverage", f"{overview['road_context_coverage_pct']:.1f}%")
cov1.progress(min(overview["road_context_coverage_pct"], 100) / 100)
cov2.metric("Weather (Open-Meteo) coverage", f"{overview['weather_coverage_pct']:.1f}%")
cov2.progress(min(overview["weather_coverage_pct"], 100) / 100)
st.caption("TomTom's free tier can't cover every coordinate -- this is the real, measured coverage, not hidden.")

st.divider()
st.subheader("Explore")

nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    st.page_link("pages/1_Severity_Distribution.py", label="Severity Distribution", icon="\U0001F4CA")
    st.page_link("pages/2_Time_Patterns.py", label="Time Patterns", icon="\U0001F550")
with nav2:
    st.page_link("pages/3_Weather_Impact.py", label="Weather Impact", icon="\U0001F327️")
    st.page_link("pages/4_Road_Type_Severity.py", label="Road Type & Severity", icon="\U0001F6E3️")
with nav3:
    st.page_link("pages/5_Hotspot_Map.py", label="Hotspot Map", icon="\U0001F5FA️")
    st.page_link("pages/6_Casualties_Vehicles.py", label="Casualties & Vehicles", icon="\U0001F695")
with nav4:
    st.page_link("pages/7_Severity_Predictor.py", label="Severity Predictor", icon="\U0001F52E")
    st.page_link("pages/8_Model_Monitoring.py", label="Model Monitoring", icon="\U0001F4C9")
