from datetime import date, timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Hotspot Map", page_icon="\U0001F5FA", layout="wide")
st.title("Accident Hotspots")

try:
    overview = get("/accidents/overview")
except ApiError as exc:
    st.error(str(exc))
    st.stop()

years = overview.get("years_covered") or []
default_from = date(min(years), 1, 1) if years else date.today() - timedelta(days=365)
default_to = date(max(years), 12, 31) if years else date.today()

st.sidebar.subheader("Filters")
severity = st.sidebar.selectbox("Severity", ["All", "Fatal", "Serious", "Slight"])
date_from = st.sidebar.date_input("From", default_from)
date_to = st.sidebar.date_input("To", default_to)

severity_map = {"Fatal": 1, "Serious": 2, "Slight": 3}
params = {"date_from": str(date_from), "date_to": str(date_to), "limit": 8000}
if severity != "All":
    params["severity"] = severity_map[severity]

try:
    rows = get("/accidents/hotspots", params=params)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

df = pd.DataFrame(rows)
st.caption(f"{len(df):,} accidents matching the current filters (server-side query, capped at 8,000 points)")

if df.empty:
    st.info("No accidents match these filters.")
    st.stop()

color_map = {"Fatal": [214, 39, 40], "Serious": [255, 127, 14], "Slight": [31, 119, 180]}
df["color"] = df["severity_label"].map(color_map)

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position="[longitude, latitude]",
    get_fill_color="color",
    get_radius=300,
    radius_min_pixels=3,
    radius_max_pixels=15,
    pickable=True,
    opacity=0.7,
)
view_state = pdk.ViewState(
    latitude=float(df["latitude"].mean()), longitude=float(df["longitude"].mean()),
    zoom=6, pitch=0,
)
st.pydeck_chart(pdk.Deck(
    layers=[layer], initial_view_state=view_state,
    tooltip={"text": "{severity_label} - {full_date}\n{street_name}"},
    map_style=None,
))

with st.expander("Top 20 locations by accident count"):
    try:
        top_rows = get("/accidents/top-locations", params={"limit": 20})
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)
    except ApiError as exc:
        st.error(str(exc))
