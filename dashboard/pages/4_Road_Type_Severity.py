import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Road Type & Severity", page_icon="\U0001F6E3", layout="wide")
st.title("Road Type vs Severity")

try:
    rows = get("/accidents/road-type-severity")
except ApiError as exc:
    st.error(str(exc))
    st.stop()

df = pd.DataFrame(rows)
if df.empty:
    st.info("No data yet -- run the ETL pipeline.")
    st.stop()

severities = sorted(df["severity_label"].unique())
selected = st.sidebar.multiselect("Severity", severities, default=severities)
normalize = st.sidebar.checkbox("Show as % within road type", value=False)
df = df[df["severity_label"].isin(selected)]

if normalize:
    totals = df.groupby("road_type")["accident_count"].transform("sum")
    df = df.assign(pct=df["accident_count"] / totals * 100)
    fig = px.bar(df, x="road_type", y="pct", color="severity_label", barmode="stack",
                 labels={"road_type": "Road type", "pct": "% of accidents"})
else:
    fig = px.bar(df, x="road_type", y="accident_count", color="severity_label", barmode="stack",
                 labels={"road_type": "Road type", "accident_count": "Accidents"})
fig.update_xaxes(categoryorder="total descending")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df, use_container_width=True, hide_index=True)
