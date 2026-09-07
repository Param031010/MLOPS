import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Model Monitoring", page_icon="\U0001F4C9", layout="wide")
st.title("Model Monitoring")

try:
    health = get("/model/health")
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if not health["model_loaded"]:
    st.warning("No Production model is loaded on the backend yet. Run `python ml/train.py` first.")
    st.stop()

info = get("/model/info")
c1, c2, c3 = st.columns(3)
c1.metric("Algorithm", info.get("algorithm", "-"))
c2.metric("Version", info.get("version", "-"))
c3.metric("Trained at", (info.get("trained_at") or "-")[:19].replace("T", " "))

st.subheader("Training metrics")
st.dataframe(pd.DataFrame([info.get("metrics", {})]), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Rolling performance (most recent accidents scored against Production model)")
try:
    perf = get("/monitoring/performance")
except ApiError as exc:
    st.error(str(exc))
    perf = None

if perf:
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Macro F1", f"{perf['macro_f1']:.3f}")
    p2.metric("Recall (macro)", f"{perf['recall_macro']:.3f}")
    p3.metric("False negative rate (Fatal/Serious)", f"{perf['false_negative_rate_fatal_serious']:.1%}")
    p4.metric("Sample size", f"{perf['sample_size']:,}")
    st.caption(f"Window: {perf['window_start']} to {perf['window_end']}")
    if perf["macro_f1"] is not None and info.get("metrics", {}).get("macro_f1") is not None:
        drop = info["metrics"]["macro_f1"] - perf["macro_f1"]
        if drop > 0.05:
            st.error(f"Macro-F1 has dropped {drop:.3f} vs the training baseline -- retrain per RETRAINING.md")
    if perf["false_negative_rate_fatal_serious"] > 0.15:
        st.error("False negative rate on Fatal/Serious exceeds the 15% retraining threshold.")

st.divider()
st.subheader("Feature drift vs training baseline")
try:
    drift = pd.DataFrame(get("/monitoring/drift"))
except ApiError as exc:
    st.error(str(exc))
    drift = pd.DataFrame()

if not drift.empty:
    fig = px.bar(drift, x="feature", y="psi", color="drifted",
                 color_discrete_map={True: "red", False: "green"},
                 labels={"psi": "Population Stability Index"})
    fig.add_hline(y=0.2, line_dash="dash", annotation_text="retrain threshold (0.2)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(drift, use_container_width=True, hide_index=True)
    if drift["drifted"].any():
        st.error("One or more features have drifted past the PSI threshold -- see RETRAINING.md.")
else:
    st.info("No drift data available yet.")
