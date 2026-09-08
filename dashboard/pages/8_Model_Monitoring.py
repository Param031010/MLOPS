import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get

st.set_page_config(page_title="Model Monitoring", page_icon="\U0001F4C9", layout="wide")
st.title("Model Monitoring")
st.caption("Drift, rolling performance and serving telemetry for the Production model, "
           "measured against the baseline written by `ml/train.py`.")

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

# ---------------------------------------------------------------- performance
st.divider()
st.subheader("Rolling performance")
st.caption("The most recently loaded accidents scored against the Production model.")
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

    baseline_f1 = info.get("metrics", {}).get("macro_f1")
    if perf["macro_f1"] is not None and baseline_f1 is not None:
        drop = baseline_f1 - perf["macro_f1"]
        if drop > 0.05:
            st.error(f"Macro-F1 has dropped {drop:.3f} vs the training baseline "
                     "-- retrain per RETRAINING.md")
    if perf["false_negative_rate_fatal_serious"] > 0.15:
        st.error("False negative rate on Fatal/Serious exceeds the 15% retraining threshold.")

# ---------------------------------------------------------------------- drift
st.divider()
st.subheader("Drift vs training baseline")
st.caption("Population Stability Index. The retraining threshold is 0.2 on any single signal.")

try:
    drift = pd.DataFrame(get("/monitoring/drift"))
except ApiError as exc:
    st.error(str(exc))
    drift = pd.DataFrame()

if drift.empty:
    st.info("No drift data available yet.")
else:
    KINDS = [
        ("feature", "Feature drift", "Model inputs -- has the data the model consumes shifted?"),
        ("location", "Location drift", "Where accidents are happening, independent of the model's features."),
        ("class", "Class drift", "The severity mix itself. Values are each class's share of the data."),
    ]
    tabs = st.tabs([label for _, label, _ in KINDS])
    for tab, (kind, label, blurb) in zip(tabs, KINDS):
        with tab:
            subset = drift[drift["kind"] == kind]
            if subset.empty:
                st.info(f"No {label.lower()} reported. Retrain so the baseline includes it.")
                continue
            st.caption(blurb)
            fig = px.bar(
                subset, x="feature", y="psi", color="drifted",
                color_discrete_map={True: "#d62728", False: "#2ca02c"},
                labels={"psi": "PSI", "feature": ""},
            )
            fig.add_hline(y=0.2, line_dash="dash", annotation_text="retrain threshold (0.2)")
            st.plotly_chart(fig, use_container_width=True)

            shown = subset.rename(columns={
                "baseline_value": "baseline (share)" if kind == "class" else "baseline (mean)",
                "current_value": "current (share)" if kind == "class" else "current (mean)",
            }).drop(columns=["kind"])
            st.dataframe(shown, use_container_width=True, hide_index=True)

            if bool(subset["drifted"].any()):
                st.error(f"{label} has passed the PSI threshold -- see RETRAINING.md.")

# ------------------------------------------------------------------- latency
st.divider()
st.subheader("Serving telemetry")
st.caption("Inference latency and failures for `POST /predict`, sampled in-process since the "
           "backend last started.")
try:
    lat = get("/monitoring/latency")
except ApiError as exc:
    st.error(str(exc))
    lat = None

if lat:
    if lat["predictions_served"] == 0 and lat["failures"] == 0:
        st.info("No predictions served yet since the backend started. "
                "Use the Severity Predictor page, then reload this one.")
    else:
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Predictions served", f"{lat['predictions_served']:,}")
        l2.metric("p50 latency", f"{lat['p50_ms']:.1f} ms" if lat["p50_ms"] is not None else "-")
        l3.metric("p95 latency", f"{lat['p95_ms']:.1f} ms" if lat["p95_ms"] is not None else "-")
        l4.metric("Failures", f"{lat['failures']:,}", delta=f"{lat['failure_rate']:.1%} of requests",
                  delta_color="inverse")
        st.dataframe(pd.DataFrame([lat]), use_container_width=True, hide_index=True)
