import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, post

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from features import CATEGORICAL_VOCAB  # noqa: E402

st.set_page_config(page_title="Severity Model", page_icon="\U0001F52E", layout="wide")
st.title("Severity Model")

SEVERITY_COLORS = {"Fatal": "#d62728", "Serious": "#ff7f0e", "Slight": "#1f77b4", "Data missing": "#7f7f7f"}
SEVERITY_LABELS = {"1": "Fatal", "2": "Serious", "3": "Slight", "-1": "Data missing"}
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
DEFAULT_CATEGORY = {
    "road_type": "Single carriageway",
    "junction_detail": "Not at junction or within 20 metres",
    "light_conditions": "Daylight",
    "weather_conditions": "Fine, no high winds",
    "road_surface_conditions": "Dry",
    "urban_or_rural_area": "Urban",
}


def default_index(field: str) -> int:
    vocab = CATEGORICAL_VOCAB[field]
    default = DEFAULT_CATEGORY.get(field)
    return vocab.index(default) if default in vocab else 0


try:
    health = get("/model/health")
    info = get("/model/info") if health["model_loaded"] else {}
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if not health["model_loaded"]:
    st.warning("No Production model is loaded on the backend yet. Run `python ml/train.py` first.")
    st.stop()

metrics = info.get("metrics", {})
h1, h2, h3, h4 = st.columns(4)
h1.metric("Algorithm", info.get("algorithm", "-"))
h2.metric("Version", info.get("version", "-"))
h3.metric("Test macro-F1", f"{metrics['macro_f1']:.3f}" if "macro_f1" in metrics else "-")
h4.metric("Trained", (info.get("trained_at") or "-")[:10])

predict_tab, monitor_tab = st.tabs(["Predict", "Monitoring"])

# -------------------------------------------------------------------- predict
with predict_tab:
    st.caption("Predicts severity for a hypothetical set of conditions using the MLflow Production model.")
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour of day", 0, 23, 17)
            day_of_week = st.selectbox("Day of week", options=list(range(1, 8)),
                                       format_func=lambda d: DAYS[d - 1], index=5)
            month = st.slider("Month", 1, 12, 6)
            speed_limit = st.select_slider("Speed limit (mph)", options=[20, 30, 40, 50, 60, 70], value=30)
            number_of_vehicles = st.slider("Number of vehicles", 1, 6, 2)
        with c2:
            road_type = st.selectbox("Road type", CATEGORICAL_VOCAB["road_type"],
                                     index=default_index("road_type"))
            junction_detail = st.selectbox("Junction detail", CATEGORICAL_VOCAB["junction_detail"],
                                           index=default_index("junction_detail"))
            light_conditions = st.selectbox("Light conditions", CATEGORICAL_VOCAB["light_conditions"],
                                            index=default_index("light_conditions"))
            weather_conditions = st.selectbox("Weather conditions", CATEGORICAL_VOCAB["weather_conditions"],
                                              index=default_index("weather_conditions"))
            road_surface_conditions = st.selectbox("Road surface", CATEGORICAL_VOCAB["road_surface_conditions"],
                                                   index=default_index("road_surface_conditions"))
            urban_or_rural_area = st.selectbox("Area", CATEGORICAL_VOCAB["urban_or_rural_area"],
                                               index=default_index("urban_or_rural_area"))
        c3, c4 = st.columns(2)
        temperature_c = c3.number_input("Temperature (C)", -10.0, 40.0, 12.0)
        precipitation_mm = c4.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
        submitted = st.form_submit_button("Predict severity", type="primary")

    if submitted:
        payload = {
            "hour": hour, "day_of_week": day_of_week, "month": month,
            "road_type": road_type, "speed_limit": speed_limit, "junction_detail": junction_detail,
            "light_conditions": light_conditions, "weather_conditions": weather_conditions,
            "road_surface_conditions": road_surface_conditions, "urban_or_rural_area": urban_or_rural_area,
            "number_of_vehicles": number_of_vehicles, "temperature_c": temperature_c,
            "precipitation_mm": precipitation_mm, "is_weekend": day_of_week in (1, 7),
        }
        resp = post("/predict", json=payload)
        if resp.status_code != 200:
            st.error(f"Prediction failed ({resp.status_code}): {resp.text}")
        else:
            result = resp.json()
            label = result["predicted_severity_label"]
            color = {"Fatal": "red", "Serious": "orange", "Slight": "blue"}.get(label, "gray")
            st.markdown(f"### Predicted severity: :{color}[{label}]")
            proba = pd.DataFrame({"severity": [SEVERITY_LABELS.get(k, k) for k in result["probabilities"]],
                                  "probability": list(result["probabilities"].values())})
            fig = px.bar(proba, x="severity", y="probability", color="severity",
                         color_discrete_map=SEVERITY_COLORS, text_auto=".1%")
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------- monitoring
with monitor_tab:
    st.caption("Measured against the baseline written by `ml/train.py`; retraining thresholds are in RETRAINING.md.")

    try:
        perf = get("/monitoring/performance")
    except ApiError as exc:
        st.error(str(exc))
        perf = None

    if perf:
        st.subheader("Rolling performance")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Macro F1", f"{perf['macro_f1']:.3f}")
        p2.metric("Recall (macro)", f"{perf['recall_macro']:.3f}")
        p3.metric("Fatal/Serious missed", f"{perf['false_negative_rate_fatal_serious']:.1%}")
        p4.metric("Sample size", f"{perf['sample_size']:,}")
        baseline_f1 = metrics.get("macro_f1")
        if baseline_f1 is not None and perf["macro_f1"] is not None and baseline_f1 - perf["macro_f1"] > 0.05:
            st.error("Macro-F1 has dropped more than 5 points vs the training baseline -- retrain.")
        if perf["false_negative_rate_fatal_serious"] > 0.15:
            st.error("Fatal/Serious false-negative rate exceeds the 15% retraining threshold.")

    st.subheader("Drift vs training baseline")
    st.caption("Population Stability Index; above 0.2 on any signal means retrain.")
    try:
        drift = pd.DataFrame(get("/monitoring/drift"))
    except ApiError as exc:
        st.error(str(exc))
        drift = pd.DataFrame()

    if drift.empty:
        st.info("No drift data available yet.")
    else:
        fig = px.bar(drift, x="feature", y="psi", color="drifted", facet_col="kind",
                     color_discrete_map={True: "#d62728", False: "#2ca02c"},
                     labels={"psi": "PSI", "feature": ""})
        fig.update_xaxes(matches=None, showticklabels=True)
        fig.add_hline(y=0.2, line_dash="dash")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].title() + " drift"))
        st.plotly_chart(fig, use_container_width=True)
        if bool(drift["drifted"].any()):
            st.error("Drift past the PSI threshold on: " + ", ".join(drift.loc[drift["drifted"], "feature"]))

    with st.expander("Serving telemetry and input-data quality"):
        try:
            lat = get("/monitoring/latency")
            dq = get("/monitoring/data-quality")
        except ApiError as exc:
            st.error(str(exc))
            lat = dq = None
        if lat:
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Predictions served", f"{lat['predictions_served']:,}")
            l2.metric("p50 latency", f"{lat['p50_ms']:.1f} ms" if lat["p50_ms"] is not None else "-")
            l3.metric("p95 latency", f"{lat['p95_ms']:.1f} ms" if lat["p95_ms"] is not None else "-")
            l4.metric("Failures", f"{lat['failures']:,}")
        if dq:
            dq_df = pd.DataFrame(dq["fields"])
            fig = px.bar(dq_df.melt(id_vars=["field"], value_vars=["missing_rate", "invalid_rate"],
                                    var_name="check", value_name="rate"),
                         x="field", y="rate", color="check", barmode="group",
                         labels={"rate": "share of rows", "field": ""})
            st.plotly_chart(fig, use_container_width=True)
            if dq_df["flagged"].any():
                st.error("Input quality degraded on: " + ", ".join(dq_df.loc[dq_df["flagged"], "field"]))
