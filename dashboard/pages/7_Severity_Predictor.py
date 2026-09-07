import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import ApiError, get, post

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
from features import CATEGORICAL_VOCAB  # noqa: E402

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

st.set_page_config(page_title="Severity Predictor", page_icon="\U0001F52E", layout="centered")
st.title("Severity Predictor")
st.caption("Predicts accident severity for a hypothetical set of conditions, "
           "using the MLflow Production model served by the backend.")

try:
    health = get("/model/health")
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if not health["model_loaded"]:
    st.warning("No Production model is loaded on the backend yet. Run `python ml/train.py` first.")
    st.stop()

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        hour = st.slider("Hour of day", 0, 23, 17)
        day_of_week = st.selectbox(
            "Day of week", options=list(range(1, 8)),
            format_func=lambda d: ["Sunday", "Monday", "Tuesday", "Wednesday",
                                    "Thursday", "Friday", "Saturday"][d - 1],
            index=5,
        )
        month = st.slider("Month", 1, 12, 6)
        speed_limit = st.select_slider("Speed limit (mph)", options=[20, 30, 40, 50, 60, 70], value=30)
        number_of_vehicles = st.slider("Number of vehicles", 1, 6, 2)
        is_weekend = st.checkbox("Weekend", value=False)
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

    submitted = st.form_submit_button("Predict severity")

if submitted:
    payload = {
        "hour": hour, "day_of_week": day_of_week, "month": month,
        "road_type": road_type, "speed_limit": speed_limit, "junction_detail": junction_detail,
        "light_conditions": light_conditions, "weather_conditions": weather_conditions,
        "road_surface_conditions": road_surface_conditions, "urban_or_rural_area": urban_or_rural_area,
        "number_of_vehicles": number_of_vehicles, "temperature_c": temperature_c,
        "precipitation_mm": precipitation_mm, "is_weekend": is_weekend,
    }
    resp = post("/predict", json=payload)
    if resp.status_code != 200:
        st.error(f"Prediction failed ({resp.status_code}): {resp.text}")
    else:
        result = resp.json()
        label = result["predicted_severity_label"]
        color = {"Fatal": "red", "Serious": "orange", "Slight": "blue"}.get(label, "gray")
        st.markdown(f"### Predicted severity: :{color}[{label}]")
        st.caption(f"Model version {result['model_version']}")

        proba_df = pd.DataFrame({
            "severity_id": list(result["probabilities"].keys()),
            "probability": list(result["probabilities"].values()),
        })
        label_map = {"1": "Fatal", "2": "Serious", "3": "Slight", "-1": "Data missing"}
        proba_df["severity"] = proba_df["severity_id"].map(label_map).fillna(proba_df["severity_id"])
        fig = px.bar(proba_df, x="severity", y="probability", color="severity",
                     color_discrete_map={"Fatal": "red", "Serious": "orange", "Slight": "blue"})
        st.plotly_chart(fig, use_container_width=True)
