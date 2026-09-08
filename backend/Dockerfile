# Inference service image: the FastAPI app plus everything it imports at
# runtime. Built from the repository root so it can copy ml/ and
# etl/stats19_lookups.py alongside backend/app:
#
#   docker build -f backend/Dockerfile -t traffic-severity-api .
#
# Postgres and Streamlit are not containerized -- this image is only the
# inference/analytics API.
FROM python:3.13-slim

WORKDIR /srv

# libgomp1: the OpenMP runtime that LightGBM's and XGBoost's compiled
# extensions need at import time -- absent from the slim base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
# ml/ supplies features.py, which the predict router imports so that serving
# builds features exactly the way training did. etl/stats19_lookups.py is in
# turn imported by ml/features.py for its categorical vocabularies.
COPY ml ./ml
COPY etl/stats19_lookups.py ./etl/stats19_lookups.py

ENV PYTHONUNBUFFERED=1
EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
