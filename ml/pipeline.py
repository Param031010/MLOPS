"""The single inference artifact: feature engineering + model in one
mlflow.pyfunc model.

The registered model accepts *raw* accident records (the /predict request
fields, or columns straight out of the warehouse) and does its own feature
encoding via features.build_features_df, so a caller can never feed it
differently-preprocessed data than it was trained on. Whichever library won
training (RandomForest / XGBoost / LightGBM) sits behind the same contract.
"""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd

from features import CATEGORICAL_VOCAB, FEATURE_COLUMNS, build_features_df


class SeverityPipeline(mlflow.pyfunc.PythonModel):
    def __init__(self, model, classes: list[int]):
        self.model = model
        self.classes = list(classes)
        # Snapshot of the vocabulary the model was fitted against, so a
        # lookup-table change after training is detectable at load time.
        self.vocab = {k: list(v) for k, v in CATEGORICAL_VOCAB.items()}
        self.feature_columns = list(FEATURE_COLUMNS)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        features = build_features_df(model_input)
        return self.model.predict_proba(features[self.feature_columns])
