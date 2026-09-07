"""Standalone evaluation: scores the current Production model on the
chronological test split and logs recall, macro-F1, and a confusion matrix
image as MLflow artifacts under a fresh 'evaluation' run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from train import EXPERIMENT_NAME, MODEL_NAME, chronological_split, default_tracking_uri, load_training_data  # noqa: E402
from features import FEATURE_COLUMNS, build_features_df  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    client = mlflow.tracking.MlflowClient()
    mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]

    df = load_training_data()
    _, _, test_df = chronological_split(df)
    X_test = build_features_df(test_df)[FEATURE_COLUMNS]
    y_true = test_df["severity_id"].tolist()

    proba = np.asarray(model.predict(X_test))
    classes = sorted(set(y_true))
    y_pred = [classes[i] for i in np.argmax(proba, axis=1)]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    print(f"[evaluate] Production v{mv.version}: macro_f1={macro_f1:.4f} recall_macro={recall_macro:.4f}")
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{MODEL_NAME} v{mv.version} - test confusion matrix")
    fig.tight_layout()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    with mlflow.start_run(run_name=f"evaluate-v{mv.version}"):
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("recall_macro", recall_macro)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_artifact(str(cm_path))
        mlflow.set_tag("evaluated_model_version", mv.version)

    print(f"[evaluate] confusion matrix saved to {cm_path} and logged to MLflow")


if __name__ == "__main__":
    main()
