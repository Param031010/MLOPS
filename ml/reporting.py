"""Evaluation artifacts shared by train.py (every candidate run) and
evaluate.py (the registered Production model)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, recall_score)

SEVERE = (1, 2)  # Fatal, Serious


def false_negative_rate_severe(y_true, y_pred) -> float:
    """Share of actual Fatal/Serious accidents predicted as non-severe -- the
    model's costliest mistake."""
    severe = [(t, p) for t, p in zip(y_true, y_pred) if t in SEVERE]
    if not severe:
        return 0.0
    return sum(1 for _, p in severe if p not in SEVERE) / len(severe)


def log_split_report(prefix: str, y_true, y_pred, classes, title: str, out_dir: Path) -> dict:
    """Log recall (macro + per class), macro-F1, severe false-negative rate,
    a confusion-matrix PNG + CSV and the classification report to the active
    MLflow run. Returns the scalar metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_fnr_fatal_serious": false_negative_rate_severe(y_true, y_pred),
    }
    per_class = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    for c, r in zip(classes, per_class):
        key = str(c).replace("-", "neg")
        metrics[f"{prefix}_recall_class_{key}"] = float(r)
    mlflow.log_metrics(metrics)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} - {prefix} confusion matrix")
    fig.tight_layout()
    png = out_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(png)
    plt.close(fig)
    csv = out_dir / f"{prefix}_confusion_matrix.csv"
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                 columns=[f"pred_{c}" for c in classes]).to_csv(csv)
    mlflow.log_artifact(str(png), artifact_path="evaluation")
    mlflow.log_artifact(str(csv), artifact_path="evaluation")
    mlflow.log_text(classification_report(y_true, y_pred, labels=classes, zero_division=0),
                    f"evaluation/{prefix}_classification_report.txt")
    return metrics
