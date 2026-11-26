"""Fixtures: load the data/ and models/ artifacts produced by stages 1-3 and hand-compute
scores with common/scores.py for comparison with reference calculations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from common import scores
from common.utils import DATA_DIR, MODEL_DIR, split_xy


def pytest_runtest_logreport(report):
    """Add a blank line after each test passes for better readability."""
    if report.when == "call" and report.outcome == "passed":
        print()


def _require(path):
    if not path.exists():
        pytest.skip(f"{path.name} not found - run stages 1-3 first "
                    "(python run_all.py or docker compose up)")


@pytest.fixture(scope="session")
def binary():
    data, model_path = DATA_DIR / "classification.csv", MODEL_DIR / "classification.joblib"
    _require(data)
    _require(model_path)
    _, X_te, _, y_te = split_xy(pd.read_csv(data), "label", stratify=True)
    model = joblib.load(model_path)
    proba = model.predict_proba(X_te)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    counts = scores.confusion_counts(y_te, y_pred)
    return {"y": y_te, "proba": proba, "pred": y_pred, "counts": counts,
            "scores": scores.binary_scores(*counts)}


@pytest.fixture(scope="session")
def multiclass():
    data = DATA_DIR / "classification_multiclass.csv"
    model_path = MODEL_DIR / "classification_multiclass.joblib"
    _require(data)
    _require(model_path)
    _, X_te, _, y_te = split_xy(pd.read_csv(data), "label", stratify=True)
    model = joblib.load(model_path)
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    tp, fp, fn, tn = scores.one_vs_rest_counts(cm)
    support = cm.sum(axis=1)
    return {"y": y_te, "pred": y_pred, "cm": cm, "counts": (tp, fp, fn, tn),
            "support": support, "per_class": scores.per_class_scores(tp, fp, fn),
            "averages": scores.average_scores(tp, fp, fn, support)}


@pytest.fixture(scope="session")
def regression():
    data, model_path = DATA_DIR / "regression.csv", MODEL_DIR / "regression.joblib"
    _require(data)
    _require(model_path)
    _, X_te, _, y_te = split_xy(pd.read_csv(data), "target")
    y_hat = joblib.load(model_path).predict(X_te)
    return {"y": y_te, "pred": y_hat, "p": X_te.shape[1],
            "scores": scores.regression_scores(y_te, y_hat, X_te.shape[1])}


@pytest.fixture(scope="session")
def ranking():
    data = DATA_DIR / "ranking.csv"
    _require(data)
    df = pd.read_csv(data)
    queries = [(g["relevance"].to_numpy(), g["score"].to_numpy())
               for _, g in df.groupby("query_id")]
    hand = pd.DataFrame([scores.query_scores(r, s) for r, s in queries]).mean().to_dict()
    return {"queries": queries, "hand": hand}
