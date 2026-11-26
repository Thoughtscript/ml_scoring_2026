"""Stage 2: train the binary and multi-class logistic regressions into models/.

Scoring demonstrations and comparisons live in tests/ (stage 4).
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from common.utils import MODEL_DIR, load_data, log, split_xy, stage


@stage(2, "Train classifiers")
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("classification", "classification_multiclass"):
        X_tr, X_te, y_tr, y_te = split_xy(load_data(name), "label", stratify=True)
        model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        cm = confusion_matrix(y_te, model.predict(X_te))
        log.info("%s confusion matrix:\n%s", name, cm)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            log.info("TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)


if __name__ == "__main__":
    main()
