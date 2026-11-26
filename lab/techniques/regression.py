"""Stage 3: train the linear regression into models/.

Scoring demonstrations and comparisons live in tests/ (stage 4).
"""
import joblib
from sklearn.linear_model import LinearRegression

from common.utils import MODEL_DIR, load_data, log, split_xy, stage


@stage(3, "Train regressor")
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_tr, X_te, y_tr, y_te = split_xy(load_data("regression"), "target")
    model = LinearRegression().fit(X_tr, y_tr)
    joblib.dump(model, MODEL_DIR / "regression.joblib")
    log.info("regression: %d train rows, %d test rows, %d features",
             len(X_tr), len(X_te), X_tr.shape[1])


if __name__ == "__main__":
    main()
