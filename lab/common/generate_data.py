"""Stage 1: generate randomized CSV datasets (classification, regression, ranking)."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from common.utils import DATA_DIR, SEED, log, stage


@stage(1, "Generate data")
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Binary classification, mildly imbalanced (70/30) so accuracy and F1 tell different stories.
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=8, n_redundant=2,
                               weights=[0.7, 0.3], class_sep=1.0, flip_y=0.01, random_state=SEED)
    clf = pd.DataFrame(np.round(X, 6), columns=[f"f{i}" for i in range(X.shape[1])])
    clf["label"] = y
    clf.to_csv(DATA_DIR / "classification.csv", index=False)
    log.info("classification.csv: %d rows x %d features | class balance=%s",
             *X.shape, {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})

    # Multi-class classification with imbalanced classes, for micro/macro/weighted averages.
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, n_redundant=2,
                               n_classes=5, weights=[0.4, 0.25, 0.15, 0.12, 0.08],
                               class_sep=1.0, flip_y=0.01, random_state=SEED)
    multi = pd.DataFrame(np.round(X, 6), columns=[f"f{i}" for i in range(X.shape[1])])
    multi["label"] = y
    multi.to_csv(DATA_DIR / "classification_multiclass.csv", index=False)
    log.info("classification_multiclass.csv: %d rows x %d features | class balance=%s",
             *X.shape, {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})

    # Regression with a strictly positive target so MAPE and RMSLE are well defined.
    X, y = make_regression(n_samples=2000, n_features=10, n_informative=8, noise=75.0,
                           random_state=SEED)
    y = y - y.min() + 1.0
    reg = pd.DataFrame(np.round(X, 6), columns=[f"f{i}" for i in range(X.shape[1])])
    reg["target"] = np.round(y, 6)
    reg.to_csv(DATA_DIR / "regression.csv", index=False)
    log.info("regression.csv: %d rows x %d features | target min=%.2f max=%.2f mean=%.2f",
             *X.shape, y.min(), y.max(), y.mean())

    # Ranking: queries x candidate items. Graded relevance 0-3 from a latent score;
    # the model's score is the latent score plus noise (an imperfect ranker).
    rng = np.random.default_rng(SEED)
    n_queries, n_items = 200, 10
    rows = []
    for q in range(n_queries):
        latent = rng.normal(0, 1, n_items)
        grade = np.clip(np.round(latent + 1.0), 0, 3).astype(int)  # relevance grade 0..3
        score = latent + rng.normal(0, 0.8, n_items)               # noisy model score
        for i in range(n_items):
            rows.append({"query_id": q, "item_id": i, "relevance": int(grade[i]),
                         "score": round(float(score[i]), 6)})
    rank = pd.DataFrame(rows)
    rank.to_csv(DATA_DIR / "ranking.csv", index=False)
    log.info("ranking.csv: %d queries x %d items | relevance distribution=%s",
             n_queries, n_items, rank["relevance"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
