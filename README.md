# ml_scoring_2026

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)](https://www.python.org/downloads/) [![scikit-learn 1.9.1](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E)](https://github.com/scikit-learn/scikit-learn/releases/tag/1.9.1) [![pandas 3.1.0](https://img.shields.io/badge/pandas-3.1.0-150458)](https://pandas.pydata.org/) [![NumPy 2.5.4](https://img.shields.io/badge/NumPy-2.5.4-013243)](https://numpy.org/) [![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docs.docker.com/compose/)

> _A simple **ML Scoring** Lab._

## Overview

Train small models on randomized CSV datasets, then verify the **top general ML scoring techniques** as **unit tests**: every score is hand-computed from first principles in [scores.py](/Thoughtscript/ml_scoring_2026/blob/main/lab/common/scores.py) and asserted against scikit-learn. Each test prints the metric, its formula, and both values before asserting they match.

| Group | Scores | Tests |
| --- | --- | --- |
| **Classification (binary)** | Accuracy, Precision, Recall, Specificity, FPR, FNR, NPV, F1, F2, Balanced Accuracy, MCC, Cohen's κ, ROC AUC, PR AUC, Log Loss | [test_classification.py](/Thoughtscript/ml_scoring_2026/blob/main/lab/tests/test_classification.py) |
| **Classification (multi-class)** | Per-class TP/FP/FN/TN one-vs-rest; Precision / Recall / F1 **micro, macro, weighted** | [test_classification_averages.py](/Thoughtscript/ml_scoring_2026/blob/main/lab/tests/test_classification_averages.py) |
| **Regression** | MAE, MSE, RMSE, MAPE, SMAPE, R², Adjusted R², RMSLE, Explained Variance, Max / Median Error | [test_regression.py](/Thoughtscript/ml_scoring_2026/blob/main/lab/tests/test_regression.py) |
| **Ranking** | Precision@k, Recall@k (**R1** = Recall@1), Hit@k, MRR, MAP, NDCG@k | [test_ranking.py](/Thoughtscript/ml_scoring_2026/blob/main/lab/tests/test_ranking.py) |

Stages run sequentially: generate data → train classifiers → train regressor → **tests** (pytest, runs automatically in `docker compose up`). Data is written to `lab/data`, models to `lab/models`.

## The scores

### Confusion matrix

Every classification score is built from four counts. Positive = the class you are trying to detect (label `1`); for multi-class problems each class gets its own one-vs-rest counts:

|  | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actually positive** | **TP** — true positive | **FN** — false negative (miss) |
| **Actually negative** | **FP** — false positive (false alarm) | **TN** — true negative |

### Classification scores (binary)

| Score | Also called | Calculation | What it tells you |
| --- | --- | --- | --- |
| **Accuracy** | | (TP+TN) / (TP+FP+FN+TN) | Overall fraction correct. Misleading under class imbalance. |
| **Precision** | Positive predictive value | TP / (TP+FP) | Of everything flagged positive, how much really is. Watch when FPs are costly. |
| **Recall** | Sensitivity, TPR, hit rate | TP / (TP+FN) | Of all real positives, how many were caught. Watch when FNs are costly. |
| **Specificity** | TNR | TN / (TN+FP) | Of all real negatives, how many were correctly rejected. |
| **FPR** | Fall-out | FP / (FP+TN) | = 1 − specificity. The x-axis of the ROC curve. |
| **FNR** | Miss rate | FN / (FN+TP) | = 1 − recall. |
| **NPV** | Negative predictive value | TN / (TN+FN) | Of everything flagged negative, how much really is. |
| **F1** | F-score (β=1) | 2TP / (2TP+FP+FN) | Harmonic mean of precision and recall; punishes a lopsided trade-off. |
| **Fβ** | F2 shown | (1+β²)·TP / ((1+β²)·TP + β²·FN + FP) | F1 generalized: β>1 weighs recall over precision. |
| **Balanced accuracy** | | (TPR+TNR) / 2 = ½·(TP/(TP+FN) + TN/(TN+FP)) | Accuracy averaged per class; robust to imbalance. |
| **MCC** | Matthews correlation | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | Correlation in [−1, 1] using all four counts; chance = 0. |
| **Cohen's κ** | Kappa | (po − pe) / (1 − pe) | Accuracy corrected for the chance agreement pe implied by the marginals. |
| **ROC AUC** | AUC | Area under the TPR-vs-FPR curve = P(a random positive is scored above a random negative) | Threshold-free ranking quality. |
| **PR AUC** | Average precision | Mean of precision at each relevant rank | More informative than ROC AUC when positives are rare. |
| **Log loss** | Cross-entropy | −mean( y·log p + (1−y)·log(1−p) ) | Scores calibrated probabilities, not just the final class. |

### Classification averages (multi-class)

With C classes, class c gets one-vs-rest counts TP_c / FP_c / FN_c / TN_c and per-class scores P_c, R_c, F1_c. The three averagings differ only in how the classes are combined:

| Average | Calculation | What it tells you |
| --- | --- | --- |
| **Micro** | Pool the counts first: F1_micro = 2·ΣTP_c / (2·ΣTP_c + ΣFP_c + ΣFN_c) | Dominated by the big classes. For single-label multi-class, micro P = micro R = micro F1 = **accuracy**. |
| **Macro** | Unweighted mean: F1_macro = (1/C)·Σ F1_c | Every class counts equally — exposes rare classes doing badly. |
| **Weighted** | Support-weighted mean: F1_weighted = Σ (support_c · F1_c) / Σ support_c | Between the two; accounts for class frequency. |

### Regression scores

Residual e = y − ŷ over n rows with p features.

| Score | Calculation | What it tells you |
| --- | --- | --- |
| **MAE** | mean \|e\| | Average error in target units. Robust to outliers. |
| **MSE** | mean(e²) | Squared error; large errors dominate. |
| **RMSE** | √mean(e²) | MSE back in target units; still outlier-sensitive. |
| **MAPE** | mean \|e / y\| (×100 for %) | Error as a fraction of the actual. Undefined at y=0; skewed by small y. |
| **SMAPE** | mean( 2\|e\| / (\|y\| + \|ŷ\|) ) | Symmetric MAPE, bounded in [0, 2]. |
| **R²** | 1 − Σe² / Σ(y − ȳ)² | Fraction of variance explained: 1 is perfect, 0 is "predict the mean", <0 is worse. |
| **Adjusted R²** | 1 − (1−R²)(n−1)/(n−p−1) | R² penalized for extra features; use when comparing models with different p. |
| **RMSLE** | √mean( (log(1+y) − log(1+ŷ))² ) | Log-scale RMSE; penalizes under-prediction more; needs y ≥ 0. |
| **Explained variance** | 1 − Var(e) / Var(y) | Like R² but not centered; equals R² when the mean residual is 0. |
| **Max error** | max \|e\| | Worst-case single prediction. |
| **Median AE** | median \|e\| | Typical error; very robust to outliers. |

### Ranking scores

Computed per query on items sorted by descending model score, then averaged. Graded relevance 0–3; an item counts as **relevant** when grade ≥ 2.

| Score | Calculation | What it tells you |
| --- | --- | --- |
| **Precision@k** | (# relevant in top k) / k | Quality of the first k results. |
| **Recall@k** | (# relevant in top k) / (total relevant) | Coverage of all relevant items in the first k. |
| **R1** | Recall@1 | Recall at the very first position — the "R1" of retrieval/ranking. |
| **Hit@k** | 1 if any relevant in top k, else 0 | Did the user get at least one good result? |
| **MRR** | mean(1 / rank of first relevant) | Rewards putting the first relevant item on top. |
| **MAP** | Mean over queries of (mean of precision at each relevant rank) | Precision across the whole ranked list. |
| **NDCG@k** | DCG@k / IDCG@k, with DCG = Σ relᵢ / log₂(i+1) | Graded relevance with a position discount; 1 = ideal ordering. |

> Note: **R1** (Recall@1) is a ranking score — don't confuse it with **R²**, the regression score above. Both are demonstrated here.

## Use

```
docker compose up
```

The test stage runs automatically as the last container. To run the tests directly (after stages 1–3 have produced `data/` and `models/`):

```
cd lab
python -m pytest tests -v -s     # -s shows each test's metric / formula / values
```

## What to look for

* **Hand vs scikit-learn**: every score is computed twice and asserted equal (`pytest.approx`) — the formulas above are exactly what runs. Follow the suite with `docker compose logs -f tests`.
* **Binary classification**: the data is 70/30 imbalanced, so accuracy 0.79 looks fine while recall 0.53 and F1 0.60 tell the real story. `test_threshold_sweep` prints TP/FP/FN/TN and the precision–recall trade-off at each decision threshold (recall 0.95 at t=0.1 → 0.03 at t=0.9).
* **Multi-class**: `test_micro_f1_equals_accuracy` shows micro-F1 = accuracy = 0.67 exactly (the single-label identity); macro-F1 0.57 < weighted-F1 0.65 because the rare classes (support 41–61) score worst — macro is the average that exposes them.
* **Regression**: same errors, different stories — MAE 58.0 vs RMSE 72.1 (outliers weigh more in RMSE), MAPE 8.2%, R² 0.877, max error 207 vs median AE 47.
* **Ranking**: the ranker almost always surfaces relevant items somewhere in the top 10 (recall@10 = 0.97) but R1 = recall@1 = 0.30 and precision@1 = 0.80 — being right at the very top is the hard part.

Generated data and models are included for study; they are recreated on each run.

The above frequently show up in the GCP PMLE but there aren't that many hands-on labs for these in [skills.google](https://www.skills.google/).

## Resources and Links

1. [https://www.skills.google/](https://www.skills.google/)
2. [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)