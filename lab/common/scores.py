"""Hand-rolled ML scores from first principles: TP/FP/FN/TN, residuals e = y - y_hat, and ranks.
Every function here is asserted against its scikit-learn equivalent in tests/ (stage 4)."""
import numpy as np

BINARY_FORMULAS = {
    "accuracy": "(TP+TN)/(TP+FP+FN+TN)",
    "precision": "TP/(TP+FP)",
    "recall": "TP/(TP+FN)",
    "specificity": "TN/(TN+FP)",
    "fpr": "FP/(FP+TN)",
    "fnr": "FN/(FN+TP)",
    "npv": "TN/(TN+FN)",
    "f1": "2TP/(2TP+FP+FN)",
    "f2": "(1+b^2)TP/((1+b^2)TP + b^2*FN + FP), b=2",
    "balanced_accuracy": "(TPR+TNR)/2",
    "mcc": "(TP*TN - FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
    "cohens_kappa": "(po - pe)/(1 - pe), pe from the marginals",
    "roc_auc": "P(positive scored above negative), over all thresholds",
    "pr_auc": "mean of precision at each relevant rank",
    "log_loss": "-mean(y*log(p) + (1-y)*log(1-p))",
}

AVERAGE_FORMULAS = {
    "micro": "pool TP/FP/FN over all classes, then compute the score",
    "macro": "unweighted mean of the per-class scores",
    "weighted": "support-weighted mean of the per-class scores",
}

REGRESSION_FORMULAS = {
    "mae": "mean|e|",
    "mse": "mean(e^2)",
    "rmse": "sqrt(mean(e^2))",
    "mape": "mean|e/y|  (x100 for %)",
    "smape": "mean(2|e|/(|y|+|y_hat|))",
    "r2": "1 - sum(e^2)/sum((y - mean(y))^2)",
    "adjusted_r2": "1 - (1 - R^2)(n-1)/(n-p-1)",
    "rmsle": "sqrt(mean((log(1+y) - log(1+y_hat))^2))",
    "explained_variance": "1 - Var(e)/Var(y)",
    "max_error": "max|e|",
    "median_ae": "median|e|",
}

RANKING_FORMULAS = {
    "precision@k": "(# relevant in top k) / k",
    "recall@k": "(# relevant in top k) / (total relevant)",
    "hit@k": "1 if any relevant in top k, else 0",
    "ndcg@k": "DCG@k / IDCG@k, DCG = sum(rel_i / log2(i+1))",
    "mrr": "1 / rank of first relevant",
    "map": "mean of precision at each relevant rank",
}


def show(metric, formula, hand, ref=None):
    """Print one score line, so every test assertion is visible with `pytest -s`."""
    value = f"{hand:.6f}" if isinstance(hand, float) else str(hand)
    print(f"    {metric}")
    print(f"    {formula}")
    line = f"    hand={value}"
    if ref is not None:
        check = f"{ref:.6f}" if isinstance(ref, float) else str(ref)
        line += f" | sklearn={check}"
    print(line)


# --- classification -----------------------------------------------------------

def confusion_counts(y_true, y_pred):
    """TP, FP, FN, TN for the positive class (label = 1)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def binary_scores(tp, fp, fn, tn):
    """The count-based binary scores, as closed-form expressions over TP/FP/FN/TN."""
    n = tp + fp + fn + tn
    tpr, tnr = tp / (tp + fn), tn / (tn + fp)
    pe = ((tp + fp) / n) * ((tp + fn) / n) + ((fn + tn) / n) * ((fp + tn) / n)
    return {
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp),
        "recall": tpr,
        "specificity": tnr,
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "npv": tn / (tn + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "f2": 5 * tp / (5 * tp + 4 * fn + fp),
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        "cohens_kappa": ((tp + tn) / n - pe) / (1 - pe),
    }


def hand_auc(y_true, proba):
    """ROC AUC as a Mann-Whitney U statistic: P(pos scored above neg) + 0.5 * P(tie)."""
    pos, neg = proba[y_true == 1], proba[y_true == 0]
    greater = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def hand_average_precision(y_true, proba):
    """AP = mean of precision at each relevant rank (items sorted by descending score)."""
    rel = y_true[np.argsort(-proba)]
    hits = np.flatnonzero(rel)
    return float(np.mean(np.cumsum(rel)[hits] / (hits + 1)))


def hand_log_loss(y_true, proba, eps=1e-15):
    p = np.clip(proba, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def safe_div(a, b):
    """Elementwise a/b with 0 where b == 0 (scikit-learn's zero_division=0)."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def one_vs_rest_counts(cm):
    """Per-class TP/FP/FN/TN from a multi-class confusion matrix (rows=true, cols=pred)."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    return tp, fp, fn, tn


def per_class_scores(tp, fp, fn):
    """One-vs-rest precision, recall, and F1 for each class."""
    return {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
    }


def average_scores(tp, fp, fn, support):
    """Micro/macro/weighted precision, recall, and F1 from per-class counts.

    Micro pools the counts first (so micro P = micro R = micro F1 = accuracy for
    single-label multi-class); macro averages per-class scores unweighted; weighted
    averages them by support."""
    per = per_class_scores(tp, fp, fn)
    return {
        "precision_micro": safe_div(tp.sum(), tp.sum() + fp.sum()),
        "precision_macro": per["precision"].mean(),
        "precision_weighted": np.average(per["precision"], weights=support),
        "recall_micro": safe_div(tp.sum(), tp.sum() + fn.sum()),
        "recall_macro": per["recall"].mean(),
        "recall_weighted": np.average(per["recall"], weights=support),
        "f1_micro": safe_div(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum()),
        "f1_macro": per["f1"].mean(),
        "f1_weighted": np.average(per["f1"], weights=support),
    }


# --- regression ---------------------------------------------------------------

def regression_scores(y, y_hat, p):
    """The regression scores from the residuals e = y - y_hat (n rows, p features)."""
    e = y - y_hat
    n = len(y)
    y_hat_pos = np.clip(y_hat, 0, None)  # RMSLE needs non-negative predictions
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "mae": np.mean(np.abs(e)),
        "mse": np.mean(e ** 2),
        "rmse": np.sqrt(np.mean(e ** 2)),
        "mape": np.mean(np.abs(e / y)),
        "smape": np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(y_hat))),
        "r2": r2,
        "adjusted_r2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
        "rmsle": np.sqrt(np.mean((np.log1p(y) - np.log1p(y_hat_pos)) ** 2)),
        "explained_variance": 1 - np.var(e) / np.var(y),
        "max_error": np.max(np.abs(e)),
        "median_ae": np.median(np.abs(e)),
    }


# --- ranking ------------------------------------------------------------------

def query_scores(relevance, score, ks=(1, 3, 5, 10), relevant=2):
    """All ranking scores for one query: items sorted by descending model score.

    Binary scores treat grade >= `relevant` as relevant; NDCG uses the grades."""
    order = np.argsort(-score)
    graded = relevance[order].astype(float)
    binary = (graded >= relevant).astype(float)
    n_rel = binary.sum()
    out = {}
    for k in ks:
        top = binary[:k]
        out[f"precision@{k}"] = top.sum() / k
        out[f"recall@{k}"] = top.sum() / n_rel if n_rel else 0.0
        out[f"hit@{k}"] = float(top.sum() > 0)
        discounts = 1 / np.log2(np.arange(2, k + 2))  # position i -> 1/log2(i+1)
        dcg = float(np.sum(graded[:k] * discounts))
        idcg = float(np.sum(np.sort(graded)[::-1][:k] * discounts))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    hits = np.flatnonzero(binary)
    out["mrr"] = 1.0 / (hits[0] + 1) if len(hits) else 0.0
    out["map"] = float(np.mean(np.cumsum(binary)[hits] / (hits + 1))) if len(hits) else 0.0
    return out
