"""Binary classification scores: hand-computed from TP/FP/FN/TN, asserted against scikit-learn."""
import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, fbeta_score,
                             log_loss, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from common import scores
from common.scores import show

F = scores.BINARY_FORMULAS
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]


def test_confusion_counts(binary):
    tp, fp, fn, tn = binary["counts"]
    tn_sk, fp_sk, fn_sk, tp_sk = confusion_matrix(binary["y"], binary["pred"]).ravel()
    show("confusion matrix", "counts at threshold=0.5", f"TP={tp} FP={fp} FN={fn} TN={tn}")
    assert (tp, fp, fn, tn) == (int(tp_sk), int(fp_sk), int(fn_sk), int(tn_sk))
    assert tp + fp + fn + tn == len(binary["y"])


def test_accuracy(binary):
    hand = binary["scores"]["accuracy"]
    ref = accuracy_score(binary["y"], binary["pred"])
    show("accuracy", F["accuracy"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_precision(binary):
    hand = binary["scores"]["precision"]
    ref = precision_score(binary["y"], binary["pred"])
    show("precision", F["precision"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_recall(binary):
    hand = binary["scores"]["recall"]
    ref = recall_score(binary["y"], binary["pred"])
    show("recall", F["recall"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_specificity(binary):
    hand = binary["scores"]["specificity"]
    tn, fp, fn, tp = confusion_matrix(binary["y"], binary["pred"]).ravel()
    ref = tn / (tn + fp)  # no direct scikit-learn function; check against its confusion matrix
    show("specificity", F["specificity"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)
    assert hand == pytest.approx(1 - binary["scores"]["fpr"], abs=1e-12)


def test_fpr(binary):
    hand = binary["scores"]["fpr"]
    tn, fp, fn, tp = confusion_matrix(binary["y"], binary["pred"]).ravel()
    ref = fp / (fp + tn)
    show("fpr", F["fpr"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_fnr(binary):
    hand = binary["scores"]["fnr"]
    ref = 1 - recall_score(binary["y"], binary["pred"])
    show("fnr", F["fnr"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_npv(binary):
    hand = binary["scores"]["npv"]
    tn, fp, fn, tp = confusion_matrix(binary["y"], binary["pred"]).ravel()
    ref = tn / (tn + fn)
    show("npv", F["npv"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_f1(binary):
    hand = binary["scores"]["f1"]
    ref = f1_score(binary["y"], binary["pred"])
    show("f1", F["f1"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_f1_is_harmonic_mean(binary):
    s = binary["scores"]
    harmonic = 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"])
    show("f1", "2PR/(P+R)", s["f1"], harmonic)
    assert s["f1"] == pytest.approx(harmonic, abs=1e-12)


def test_f2(binary):
    hand = binary["scores"]["f2"]
    ref = fbeta_score(binary["y"], binary["pred"], beta=2)
    show("f2", F["f2"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_balanced_accuracy(binary):
    hand = binary["scores"]["balanced_accuracy"]
    ref = balanced_accuracy_score(binary["y"], binary["pred"])
    show("balanced_accuracy", F["balanced_accuracy"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_mcc(binary):
    hand = binary["scores"]["mcc"]
    ref = matthews_corrcoef(binary["y"], binary["pred"])
    show("mcc", F["mcc"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_cohens_kappa(binary):
    hand = binary["scores"]["cohens_kappa"]
    ref = cohen_kappa_score(binary["y"], binary["pred"])
    show("cohens_kappa", F["cohens_kappa"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_roc_auc(binary):
    hand = scores.hand_auc(binary["y"], binary["proba"])
    ref = roc_auc_score(binary["y"], binary["proba"])
    show("roc_auc", F["roc_auc"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_average_precision(binary):
    hand = scores.hand_average_precision(binary["y"], binary["proba"])
    ref = average_precision_score(binary["y"], binary["proba"])
    show("average_precision", F["average_precision"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_average_precision_groups_tied_scores():
    y = np.array([1, 1, 0])
    proba = np.array([0.9, 0.5, 0.5])
    hand = scores.hand_average_precision(y, proba)
    ref = average_precision_score(y, proba)
    show("average_precision (tied scores)", F["average_precision"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_log_loss(binary):
    hand = scores.hand_log_loss(binary["y"], binary["proba"])
    ref = log_loss(binary["y"], binary["proba"])
    show("log_loss", F["log_loss"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("t", THRESHOLDS)
def test_threshold_sweep(binary, t):
    """TP/FP/FN/TN at each decision threshold: the precision-recall trade-off in action."""
    tp, fp, fn, tn = scores.confusion_counts(binary["y"], (binary["proba"] >= t).astype(int))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    show(f"threshold={t:.1f}", "scores from TP/FP/FN/TN",
         f"TP={tp} FP={fp} FN={fn} TN={tn} precision={p:.4f} recall={r:.4f} f1={f1:.4f}")
    tp0, fp0, fn0, tn0 = binary["counts"]
    assert tp + fn == tp0 + fn0  # the actual positives/negatives don't move with the threshold
    assert fp + tn == fp0 + tn0


def test_recall_falls_as_threshold_rises(binary):
    recalls = []
    for t in THRESHOLDS:
        tp, fp, fn, tn = scores.confusion_counts(binary["y"], (binary["proba"] >= t).astype(int))
        recalls.append(tp / (tp + fn))
    show("recall by threshold", "monotone non-increasing",
         " ".join(f"{r:.4f}" for r in recalls))
    assert all(a >= b - 1e-9 for a, b in zip(recalls, recalls[1:]))
