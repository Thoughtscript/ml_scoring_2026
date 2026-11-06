"""Multi-class averages: per-class one-vs-rest TP/FP/FN/TN, then micro/macro/weighted."""
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from common import scores
from common.scores import show

CLASSES = list(range(5))
CASES = [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]
AVGS = ["micro", "macro", "weighted"]


@pytest.mark.parametrize("c", CLASSES)
def test_per_class_scores(multiclass, c):
    tp, fp, fn, tn = multiclass["counts"]
    hand = {k: float(v[c]) for k, v in multiclass["per_class"].items()}
    ref = {
        "precision": precision_score(multiclass["y"], multiclass["pred"], average=None,
                                     zero_division=0)[c],
        "recall": recall_score(multiclass["y"], multiclass["pred"], average=None,
                               zero_division=0)[c],
        "f1": f1_score(multiclass["y"], multiclass["pred"], average=None, zero_division=0)[c],
    }
    show(f"class={c}", "one-vs-rest counts",
         f"TP={tp[c]} FP={fp[c]} FN={fn[c]} TN={tn[c]} support={multiclass['support'][c]}")
    for name in ("precision", "recall", "f1"):
        show(f"  {name}[class={c}]", f"per-class {name}", hand[name], float(ref[name]))
        assert hand[name] == pytest.approx(ref[name], abs=1e-6)


@pytest.mark.parametrize("avg", AVGS)
@pytest.mark.parametrize("name,fn", CASES)
def test_average(multiclass, name, fn, avg):
    hand = float(multiclass["averages"][f"{name}_{avg}"])
    ref = fn(multiclass["y"], multiclass["pred"], average=avg, zero_division=0)
    show(f"{name}_{avg}", scores.AVERAGE_FORMULAS[avg], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_micro_f1_equals_accuracy(multiclass):
    """Single-label multi-class identity: micro P = micro R = micro F1 = accuracy."""
    hand = float(multiclass["averages"]["f1_micro"])
    ref = accuracy_score(multiclass["y"], multiclass["pred"])
    show("f1_micro == accuracy", "single-label multi-class identity", hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)
