"""Ranking scores: Precision@k, Recall@k (R1), Hit@k, MRR, MAP, NDCG@k — averaged over queries.

Hand-computed in common/scores.py; asserted here against scikit-learn (NDCG) or a second,
independent inline computation (the binary scores, which have no scikit-learn equivalent).
"""
import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from common import scores
from common.scores import show

F = scores.RANKING_FORMULAS
KS = [1, 3, 5, 10]
RELEVANT = 2  # grade >= 2 counts as relevant for the binary scores


def inline(query, k):
    """Second, independent computation of precision/recall/hit at k for one query."""
    rel, score = query
    top = (rel[np.argsort(-score)] >= RELEVANT)[:k]
    total = int((rel >= RELEVANT).sum())
    return top.sum() / k, (top.sum() / total if total else 0.0), float(top.any())


@pytest.mark.parametrize("k", KS)
def test_precision_at_k(ranking, k):
    hand = ranking["hand"][f"precision@{k}"]
    ref = float(
        np.mean([inline(q, k)[0] for q in ranking["queries"]])
    )
    show(f"precision@{k}", F["precision@k"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("k", KS)
def test_recall_at_k(ranking, k):
    hand = ranking["hand"][f"recall@{k}"]
    ref = float(
        np.mean([inline(q, k)[1] for q in ranking["queries"]])
    )
    show(f"recall@{k}", F["recall@k"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_r1_recall_at_1(ranking):
    """R1 = Recall@1: recall at the very first position."""
    hand = ranking["hand"]["recall@1"]
    ref = float(np.mean([inline(q, 1)[1] for q in ranking["queries"]]))
    show("R1 = recall@1", F["recall@k"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("k", KS)
def test_hit_at_k(ranking, k):
    hand = ranking["hand"][f"hit@{k}"]
    ref = float(
        np.mean([inline(q, k)[2] for q in ranking["queries"]])
    )
    show(f"hit@{k}", F["hit@k"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("k", KS)
def test_ndcg_at_k(ranking, k):
    hand = ranking["hand"][f"ndcg@{k}"]
    ref = float(
        np.mean([
            ndcg_score(r[None, :], s[None, :], k=k)
            for r, s in ranking["queries"]
        ])
    )
    show(f"ndcg@{k}", F["ndcg@k"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_mrr(ranking):
    hand = ranking["hand"]["mrr"]
    refs = []
    for rel, score in ranking["queries"]:
        hits = np.flatnonzero(rel[np.argsort(-score)] >= RELEVANT)
        refs.append(
            1.0 / (hits[0] + 1) if len(hits) else 0.0
        )
    ref = float(np.mean(refs))
    show("mrr", F["mrr"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)
    assert hand >= ranking["hand"]["hit@1"] - 1e-9


def test_map(ranking):
    hand = ranking["hand"]["map"]
    refs = []
    for rel, score in ranking["queries"]:
        binary = (rel[np.argsort(-score)] >= RELEVANT).astype(float)
        hits = np.flatnonzero(binary)
        refs.append(
            float(np.mean(np.cumsum(binary)[hits] / (hits + 1)))
            if len(hits) else 0.0
        )
    ref = float(np.mean(refs))
    show("map", F["map"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_recall_and_hit_grow_with_k(ranking):
    recalls = [ranking["hand"][f"recall@{k}"] for k in KS]
    hits = [ranking["hand"][f"hit@{k}"] for k in KS]
    show("recall@k by k", "monotone non-decreasing", " ".join(f"{r:.4f}" for r in recalls))
    show("hit@k by k", "monotone non-decreasing", " ".join(f"{h:.4f}" for h in hits))
    assert all(a <= b + 1e-9 for a, b in zip(recalls, recalls[1:]))
    assert all(a <= b + 1e-9 for a, b in zip(hits, hits[1:]))
