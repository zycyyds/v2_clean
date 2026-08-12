from __future__ import annotations

from typing import Any

import numpy as np


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    tn = int(np.sum(~labels & ~predictions))
    fn = int(np.sum(labels & ~predictions))
    return tp, fp, tn, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return (2.0 * tp / denominator) if denominator else 0.0


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order].astype(np.int64)
    sorted_probabilities = probabilities[order]
    positives = int(sorted_labels.sum())
    if positives == 0:
        return float("nan")
    true_positives = 0
    seen = 0
    average_precision = 0.0
    position = 0
    while position < len(labels):
        end = position + 1
        while (
            end < len(labels)
            and sorted_probabilities[end] == sorted_probabilities[position]
        ):
            end += 1
        group_positives = int(sorted_labels[position:end].sum())
        true_positives += group_positives
        seen = end
        average_precision += (group_positives / positives) * (true_positives / seen)
        position = end
    return float(average_precision)


def _auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def select_macro_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be aligned one-dimensional arrays")
    if not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("threshold selection requires finite, non-empty probabilities")
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    total_positive = int(labels.sum())
    total_negative = len(labels) - total_positive
    tp = fp = 0
    # Start with the all-clean classifier. Ties prefer higher dirty recall and
    # then the lower threshold, so the rule is deterministic and sensitivity-led.
    best_score = _f1(total_negative, total_positive, 0) / 2.0
    best_tp = 0
    best_threshold = float(np.nextafter(sorted_probabilities.max(initial=1.0), np.inf))
    position = 0
    while position < len(labels):
        threshold = float(sorted_probabilities[position])
        end = position + 1
        while end < len(labels) and sorted_probabilities[end] == threshold:
            end += 1
        group_positive = int(sorted_labels[position:end].sum())
        tp += group_positive
        fp += (end - position) - group_positive
        fn = total_positive - tp
        tn = total_negative - fp
        score = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2.0
        if (
            score > best_score + 1e-15
            or (
                abs(score - best_score) <= 1e-15
                and (tp > best_tp or (tp == best_tp and threshold < best_threshold))
            )
        ):
            best_score = score
            best_tp = tp
            best_threshold = threshold
        position = end
    return best_threshold


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= threshold
    tp, fp, tn, fn = _confusion(labels, predictions)
    dirty_precision = tp / (tp + fp) if tp + fp else 0.0
    dirty_recall = tp / (tp + fn) if tp + fn else 0.0
    dirty_f1 = _f1(tp, fp, fn)
    clean_f1 = _f1(tn, fn, fp)
    average_precision = _average_precision(labels, probabilities)
    prevalence = float(labels.mean()) if len(labels) else float("nan")
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "positive_prevalence": prevalence,
        "threshold": float(threshold),
        "average_precision": average_precision,
        "auprc": average_precision,
        "auprc_method": "average_precision",
        "auprc_lift": (
            average_precision / prevalence if len(labels) and prevalence > 0 else float("nan")
        ),
        "auroc": _auroc(labels, probabilities),
        "macro_f1": (dirty_f1 + clean_f1) / 2.0,
        "dirty_precision": dirty_precision,
        "dirty_recall": dirty_recall,
        "dirty_f1": dirty_f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
