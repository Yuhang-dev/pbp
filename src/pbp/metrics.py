from __future__ import annotations

from typing import Any

import numpy as np


def positive_margin_thresholds(deltas: list[float]) -> dict[str, float]:
    positive = np.asarray([d for d in deltas if d > 0], dtype=float)
    if positive.size == 0:
        raise ValueError("Cannot compute q25/q50/q75: no positive dense margins")
    return {
        "0": 0.0,
        "q25": float(np.quantile(positive, 0.25)),
        "q50": float(np.quantile(positive, 0.50)),
        "q75": float(np.quantile(positive, 0.75)),
    }


def coverage_at_thresholds(
    deltas: list[float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    if not deltas:
        raise ValueError("No dense margins provided")
    if thresholds is None:
        thresholds = positive_margin_thresholds(deltas)
    values = np.asarray(deltas, dtype=float)
    return {
        f"coverage@{name}": float(np.mean(values > tau))
        for name, tau in thresholds.items()
    }


def coverage_at_thresholds_snake(
    deltas: list[float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = positive_margin_thresholds(deltas)
    values = np.asarray(deltas, dtype=float)
    return {
        f"coverage_at_{name}": float(np.mean(values > tau))
        for name, tau in thresholds.items()
    }


def preference_accuracy(deltas: list[float]) -> float:
    if not deltas:
        raise ValueError("No dense margins provided")
    return float(np.mean(np.asarray(deltas, dtype=float) > 0.0))


def bcr_at_thresholds(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    dense_key: str = "delta_dense",
    pruned_key: str = "delta_pruned",
    boundary: float = 0.0,
) -> dict[str, float]:
    if not records:
        raise ValueError("No margin records provided")
    out: dict[str, float] = {}
    for name, tau in thresholds.items():
        covered = [
            record
            for record in records
            if float(record[dense_key]) > tau
        ]
        if not covered:
            raise ValueError(f"Cannot compute BCR@{name}: no dense margins above threshold")
        out[f"bcr@{name}"] = float(
            np.mean([float(record[pruned_key]) <= boundary for record in covered])
        )
    return out


def bcr_at_thresholds_snake(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    dense_key: str = "delta_dense",
    pruned_key: str = "delta_pruned",
    boundary: float = 0.0,
) -> dict[str, float]:
    return {
        key.replace("bcr@", "bcr_at_"): value
        for key, value in bcr_at_thresholds(
            records,
            thresholds,
            dense_key=dense_key,
            pruned_key=pruned_key,
            boundary=boundary,
        ).items()
    }


def summarize_bcr(
    records: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("No margin records provided")
    dense = np.asarray([float(record["delta_dense"]) for record in records], dtype=float)
    pruned = np.asarray([float(record["delta_pruned"]) for record in records], dtype=float)
    drops = dense - pruned
    if thresholds is None:
        thresholds = positive_margin_thresholds(dense.tolist())
    summary: dict[str, Any] = {
        "num_pairs": int(dense.size),
        "thresholds": thresholds,
        "preference_accuracy_dense": float(np.mean(dense > 0.0)),
        "preference_accuracy_pruned": float(np.mean(pruned > 0.0)),
        "mean_margin_drop": float(np.mean(drops)),
        "median_margin_drop": float(np.median(drops)),
        "min_margin_drop": float(np.min(drops)),
        "max_margin_drop": float(np.max(drops)),
        "mean_delta_dense": float(np.mean(dense)),
        "mean_delta_pruned": float(np.mean(pruned)),
    }
    summary.update(coverage_at_thresholds(dense.tolist(), thresholds))
    summary.update(coverage_at_thresholds_snake(dense.tolist(), thresholds))
    summary.update(bcr_at_thresholds(records, thresholds))
    summary.update(bcr_at_thresholds_snake(records, thresholds))
    return summary


def histogram(deltas: list[float], bins: int = 30) -> dict[str, Any]:
    if not deltas:
        raise ValueError("No dense margins provided")
    counts, edges = np.histogram(np.asarray(deltas, dtype=float), bins=bins)
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def summarize_dense_margins(
    records: list[dict[str, Any]],
    *,
    bins: int = 30,
) -> dict[str, Any]:
    deltas = [float(record["delta_dense"]) for record in records]
    thresholds = positive_margin_thresholds(deltas)
    positive = np.asarray([d for d in deltas if d > 0], dtype=float)
    arr = np.asarray(deltas, dtype=float)
    summary: dict[str, Any] = {
        "num_pairs": int(arr.size),
        "thresholds": thresholds,
        "positive_margin_quantiles": {
            "q25": float(np.quantile(positive, 0.25)),
            "q50": float(np.quantile(positive, 0.50)),
            "q75": float(np.quantile(positive, 0.75)),
        },
        "preference_accuracy": preference_accuracy(deltas),
        "mean_delta_dense": float(np.mean(arr)),
        "median_delta_dense": float(np.median(arr)),
        "mean_dense_margin": float(np.mean(arr)),
        "std_dense_margin": float(np.std(arr)),
        "min_dense_margin": float(np.min(arr)),
        "max_dense_margin": float(np.max(arr)),
        "margin_quantiles": {
            "q05": float(np.quantile(arr, 0.05)),
            "q25": float(np.quantile(arr, 0.25)),
            "q50": float(np.quantile(arr, 0.50)),
            "q75": float(np.quantile(arr, 0.75)),
            "q95": float(np.quantile(arr, 0.95)),
        },
        "histogram": histogram(deltas, bins=bins),
    }
    summary.update(coverage_at_thresholds(deltas, thresholds))
    summary.update(coverage_at_thresholds_snake(deltas, thresholds))
    return summary


def histogram_rows(hist: dict[str, Any]) -> list[dict[str, float | int]]:
    edges = hist["bin_edges"]
    counts = hist["counts"]
    return [
        {
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]
