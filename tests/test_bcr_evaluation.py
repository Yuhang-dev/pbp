from __future__ import annotations

import pytest

from pbp.metrics import bcr_at_thresholds, positive_margin_thresholds, summarize_bcr


def test_bcr_uses_dense_coverage_and_pruned_boundary_crossing():
    records = [
        {"delta_dense": 0.3, "delta_pruned": 0.2},
        {"delta_dense": 0.2, "delta_pruned": -0.1},
        {"delta_dense": 0.1, "delta_pruned": -0.2},
        {"delta_dense": -0.1, "delta_pruned": -0.3},
    ]
    thresholds = {"0": 0.0, "q25": 0.15, "q50": 0.2, "q75": 0.25}

    bcr = bcr_at_thresholds(records, thresholds)

    assert bcr["bcr@0"] == pytest.approx(2 / 3)
    assert bcr["bcr@q25"] == pytest.approx(1 / 2)
    assert bcr["bcr@q50"] == pytest.approx(0.0)
    assert bcr["bcr@q75"] == pytest.approx(0.0)


def test_summarize_bcr_reports_required_m7_metrics():
    records = [
        {"delta_dense": 0.3, "delta_pruned": 0.2},
        {"delta_dense": 0.2, "delta_pruned": -0.1},
        {"delta_dense": 0.1, "delta_pruned": -0.2},
        {"delta_dense": -0.1, "delta_pruned": -0.3},
    ]

    summary = summarize_bcr(records, thresholds=positive_margin_thresholds([0.3, 0.2, 0.1, -0.1]))

    assert summary["num_pairs"] == 4
    assert summary["coverage_at_0"] == pytest.approx(0.75)
    assert summary["bcr_at_0"] == pytest.approx(2 / 3)
    assert summary["preference_accuracy_dense"] == pytest.approx(0.75)
    assert summary["preference_accuracy_pruned"] == pytest.approx(0.25)
    assert summary["mean_margin_drop"] == pytest.approx(0.25)
