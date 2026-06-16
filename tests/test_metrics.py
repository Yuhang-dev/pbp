from __future__ import annotations

import pytest

from pbp.metrics import coverage_at_thresholds, positive_margin_thresholds, summarize_dense_margins


def test_coverage_uses_positive_dense_margin_quantile_thresholds():
    deltas = [1.0, 2.0, 3.0, -1.0]

    thresholds = positive_margin_thresholds(deltas)
    coverage = coverage_at_thresholds(deltas, thresholds)

    assert thresholds["0"] == pytest.approx(0.0)
    assert thresholds["q25"] == pytest.approx(1.5)
    assert thresholds["q50"] == pytest.approx(2.0)
    assert thresholds["q75"] == pytest.approx(2.5)
    assert coverage["coverage@0"] == pytest.approx(0.75)
    assert coverage["coverage@q25"] == pytest.approx(0.50)
    assert coverage["coverage@q50"] == pytest.approx(0.25)
    assert coverage["coverage@q75"] == pytest.approx(0.25)


def test_dense_margin_summary_reports_preference_accuracy_and_histogram():
    records = [
        {"delta_dense": 1.0},
        {"delta_dense": 2.0},
        {"delta_dense": 3.0},
        {"delta_dense": -1.0},
    ]

    summary = summarize_dense_margins(records, bins=2)

    assert summary["num_pairs"] == 4
    assert summary["preference_accuracy"] == pytest.approx(0.75)
    assert summary["coverage@0"] == pytest.approx(0.75)
    assert len(summary["histogram"]["counts"]) == 2
