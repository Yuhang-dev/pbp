from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pbp.metrics import coverage_at_thresholds, positive_margin_thresholds, summarize_dense_margins


def test_coverage_uses_positive_dense_margin_quantiles():
    deltas = [0.1, 0.2, 0.3, -0.2]

    thresholds = positive_margin_thresholds(deltas)
    coverage = coverage_at_thresholds(deltas, thresholds)

    assert thresholds["0"] == pytest.approx(0.0)
    assert thresholds["q25"] == pytest.approx(0.15)
    assert thresholds["q50"] == pytest.approx(0.2)
    assert thresholds["q75"] == pytest.approx(0.25)
    assert coverage["coverage@0"] == pytest.approx(0.75)
    assert coverage["coverage@q25"] == pytest.approx(0.50)
    assert coverage["coverage@q50"] == pytest.approx(0.25)
    assert coverage["coverage@q75"] == pytest.approx(0.25)


def test_dense_margin_summary_includes_m4_required_metrics():
    records = [
        {"delta_dense": 0.1},
        {"delta_dense": 0.2},
        {"delta_dense": 0.3},
        {"delta_dense": -0.2},
    ]

    summary = summarize_dense_margins(records, bins=2)

    assert summary["num_pairs"] == 4
    assert summary["preference_accuracy"] == pytest.approx(0.75)
    assert summary["coverage_at_0"] == pytest.approx(0.75)
    assert summary["coverage_at_q25"] == pytest.approx(0.50)
    assert summary["mean_delta_dense"] == pytest.approx(0.1)
    assert summary["median_delta_dense"] == pytest.approx(0.15)
    assert summary["positive_margin_quantiles"]["q25"] == pytest.approx(0.15)
    assert len(summary["histogram"]["counts"]) == 2


def test_report_coverage_cli_fixture_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dense_margins = repo_root / "tests" / "fixtures" / "dense_margins_fixture.jsonl"
    out = tmp_path / "coverage.json"
    histogram_out = tmp_path / "histogram.csv"
    runs_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "report_coverage.py"),
            "--dense-margins",
            str(dense_margins),
            "--out",
            str(out),
            "--histogram-out",
            str(histogram_out),
            "--histogram-bins",
            "2",
            "--runs-dir",
            str(runs_dir),
            "--run-name",
            "m4_coverage_fixture_test",
            "--seed",
            "42",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["metrics"]["num_examples"] == 4
    assert payload["metrics"]["coverage_metrics_valid"] is True
    assert payload["metrics"]["numeric_metrics_finite"] is True

    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["coverage_at_0"] == pytest.approx(0.75)
    assert summary["coverage_at_q25"] == pytest.approx(0.50)
    assert summary["preference_accuracy"] == pytest.approx(0.75)

    with histogram_out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    run_dirs = list(runs_dir.glob("*_m4_coverage_fixture_test"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "metrics.json").is_file()
    assert (run_dirs[0] / "status.json").is_file()
