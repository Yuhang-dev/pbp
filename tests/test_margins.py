from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pbp.margins import compute_preference_margin, dense_margin_record


def test_compute_preference_margin_uses_base_normalized_pair_difference():
    delta = compute_preference_margin(
        ell_model_chosen=-1.0,
        ell_model_rejected=-2.0,
        ell_base_chosen=-1.5,
        ell_base_rejected=-1.75,
    )

    assert delta == pytest.approx(0.75)


def test_dense_margin_record_extracts_length_normalized_logprobs():
    record = dense_margin_record(
        "item-1",
        dense_chosen={"length_normalized_logprob": -1.0},
        dense_rejected={"length_normalized_logprob": -2.0},
        base_record={
            "chosen": {"length_normalized_logprob": -1.5},
            "rejected": {"length_normalized_logprob": -1.75},
        },
        prompt_sha256="abc123",
    )

    assert record["id"] == "item-1"
    assert record["ell_dense_chosen"] == pytest.approx(-1.0)
    assert record["ell_dense_rejected"] == pytest.approx(-2.0)
    assert record["ell_base_chosen"] == pytest.approx(-1.5)
    assert record["ell_base_rejected"] == pytest.approx(-1.75)
    assert record["delta_dense"] == pytest.approx(0.75)
    assert record["prompt_sha256"] == "abc123"


def test_compute_dense_margins_cli_dry_run_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data = repo_root / "tests" / "fixtures" / "hh_rlhf_processed_fixture.jsonl"
    out = tmp_path / "dense_margins.jsonl"
    runs_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compute_dense_margins.py"),
            "--instruct-model",
            "dense-dry-run",
            "--base-model",
            "base-dry-run",
            "--data",
            str(data),
            "--out",
            str(out),
            "--runs-dir",
            str(runs_dir),
            "--run-name",
            "m3_dense_margin_dry_run_test",
            "--max-samples",
            "2",
            "--seed",
            "42",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["metrics"]["num_examples"] == 2
    assert payload["metrics"]["delta_dense_finite"] is True
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["ell_dense_chosen"] != record["ell_base_chosen"]
        assert record["ell_dense_rejected"] != record["ell_base_rejected"]
        assert record["delta_dense"] > 0
        assert record["dry_run"] is True
    run_dirs = list(runs_dir.glob("*_m3_dense_margin_dry_run_test"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "metrics.json").is_file()
    assert (run_dirs[0] / "status.json").is_file()
