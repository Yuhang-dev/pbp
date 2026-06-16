from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pbp.data import prepare_preference_records, split_hh_rlhf_pair, validate_disjoint_splits


def test_split_hh_rlhf_pair_uses_final_assistant_turn_as_prompt_boundary():
    chosen = "\n\nHuman: hello\n\nAssistant: helpful answer"
    rejected = "\n\nHuman: hello\n\nAssistant: unhelpful answer"

    prompt, chosen_response, rejected_response = split_hh_rlhf_pair(chosen, rejected)

    assert prompt == "\n\nHuman: hello\n\nAssistant:"
    assert chosen_response == " helpful answer"
    assert rejected_response == " unhelpful answer"


def test_prepare_preference_records_creates_disjoint_valid_splits():
    raw_records = [
        {
            "chosen": f"\n\nHuman: prompt {i}\n\nAssistant: chosen {i}",
            "rejected": f"\n\nHuman: prompt {i}\n\nAssistant: rejected {i}",
        }
        for i in range(5)
    ]

    calib, eval_records, skipped = prepare_preference_records(
        raw_records,
        calib_size=2,
        eval_size=2,
        seed=42,
    )

    assert len(calib) == 2
    assert len(eval_records) == 2
    assert skipped == 0
    validate_disjoint_splits(calib, eval_records)
    for record in calib + eval_records:
        assert record["prompt"]
        assert record["chosen"]
        assert record["rejected"]
        assert record["source"] == "hh-rlhf"


def test_prepare_hh_rlhf_cli_fixture_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    input_jsonl = repo_root / "tests" / "fixtures" / "hh_rlhf_raw_fixture.jsonl"
    out_dir = tmp_path / "processed"
    runs_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_hh_rlhf.py"),
            "--input-jsonl",
            str(input_jsonl),
            "--calib-size",
            "2",
            "--eval-size",
            "2",
            "--seed",
            "42",
            "--out-dir",
            str(out_dir),
            "--runs-dir",
            str(runs_dir),
            "--run-name",
            "m1_fixture_test",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["metrics"]["num_calib_records"] == 2
    assert payload["metrics"]["num_eval_records"] == 2
    assert (out_dir / "hh_rlhf_calib.jsonl").is_file()
    assert (out_dir / "hh_rlhf_eval.jsonl").is_file()
    run_dirs = list(runs_dir.glob("*_m1_fixture_test"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "config.yaml").is_file()
    assert (run_dirs[0] / "metrics.json").is_file()
    assert (run_dirs[0] / "status.json").is_file()
