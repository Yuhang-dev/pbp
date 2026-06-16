from __future__ import annotations

import json

import pytest

from pbp.logging_utils import create_run_dir, initialize_run, sanitize_run_name


def test_sanitize_run_name():
    assert sanitize_run_name("M0 Run Dir Smoke") == "m0_run_dir_smoke"
    assert sanitize_run_name("abc-123") == "abc-123"
    with pytest.raises(ValueError):
        sanitize_run_name("   !!!   ")


def test_create_run_dir_creates_required_layout(tmp_path):
    paths = create_run_dir("m0 smoke", out_root=tmp_path, timestamp="20260616_223500")

    assert paths.run_id == "20260616_223500_m0_smoke"
    assert paths.run_dir.is_dir()
    assert paths.artifacts_dir.is_dir()
    assert paths.config_path.name == "config.yaml"
    assert paths.command_path.name == "command.sh"
    assert paths.stdout_path.name == "stdout.log"
    assert paths.stderr_path.name == "stderr.log"
    assert paths.metrics_path.name == "metrics.json"
    assert paths.status_path.name == "status.json"
    assert paths.environment_path.name == "environment.json"

    with pytest.raises(FileExistsError):
        create_run_dir("m0 smoke", out_root=tmp_path, timestamp="20260616_223500")


def test_initialize_run_writes_required_files(tmp_path):
    paths = initialize_run(
        "m0 helper",
        out_root=tmp_path,
        timestamp="20260616_223501",
        command="python -m pytest tests/test_logging_utils.py",
        config={
            "script": "pytest",
            "model": None,
            "base_model": None,
            "dataset": None,
            "data_path": None,
            "seed": 42,
            "dtype": None,
            "device": "cpu",
            "batch_size": None,
            "max_samples": None,
            "output_path": None,
            "notes": "M0 run directory helper test",
        },
    )

    assert paths.config_path.is_file()
    assert paths.command_path.read_text(encoding="utf-8").strip() == "python -m pytest tests/test_logging_utils.py"
    assert json.loads(paths.metrics_path.read_text(encoding="utf-8")) == {}
    status = json.loads(paths.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "running"
    env = json.loads(paths.environment_path.read_text(encoding="utf-8"))
    assert "python_version" in env
