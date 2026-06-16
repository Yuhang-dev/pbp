from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from pbp.io import ensure_parent, write_json


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_timestamp_for_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def sanitize_run_name(run_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        raise ValueError("run_name must contain at least one alphanumeric character")
    return cleaned.lower()


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment(cwd: str | Path | None = None) -> dict[str, Any]:
    torch_version = None
    cuda_available = False
    gpu_name = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "transformers_version": _package_version("transformers"),
        "datasets_version": _package_version("datasets"),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "git_commit": get_git_commit(cwd),
    }


def _dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    out_path = ensure_parent(path)
    try:
        import yaml

        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except Exception:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    out_path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    artifacts_dir: Path
    config_path: Path
    command_path: Path
    stdout_path: Path
    stderr_path: Path
    metrics_path: Path
    status_path: Path
    environment_path: Path


def create_run_dir(
    run_name: str,
    *,
    out_root: str | Path = "outputs/runs",
    timestamp: str | None = None,
) -> RunPaths:
    timestamp = timestamp or local_timestamp_for_run_id()
    run_id = f"{timestamp}_{sanitize_run_name(run_name)}"
    run_dir = Path(out_root) / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        config_path=run_dir / "config.yaml",
        command_path=run_dir / "command.sh",
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        metrics_path=run_dir / "metrics.json",
        status_path=run_dir / "status.json",
        environment_path=run_dir / "environment.json",
    )


def initialize_run(
    run_name: str,
    *,
    config: dict[str, Any],
    command: str | list[str],
    out_root: str | Path = "outputs/runs",
    cwd: str | Path | None = None,
    timestamp: str | None = None,
) -> RunPaths:
    paths = create_run_dir(run_name, out_root=out_root, timestamp=timestamp)
    now = utc_timestamp()
    config_with_run = {
        "run_id": paths.run_id,
        "timestamp": now,
        **config,
    }
    command_text = command if isinstance(command, str) else " ".join(command)

    _dump_yaml(config_with_run, paths.config_path)
    ensure_parent(paths.command_path).write_text(command_text + "\n", encoding="utf-8")
    paths.stdout_path.write_text("", encoding="utf-8")
    paths.stderr_path.write_text("", encoding="utf-8")
    write_json({}, paths.metrics_path)
    write_json(collect_environment(cwd), paths.environment_path)
    write_json(
        {
            "status": "running",
            "start_time": now,
            "end_time": None,
            "runtime_seconds": None,
            "error": None,
        },
        paths.status_path,
    )
    return paths


def finalize_run(
    paths: RunPaths,
    *,
    start_monotonic: float,
    metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    if metrics is not None:
        write_json(metrics, paths.metrics_path)
    status = "failed" if error else "success"
    start_time = None
    if paths.status_path.exists():
        try:
            start_time = json.loads(paths.status_path.read_text(encoding="utf-8")).get("start_time")
        except Exception:
            start_time = None
    write_json(
        {
            "status": status,
            "start_time": start_time,
            "end_time": utc_timestamp(),
            "runtime_seconds": round(time.monotonic() - start_monotonic, 6),
            "error": error,
        },
        paths.status_path,
    )


class RunLogger:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    def stdout(self, message: str) -> None:
        with self.paths.stdout_path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + os.linesep)

    def stderr(self, message: str) -> None:
        with self.paths.stderr_path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + os.linesep)
