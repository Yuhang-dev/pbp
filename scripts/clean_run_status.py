from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.io import ensure_parent, write_json
from pbp.logging_utils import RunLogger, finalize_run, initialize_run, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark stale running run status files as interrupted or failed.")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-dir-name-contains", action="append", default=None)
    parser.add_argument("--older-than-minutes", type=float, default=0.0)
    parser.add_argument("--to-status", choices=["failed", "interrupted"], default="interrupted")
    parser.add_argument("--note", default="Marked stale by scripts/clean_run_status.py")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _runtime_seconds(start_time: Any, end_time: datetime) -> float | None:
    start = _parse_time(start_time)
    if start is None:
        return None
    return round((end_time - start).total_seconds(), 6)


def _matches_filters(run_dir: Path, needles: list[str] | None) -> bool:
    if not needles:
        return True
    name = run_dir.name.lower()
    return any(needle.lower() in name for needle in needles)


def discover_stale_statuses(
    *,
    runs_dir: Path,
    excluded_run_dir: Path,
    name_contains: list[str] | None,
    older_than_minutes: float,
) -> list[tuple[Path, dict[str, Any]]]:
    now = datetime.now(timezone.utc)
    matches: list[tuple[Path, dict[str, Any]]] = []
    for status_path in sorted(runs_dir.glob("*/status.json")):
        run_dir = status_path.parent
        if run_dir.resolve() == excluded_run_dir.resolve():
            continue
        if not _matches_filters(run_dir, name_contains):
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("status") != "running":
            continue
        if older_than_minutes > 0:
            start = _parse_time(status.get("start_time"))
            if start is not None and (now - start).total_seconds() < older_than_minutes * 60:
                continue
        matches.append((status_path, status))
    return matches


def mark_status(
    *,
    status_path: Path,
    status: dict[str, Any],
    to_status: str,
    note: str,
) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    notes = status.get("notes")
    if notes is None:
        notes_list: list[str] = []
    elif isinstance(notes, list):
        notes_list = [str(item) for item in notes]
    else:
        notes_list = [str(notes)]
    notes_list.append(note)

    updated = {
        **status,
        "status": to_status,
        "end_time": utc_timestamp(),
        "runtime_seconds": _runtime_seconds(status.get("start_time"), end),
        "error": status.get("error") or note,
        "notes": notes_list,
    }
    write_json(updated, status_path)
    return updated


def config_for_run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/clean_run_status.py",
        "model": None,
        "base_model": None,
        "dataset": None,
        "data_path": args.runs_dir,
        "seed": args.seed,
        "dtype": None,
        "device": "cpu",
        "batch_size": None,
        "max_samples": None,
        "output_path": args.out,
        "notes": "M10B stale run-status cleanup",
        "name_contains": args.runs_dir_name_contains,
        "older_than_minutes": args.older_than_minutes,
        "to_status": args.to_status,
        "dry_run": args.dry_run,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}. Use --overwrite.")
        stale = discover_stale_statuses(
            runs_dir=Path(args.runs_dir),
            excluded_run_dir=run_paths.run_dir,
            name_contains=args.runs_dir_name_contains,
            older_than_minutes=args.older_than_minutes,
        )
        records: list[dict[str, Any]] = []
        for status_path, status in stale:
            if args.dry_run:
                new_status = None
            else:
                new_status = mark_status(
                    status_path=status_path,
                    status=status,
                    to_status=args.to_status,
                    note=args.note,
                )
            records.append(
                {
                    "status_path": str(status_path),
                    "previous_status": status,
                    "new_status": new_status,
                    "changed": not args.dry_run,
                }
            )

        report = {
            "runs_dir": args.runs_dir,
            "num_running_statuses_matched": len(stale),
            "num_statuses_changed": 0 if args.dry_run else len(stale),
            "to_status": args.to_status,
            "dry_run": args.dry_run,
            "records": records,
        }
        ensure_parent(out_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "num_running_statuses_matched": len(stale),
            "num_statuses_changed": 0 if args.dry_run else len(stale),
            "to_status": args.to_status,
            "dry_run": args.dry_run,
            "out": str(out_path),
        }
        finalize_run(run_paths, start_monotonic=start, metrics=metrics)
        message = {"run_id": run_paths.run_id, "out": str(out_path), "metrics": metrics}
        print(json.dumps(message, ensure_ascii=False, indent=2))
        logger.stdout(json.dumps(message, ensure_ascii=False))
    except Exception as exc:
        logger.stderr(f"{type(exc).__name__}: {exc}")
        finalize_run(run_paths, start_monotonic=start, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
