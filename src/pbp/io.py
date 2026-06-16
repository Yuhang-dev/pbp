from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    out_path = ensure_parent(path)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(record: dict[str, Any], path: str | Path) -> None:
    out_path = ensure_parent(path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_jsonl_map(path: str | Path, key: str = "id") -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if key not in record:
            raise KeyError(f"Record in {path} is missing key {key!r}")
        record_key = str(record[key])
        if record_key in out:
            raise ValueError(f"Duplicate {key}={record_key!r} in {path}")
        out[record_key] = record
    return out
