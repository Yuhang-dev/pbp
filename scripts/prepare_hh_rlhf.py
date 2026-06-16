from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.data import prepare_preference_records
from pbp.io import read_jsonl, write_jsonl
from pbp.logging_utils import RunLogger, finalize_run, initialize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF preference JSONL files.")
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--input-jsonl", default=None, help="Local raw HH-RLHF-style JSONL fixture/input.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--calib-size", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--skip-bad-records", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def load_raw_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl:
        return read_jsonl(args.input_jsonl)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to run this script: pip install datasets") from exc

    load_args = [args.dataset]
    if args.config:
        load_args.append(args.config)
    dataset = load_dataset(*load_args, split=args.split)
    return list(tqdm(dataset, desc="loading dataset records"))


def config_for_run(args: argparse.Namespace, output_path: str) -> dict[str, Any]:
    return {
        "script": "scripts/prepare_hh_rlhf.py",
        "model": None,
        "base_model": None,
        "dataset": args.dataset,
        "data_path": args.input_jsonl,
        "seed": args.seed,
        "dtype": None,
        "device": "cpu",
        "batch_size": None,
        "max_samples": args.max_samples,
        "output_path": output_path,
        "notes": "M1 HH-RLHF preprocessing",
        "split": args.split,
        "calib_size": args.calib_size,
        "eval_size": args.eval_size,
        "skip_bad_records": args.skip_bad_records,
    }


def assert_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    calib_out = out_dir / "hh_rlhf_calib.jsonl"
    eval_out = out_dir / "hh_rlhf_eval.jsonl"

    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, str(out_dir)),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write(calib_out, overwrite=args.overwrite)
        assert_can_write(eval_out, overwrite=args.overwrite)
        logger.stdout(f"Loading raw records from {args.input_jsonl or args.dataset}")
        raw_records = load_raw_records(args)
        logger.stdout(f"Loaded {len(raw_records)} raw records")

        calib_records, eval_records, skipped = prepare_preference_records(
            raw_records,
            calib_size=args.calib_size,
            eval_size=args.eval_size,
            seed=args.seed,
            source="hh-rlhf",
            id_prefix="hh-rlhf",
            skip_bad_records=args.skip_bad_records,
            max_samples=args.max_samples,
        )

        write_jsonl(calib_records, calib_out)
        write_jsonl(eval_records, eval_out)

        metrics = {
            "num_raw_records": len(raw_records),
            "num_calib_records": len(calib_records),
            "num_eval_records": len(eval_records),
            "num_total_records": len(calib_records) + len(eval_records),
            "num_skipped_records": skipped,
            "calib_eval_disjoint": True,
            "empty_chosen_or_rejected": 0,
        }
        finalize_run(run_paths, start_monotonic=start, metrics=metrics)

        message = {
            "run_id": run_paths.run_id,
            "calib_out": str(calib_out),
            "eval_out": str(eval_out),
            "metrics": metrics,
        }
        print(json.dumps(message, ensure_ascii=False, indent=2))
        logger.stdout(json.dumps(message, ensure_ascii=False))
    except Exception as exc:
        logger.stderr(f"{type(exc).__name__}: {exc}")
        finalize_run(run_paths, start_monotonic=start, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
