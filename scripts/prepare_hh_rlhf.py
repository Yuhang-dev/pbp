from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.data import hh_rlhf_record_to_preference
from pbp.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF preference JSONL files.")
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--calib-size", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--skip-bad-records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calib_size < 0 or args.eval_size <= 0:
        raise SystemExit("--calib-size must be >= 0 and --eval-size must be > 0")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets to run this script: pip install datasets") from exc

    load_args = [args.dataset]
    if args.config:
        load_args.append(args.config)
    dataset = load_dataset(*load_args, split=args.split)
    dataset = dataset.shuffle(seed=args.seed)

    total_needed = args.calib_size + args.eval_size
    records = []
    skipped = 0
    pbar = tqdm(total=total_needed, desc="preparing HH-RLHF")
    for raw_index, row in enumerate(dataset):
        try:
            record = hh_rlhf_record_to_preference(row, raw_index)
        except Exception:
            if not args.skip_bad_records:
                raise
            skipped += 1
            continue
        records.append(record)
        pbar.update(1)
        if len(records) >= total_needed:
            break
    pbar.close()

    if len(records) < total_needed:
        raise SystemExit(f"Only prepared {len(records)} records, needed {total_needed}")

    out_dir = Path(args.out_dir)
    calib_records = records[: args.calib_size]
    eval_records = records[args.calib_size :]
    calib_out = out_dir / "hh_rlhf_calib.jsonl"
    eval_out = out_dir / "hh_rlhf_eval.jsonl"
    write_jsonl(calib_records, calib_out)
    write_jsonl(eval_records, eval_out)
    print(f"Wrote {len(calib_records)} calibration records to {calib_out}")
    print(f"Wrote {len(eval_records)} evaluation records to {eval_out}")
    if skipped:
        print(f"Skipped {skipped} malformed records")


if __name__ == "__main__":
    main()
