from __future__ import annotations

import random
from os.path import commonprefix
from typing import Any

ASSISTANT_MARKER = "\n\nAssistant:"


def split_hh_rlhf_pair(chosen: str, rejected: str) -> tuple[str, str, str]:
    """Split HH-RLHF full transcripts into prompt, chosen response, rejected response."""
    if not chosen or not rejected:
        raise ValueError("chosen and rejected must be non-empty strings")

    chosen_idx = chosen.rfind(ASSISTANT_MARKER)
    rejected_idx = rejected.rfind(ASSISTANT_MARKER)
    if chosen_idx >= 0 and rejected_idx >= 0:
        chosen_prompt = chosen[: chosen_idx + len(ASSISTANT_MARKER)]
        rejected_prompt = rejected[: rejected_idx + len(ASSISTANT_MARKER)]
        if chosen_prompt == rejected_prompt:
            chosen_response = chosen[len(chosen_prompt) :]
            rejected_response = rejected[len(rejected_prompt) :]
            if chosen_response and rejected_response:
                return chosen_prompt, chosen_response, rejected_response

    prefix = commonprefix([chosen, rejected])
    marker_idx = prefix.rfind(ASSISTANT_MARKER)
    if marker_idx >= 0:
        prompt = prefix[: marker_idx + len(ASSISTANT_MARKER)]
    else:
        prompt = prefix

    if not prompt:
        raise ValueError("Could not infer a shared prompt from chosen/rejected texts")

    chosen_response = chosen[len(prompt) :]
    rejected_response = rejected[len(prompt) :]
    if not chosen_response or not rejected_response:
        raise ValueError("Could not infer non-empty chosen/rejected responses")
    return prompt, chosen_response, rejected_response


def hh_rlhf_record_to_preference(
    record: dict[str, Any],
    index: int,
    *,
    source: str = "hh-rlhf",
    id_prefix: str = "hh-rlhf",
) -> dict[str, str]:
    if "chosen" not in record or "rejected" not in record:
        raise KeyError("HH-RLHF record must contain 'chosen' and 'rejected'")
    prompt, chosen, rejected = split_hh_rlhf_pair(str(record["chosen"]), str(record["rejected"]))
    return {
        "id": f"{id_prefix}-{index:08d}",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source,
    }


def prepare_preference_records(
    raw_records: list[dict[str, Any]],
    *,
    calib_size: int,
    eval_size: int,
    seed: int,
    source: str = "hh-rlhf",
    id_prefix: str = "hh-rlhf",
    skip_bad_records: bool = False,
    max_samples: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int]:
    if calib_size < 0:
        raise ValueError("calib_size must be >= 0")
    if eval_size <= 0:
        raise ValueError("eval_size must be > 0")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided")

    total_needed = calib_size + eval_size
    if max_samples is not None:
        total_needed = min(total_needed, max_samples)
    if total_needed <= calib_size:
        raise ValueError("max_samples leaves no evaluation records")

    shuffled = list(raw_records)
    random.Random(seed).shuffle(shuffled)

    records: list[dict[str, str]] = []
    skipped = 0
    for raw_index, row in enumerate(shuffled):
        try:
            record = hh_rlhf_record_to_preference(
                row,
                raw_index,
                source=source,
                id_prefix=id_prefix,
            )
        except Exception:
            if not skip_bad_records:
                raise
            skipped += 1
            continue
        records.append(record)
        if len(records) >= total_needed:
            break

    if len(records) < total_needed:
        raise ValueError(f"Only prepared {len(records)} records, needed {total_needed}")

    calib_records = records[:calib_size]
    eval_records = records[calib_size:total_needed]
    validate_disjoint_splits(calib_records, eval_records)
    for record in calib_records + eval_records:
        validate_preference_record(record)
    return calib_records, eval_records, skipped


def validate_disjoint_splits(
    calib_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> None:
    calib_ids = {str(record["id"]) for record in calib_records}
    eval_ids = {str(record["id"]) for record in eval_records}
    overlap = calib_ids.intersection(eval_ids)
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"Calibration/evaluation splits overlap: {sample}")


def validate_preference_record(record: dict[str, Any]) -> None:
    required = {"id", "prompt", "chosen", "rejected", "source"}
    missing = required.difference(record)
    if missing:
        raise KeyError(f"Preference record missing required fields: {sorted(missing)}")
    for key in required:
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"Preference record field {key!r} must be a non-empty string")
