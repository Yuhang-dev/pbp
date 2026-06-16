from __future__ import annotations

from typing import Any


def compute_preference_margin(
    ell_model_chosen: float,
    ell_model_rejected: float,
    ell_base_chosen: float,
    ell_base_rejected: float,
) -> float:
    return (ell_model_chosen - ell_base_chosen) - (ell_model_rejected - ell_base_rejected)


def length_normalized_from_logprob(record: dict[str, Any]) -> float:
    try:
        return float(record["length_normalized_logprob"])
    except KeyError as exc:
        raise KeyError("logprob record is missing length_normalized_logprob") from exc


def dense_margin_record(
    example_id: str,
    dense_chosen: dict[str, Any],
    dense_rejected: dict[str, Any],
    base_record: dict[str, Any],
    *,
    prompt_sha256: str | None = None,
) -> dict[str, Any]:
    ell_dense_chosen = length_normalized_from_logprob(dense_chosen)
    ell_dense_rejected = length_normalized_from_logprob(dense_rejected)
    ell_base_chosen = length_normalized_from_logprob(base_record["chosen"])
    ell_base_rejected = length_normalized_from_logprob(base_record["rejected"])
    delta_dense = compute_preference_margin(
        ell_dense_chosen,
        ell_dense_rejected,
        ell_base_chosen,
        ell_base_rejected,
    )
    record: dict[str, Any] = {
        "id": example_id,
        "ell_dense_chosen": ell_dense_chosen,
        "ell_dense_rejected": ell_dense_rejected,
        "ell_base_chosen": ell_base_chosen,
        "ell_base_rejected": ell_base_rejected,
        "delta_dense": delta_dense,
    }
    if prompt_sha256 is not None:
        record["prompt_sha256"] = prompt_sha256
    return record
