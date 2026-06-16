from __future__ import annotations

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
