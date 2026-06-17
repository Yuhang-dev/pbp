from __future__ import annotations

import json
from argparse import Namespace

from scripts.report_mask_distribution import rows_for_mask_dir
from scripts.summarize_m10b_matched_utility import build_rows


def test_mask_distribution_reports_layer_pruning_ratios(tmp_path):
    mask_dir = tmp_path / "mask"
    mask_dir.mkdir()
    (mask_dir / "mask_config.json").write_text(
        json.dumps(
            {
                "method": "activation",
                "actual_ratio": 0.25,
                "groups": [
                    {"module_name": "model.layers.0.mlp", "layer": 0},
                    {"module_name": "model.layers.1.mlp", "layer": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    (mask_dir / "masks.json").write_text(
        json.dumps(
            {
                "model.layers.0.mlp": [1, 0, 1, 0],
                "model.layers.1.mlp": [1, 1, 1, 0],
            }
        ),
        encoding="utf-8",
    )

    rows = rows_for_mask_dir(mask_dir)

    assert rows == [
            {
                "method": "activation",
                "selection_scope": "global",
                "protection": "none",
                "ratio": "0.25",
                "requested_ratio": "0.25",
                "actual_global_ratio": "0.25",
                "actual_unprotected_ratio": "0.25",
                "layer": "0",
                "total_units": "4",
                "pruned_units": "2",
            "pruned_ratio": "0.5",
        },
            {
                "method": "activation",
                "selection_scope": "global",
                "protection": "none",
                "ratio": "0.25",
                "requested_ratio": "0.25",
                "actual_global_ratio": "0.25",
                "actual_unprotected_ratio": "0.25",
                "layer": "1",
                "total_units": "4",
                "pruned_units": "1",
            "pruned_ratio": "0.25",
        },
    ]


def test_m10b_summary_answers_matched_utility_questions(tmp_path):
    bcr = tmp_path / "bcr.csv"
    bcr.write_text(
        "\n".join(
            [
                "model,method,ratio,coverage@0,coverage@q25,bcr@0,bcr@q25,pref_acc,mean_margin_drop",
                "model,activation,0.1,0.5,0.4,0.1,0.05,0.6,0.01",
                "model,boundary_taylor_weighted,0.1,0.5,0.4,0.2,0.03,0.6,0.02",
                "model,activation,0.2,0.5,0.4,0.3,0.20,0.5,0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payloads = {
        "dense": {"model": "model", "method": "dense", "ratio": 0.0, "ppl": 10.0, "arc_c": 0.5, "hellaswag": 0.5},
        "activation_10p": {
            "model": "model",
            "method": "activation",
            "ratio": 0.1,
            "ppl": 10.5,
            "arc_c": 0.48,
            "hellaswag": 0.48,
        },
        "boundary_10p": {
            "model": "model",
            "method": "boundary_taylor_weighted",
            "ratio": 0.1,
            "ppl": 10.6,
            "arc_c": 0.49,
            "hellaswag": 0.49,
        },
        "activation_20p": {
            "model": "model",
            "method": "activation",
            "ratio": 0.2,
            "ppl": 20.0,
            "arc_c": 0.30,
            "hellaswag": 0.35,
        },
    }
    general_inputs = []
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        general_inputs.append(str(path))

    args = Namespace(
        general_inputs=general_inputs,
        bcr_table=str(bcr),
        max_ppl_relative_delta=0.10,
        max_accuracy_drop=0.05,
    )

    rows, summary = build_rows(args)

    assert len(rows) == 4
    assert summary["answers"]["is_any_10p_pruned_model_matched_utility"] is True
    assert summary["answers"]["is_any_20p_pruned_model_matched_utility"] is False
    assert summary["answers"]["lowest_bcr_q25_among_matched_utility_models"]["method"] == "boundary_taylor_weighted"
    assert summary["answers"]["twenty_percent_mild_regime_assessment"] == (
        "20% is not a mild regime under current masking."
    )
