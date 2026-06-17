from __future__ import annotations

import csv
import json
from argparse import Namespace

from scripts.summarize_m10a_matched_utility import build_rows


def test_m10a_summary_joins_general_utility_with_bcr_table(tmp_path):
    bcr_table = tmp_path / "m9.csv"
    with bcr_table.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "method",
                "ratio",
                "coverage@0",
                "coverage@q25",
                "bcr@0",
                "bcr@q25",
                "pref_acc",
                "mean_margin_drop",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "method": "activation",
                "ratio": "0.2",
                "coverage@0": "0.533",
                "coverage@q25": "0.399",
                "bcr@0": "0.35",
                "bcr@q25": "0.31",
                "pref_acc": "0.52",
                "mean_margin_drop": "0.01",
            }
        )
        writer.writerow(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "method": "boundary_taylor_weighted",
                "ratio": "0.2",
                "coverage@0": "0.533",
                "coverage@q25": "0.399",
                "bcr@0": "0.32",
                "bcr@q25": "0.25",
                "pref_acc": "0.51",
                "mean_margin_drop": "0.015",
            }
        )

    dense = tmp_path / "general_dense.json"
    activation = tmp_path / "general_activation.json"
    boundary = tmp_path / "general_boundary.json"
    dense.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "method": "dense",
                "ratio": 0.0,
                "ppl": 10.0,
                "arc_c": 0.4,
                "hellaswag": 0.5,
            }
        ),
        encoding="utf-8",
    )
    activation.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "method": "activation",
                "ratio": 0.2,
                "ppl": 10.5,
                "arc_c": 0.38,
                "hellaswag": 0.48,
            }
        ),
        encoding="utf-8",
    )
    boundary.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "method": "boundary_taylor_weighted",
                "ratio": 0.2,
                "ppl": 10.4,
                "arc_c": 0.39,
                "hellaswag": 0.49,
            }
        ),
        encoding="utf-8",
    )

    args = Namespace(
        general_inputs=[str(dense), str(activation), str(boundary)],
        bcr_table=str(bcr_table),
        max_ppl_relative_delta=0.10,
        max_accuracy_drop=0.05,
    )

    rows, summary = build_rows(args)

    assert [row["method"] for row in rows] == ["dense", "activation", "boundary_taylor_weighted"]
    assert rows[0]["bcr@q25"] == "0"
    assert rows[0]["pref_acc"] == "0.533"
    assert rows[1]["matched_utility_flag"] == "true"
    assert rows[2]["bcr@q25"] == "0.25"
    assert summary["boundary_vs_activation"]["boundary_lower_bcr_q25_than_activation"] is True
