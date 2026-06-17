from __future__ import annotations

import csv
import json
from argparse import Namespace

from scripts.summarize_m11a_layerwise import build_rows


def test_m11a_summary_joins_general_bcr_and_mask_distribution(tmp_path):
    dense = tmp_path / "general_dense.json"
    pruned = tmp_path / "general_boundary_layerwise_2p.json"
    dense.write_text(
        json.dumps(
            {
                "model": "model",
                "method": "dense",
                "selection_scope": "dense",
                "protection": "none",
                "requested_ratio": 0.0,
                "actual_global_ratio": 0.0,
                "actual_unprotected_ratio": 0.0,
                "ppl": 10.0,
                "arc_c": 0.5,
                "hellaswag": 0.5,
            }
        ),
        encoding="utf-8",
    )
    pruned.write_text(
        json.dumps(
            {
                "model": "model",
                "method": "boundary_taylor_weighted",
                "selection_scope": "layerwise",
                "protection": "none",
                "requested_ratio": 0.02,
                "actual_global_ratio": 0.02,
                "actual_unprotected_ratio": 0.02,
                "ppl": 10.5,
                "arc_c": 0.48,
                "hellaswag": 0.49,
            }
        ),
        encoding="utf-8",
    )
    bcr = tmp_path / "bcr_boundary.json"
    bcr.write_text(
        json.dumps(
            {
                "mask_method": "boundary_taylor_weighted",
                "selection_scope": "layerwise",
                "protection": "none",
                "requested_ratio": 0.02,
                "bcr_at_q25": 0.03,
                "bcr_at_0": 0.05,
                "preference_accuracy_pruned": 0.6,
                "mean_margin_drop": 0.01,
            }
        ),
        encoding="utf-8",
    )
    mask_distribution = tmp_path / "mask_distribution.csv"
    with mask_distribution.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "selection_scope",
                "protection",
                "ratio",
                "requested_ratio",
                "actual_global_ratio",
                "actual_unprotected_ratio",
                "layer",
                "total_units",
                "pruned_units",
                "pruned_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "boundary_taylor_weighted",
                "selection_scope": "layerwise",
                "protection": "none",
                "ratio": "0.02",
                "requested_ratio": "0.02",
                "actual_global_ratio": "0.02",
                "actual_unprotected_ratio": "0.02",
                "layer": "0",
                "total_units": "100",
                "pruned_units": "2",
                "pruned_ratio": "0.02",
            }
        )
        writer.writerow(
            {
                "method": "boundary_taylor_weighted",
                "selection_scope": "layerwise",
                "protection": "none",
                "ratio": "0.02",
                "requested_ratio": "0.02",
                "actual_global_ratio": "0.02",
                "actual_unprotected_ratio": "0.02",
                "layer": "1",
                "total_units": "100",
                "pruned_units": "2",
                "pruned_ratio": "0.02",
            }
        )

    args = Namespace(
        general_inputs=[str(dense), str(pruned)],
        bcr_inputs=[str(bcr)],
        mask_distribution=str(mask_distribution),
        max_ppl_relative_delta=0.10,
        max_accuracy_drop=0.05,
        layerwise_ratio_tolerance=0.005,
    )

    rows, summary = build_rows(args)

    assert len(rows) == 2
    assert rows[1]["matched_utility_flag"] == "true"
    assert summary["answers"]["lowest_bcr_q25_among_matched_utility_settings"]["method"] == (
        "boundary_taylor_weighted"
    )
    assert summary["answers"]["layerwise_selection_avoids_early_layer_pruning_collapse"] is True
    assert summary["answers"]["mild_pruning_ratio_recommendation"]["recommended_mild_ratio"] == 0.02
