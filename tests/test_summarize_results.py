from __future__ import annotations

import json

from scripts.summarize_results import row_from_result


def test_row_from_bcr_result_uses_pruned_preference_accuracy(tmp_path):
    result_path = tmp_path / "bcr_qwen2p5_1p5b_activation_10p_m9.json"
    result_path.write_text(
        json.dumps(
            {
                "model_load_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "mask_method": "activation",
                "mask_stats": {"actual_ratio": 0.1},
                "coverage_at_0": 0.6,
                "coverage_at_q25": 0.45,
                "bcr_at_0": 0.1,
                "bcr_at_q25": 0.05,
                "preference_accuracy_dense": 0.6,
                "preference_accuracy_pruned": 0.58,
                "mean_margin_drop": 0.03,
            }
        ),
        encoding="utf-8",
    )

    row = row_from_result(result_path)

    assert row == {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "method": "activation",
        "ratio": "0.1",
        "coverage@0": "0.6",
        "coverage@q25": "0.45",
        "bcr@0": "0.1",
        "bcr@q25": "0.05",
        "pref_acc": "0.58",
        "mean_margin_drop": "0.03",
    }
