from __future__ import annotations

import pytest

from pbp.ffn_units import CoupledFFNUnitGroup
from pbp.scoring import (
    UnitScore,
    random_scores,
    score_stats,
    scores_by_module,
    select_lowest_score_mask_plan,
)


def group(module_name: str, layer: int, intermediate_size: int) -> CoupledFFNUnitGroup:
    return CoupledFFNUnitGroup(
        layer=layer,
        module_name=module_name,
        intermediate_size=intermediate_size,
        gate_shape=(intermediate_size, 4),
        up_shape=(intermediate_size, 4),
        down_shape=(4, intermediate_size),
    )


def test_random_scores_are_deterministic_and_cover_all_units():
    groups = [group("model.layers.0.mlp", 0, 3), group("model.layers.1.mlp", 1, 2)]

    first = random_scores(groups, seed=42)
    second = random_scores(groups, seed=42)

    assert first == second
    assert len(first) == 5
    assert {(item.module_name, item.unit_index) for item in first} == {
        ("model.layers.0.mlp", 0),
        ("model.layers.0.mlp", 1),
        ("model.layers.0.mlp", 2),
        ("model.layers.1.mlp", 0),
        ("model.layers.1.mlp", 1),
    }


def test_select_lowest_score_mask_plan_prunes_exact_global_ratio():
    groups = [group("model.layers.0.mlp", 0, 3), group("model.layers.1.mlp", 1, 3)]
    scores = [
        UnitScore(0, "model.layers.0.mlp", 0, 0.6),
        UnitScore(0, "model.layers.0.mlp", 1, 0.1),
        UnitScore(0, "model.layers.0.mlp", 2, 0.5),
        UnitScore(1, "model.layers.1.mlp", 0, 0.2),
        UnitScore(1, "model.layers.1.mlp", 1, 0.4),
        UnitScore(1, "model.layers.1.mlp", 2, 0.3),
    ]

    mask_plan = select_lowest_score_mask_plan(groups, scores, ratio=1 / 3, method="magnitude", seed=7)

    assert mask_plan["num_pruned_units"] == 2
    assert mask_plan["actual_ratio"] == pytest.approx(1 / 3)
    assert mask_plan["masks_by_module"]["model.layers.0.mlp"] == [1, 0, 1]
    assert mask_plan["masks_by_module"]["model.layers.1.mlp"] == [0, 1, 1]


def test_score_stats_and_scores_by_module_are_finite():
    scores = [
        UnitScore(0, "model.layers.0.mlp", 0, 1.0),
        UnitScore(0, "model.layers.0.mlp", 1, 3.0),
    ]

    stats = score_stats(scores)
    grouped = scores_by_module(scores)

    assert stats["num_scores"] == 2
    assert stats["scores_finite"] is True
    assert stats["mean_score"] == pytest.approx(2.0)
    assert grouped == {"model.layers.0.mlp": [1.0, 3.0]}
