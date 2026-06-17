from __future__ import annotations

import pytest

from pbp.ffn_units import CoupledFFNUnitGroup, discover_coupled_ffn_unit_groups
from pbp.pruning import apply_mask_plan_to_model, create_global_random_mask_plan, mask_plan_stats


class FakeWeight:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape


class FakeLinear:
    def __init__(self, out_features: int, in_features: int) -> None:
        self.weight = FakeWeight((out_features, in_features))


class FakeMLP:
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        self.gate_proj = FakeLinear(intermediate_size, hidden_size)
        self.up_proj = FakeLinear(intermediate_size, hidden_size)
        self.down_proj = FakeLinear(hidden_size, intermediate_size)


class FakeModel:
    def __init__(self) -> None:
        self.mlp0 = FakeMLP(hidden_size=4, intermediate_size=8)
        self.mlp1 = FakeMLP(hidden_size=4, intermediate_size=16)

    def named_modules(self):
        yield "", self
        yield "model.layers.0.mlp", self.mlp0
        yield "model.layers.1.mlp", self.mlp1


def group(module_name: str, intermediate_size: int, layer: int = 0) -> CoupledFFNUnitGroup:
    return CoupledFFNUnitGroup(
        layer=layer,
        module_name=module_name,
        intermediate_size=intermediate_size,
        gate_shape=(intermediate_size, 4),
        up_shape=(intermediate_size, 4),
        down_shape=(4, intermediate_size),
    )


def test_discovers_coupled_swiglu_unit_groups_from_weight_shapes():
    groups = discover_coupled_ffn_unit_groups(FakeModel())

    assert [item.module_name for item in groups] == ["model.layers.0.mlp", "model.layers.1.mlp"]
    assert [item.layer for item in groups] == [0, 1]
    assert [item.intermediate_size for item in groups] == [8, 16]
    assert groups[0].gate_shape == (8, 4)
    assert groups[0].up_shape == (8, 4)
    assert groups[0].down_shape == (4, 8)


def test_random_mask_plan_prunes_requested_global_unit_count_deterministically():
    groups = [group("model.layers.0.mlp", 8, 0), group("model.layers.1.mlp", 8, 1)]

    first = create_global_random_mask_plan(groups, ratio=0.25, seed=123)
    second = create_global_random_mask_plan(groups, ratio=0.25, seed=123)
    stats = mask_plan_stats(first)

    assert first == second
    assert first["total_units"] == 16
    assert first["num_pruned_units"] == 4
    assert first["actual_ratio"] == pytest.approx(0.25)
    assert stats["num_pruned_units"] == 4
    assert stats["num_kept_units"] == 12
    assert {len(mask) for mask in first["masks_by_module"].values()} == {8}
    assert sum(value == 0 for mask in first["masks_by_module"].values() for value in mask) == 4


def test_random_mask_plan_layerwise_protection_accounts_for_global_and_unprotected_ratio():
    groups = [group("model.layers.0.mlp", 8, 0), group("model.layers.1.mlp", 8, 1)]

    mask_plan = create_global_random_mask_plan(
        groups,
        ratio=0.25,
        seed=123,
        selection_scope="layerwise",
        protect_first_n_layers=1,
    )
    stats = mask_plan_stats(mask_plan)

    assert mask_plan["masks_by_module"]["model.layers.0.mlp"] == [1] * 8
    assert sum(value == 0 for value in mask_plan["masks_by_module"]["model.layers.1.mlp"]) == 2
    assert stats["actual_global_ratio"] == pytest.approx(0.125)
    assert stats["actual_unprotected_ratio"] == pytest.approx(0.25)
    assert stats["num_protected_layers"] == 1


def test_masked_swiglu_forward_zeroes_pruned_intermediate_units():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class ToyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(3, 3, bias=False)
            self.up_proj = nn.Linear(3, 3, bias=False)
            self.down_proj = nn.Linear(3, 1, bias=False)
            self.act_fn = nn.Identity()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = ToyMLP()

    model = ToyModel()
    with torch.no_grad():
        model.mlp.gate_proj.weight.copy_(torch.eye(3))
        model.mlp.up_proj.weight.copy_(torch.eye(3))
        model.mlp.down_proj.weight.copy_(torch.ones(1, 3))

    x = torch.tensor([[1.0, 2.0, 3.0]])
    before = model.mlp(x)
    stats = apply_mask_plan_to_model(model, {"masks_by_module": {"mlp": [1, 0, 1]}})
    after = model.mlp(x)

    assert before.item() == pytest.approx(14.0)
    assert after.item() == pytest.approx(10.0)
    assert stats["total_units"] == 3
    assert stats["num_pruned_units"] == 1
    assert hasattr(model.mlp, "_pbp_intermediate_mask")
