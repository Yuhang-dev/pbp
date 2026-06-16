from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CoupledFFNUnitGroup:
    layer: int
    module_name: str
    intermediate_size: int
    gate_shape: tuple[int, ...]
    up_shape: tuple[int, ...]
    down_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gate_shape"] = list(self.gate_shape)
        data["up_shape"] = list(self.up_shape)
        data["down_shape"] = list(self.down_shape)
        return data


def _shape_of_weight(module: Any) -> tuple[int, ...]:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def _linear_features(module: Any) -> tuple[int | None, int | None]:
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    if out_features is not None and in_features is not None:
        return int(out_features), int(in_features)

    shape = _shape_of_weight(module)
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    return None, None


def _layer_index_from_name(name: str, fallback: int) -> int:
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", name)
    if match:
        return int(match.group(1))
    return fallback


def discover_coupled_ffn_unit_groups(model: Any) -> list[CoupledFFNUnitGroup]:
    """Find modules with Qwen-style gate_proj/up_proj/down_proj coupled FFN units."""
    if not hasattr(model, "named_modules"):
        raise TypeError("model must expose named_modules()")

    groups: list[CoupledFFNUnitGroup] = []
    fallback_layer = 0
    for module_name, module in model.named_modules():
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            continue

        gate_out, _ = _linear_features(module.gate_proj)
        up_out, _ = _linear_features(module.up_proj)
        _, down_in = _linear_features(module.down_proj)
        if gate_out is None or up_out is None or down_in is None:
            continue
        if not (gate_out == up_out == down_in):
            raise ValueError(
                f"Invalid coupled FFN dimensions at {module_name}: "
                f"gate_out={gate_out}, up_out={up_out}, down_in={down_in}"
            )

        groups.append(
            CoupledFFNUnitGroup(
                layer=_layer_index_from_name(module_name, fallback_layer),
                module_name=module_name,
                intermediate_size=int(gate_out),
                gate_shape=_shape_of_weight(module.gate_proj),
                up_shape=_shape_of_weight(module.up_proj),
                down_shape=_shape_of_weight(module.down_proj),
            )
        )
        fallback_layer += 1

    if not groups:
        raise ValueError("No coupled FFN unit groups found")
    return groups


def expand_unit_metadata(groups: list[CoupledFFNUnitGroup]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in groups:
        for unit_index in range(group.intermediate_size):
            records.append(
                {
                    "layer": group.layer,
                    "module_name": group.module_name,
                    "unit_index": unit_index,
                    "gate_shape": list(group.gate_shape),
                    "up_shape": list(group.up_shape),
                    "down_shape": list(group.down_shape),
                }
            )
    return records
