"""Lookup and auto-detection for MotionLayouts.

Callers can name a layout explicitly (`layout="mediapipe_33"`) or pass
`layout=None` and let `detect_layout` guess from the point count — useful
when a caller only knows "I have 33 keypoints" and not which convention
produced them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from src.gpsm.smplx.layouts.builtin import ALL_BUILTIN_LAYOUTS
from src.gpsm.smplx.layouts.correspondence import MotionLayout

_REGISTRY: Dict[str, MotionLayout] = {}


def register_layout(layout: MotionLayout, overwrite: bool = False) -> None:
    if not overwrite and layout.name in _REGISTRY:
        raise ValueError(f"Layout {layout.name!r} is already registered")
    _REGISTRY[layout.name] = layout


def get_layout(name: str) -> MotionLayout:
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown layout {name!r}. Registered layouts: {sorted(_REGISTRY)}"
        ) from e


def list_layouts() -> List[str]:
    return sorted(_REGISTRY)


def detect_layout(num_points: int) -> MotionLayout:
    """Guess a layout purely from point count. Ambiguous counts raise."""
    matches = [l for l in _REGISTRY.values() if l.num_points == num_points]
    if not matches:
        raise ValueError(
            f"No registered layout has {num_points} points. "
            f"Registered layouts: {[(l.name, l.num_points) for l in _REGISTRY.values()]}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"{num_points} points is ambiguous between layouts "
            f"{[l.name for l in matches]}; pass `layout=` explicitly."
        )
    return matches[0]


for _layout in ALL_BUILTIN_LAYOUTS:
    register_layout(_layout)
