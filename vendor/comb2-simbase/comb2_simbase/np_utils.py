from __future__ import annotations

import numpy as np


def search_sorted_idx(arr, target):
    values = np.asarray(arr)
    idx = int(np.searchsorted(values, target, side="left"))
    if idx < len(values) and values[idx] == target:
        return idx
    return None


def search_sorted_side_idx(arr, target, side: str = "left") -> int:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    return int(np.searchsorted(np.asarray(arr), target, side=side))


def search_sorted_left_idx(arr, target) -> int:
    values = np.asarray(arr)
    if len(values) == 0:
        raise ValueError("cannot search an empty array")
    idx = int(np.searchsorted(values, target, side="right") - 1)
    return max(0, min(idx, len(values) - 1))


def search_sorted_right_idx(arr, target) -> int:
    values = np.asarray(arr)
    if len(values) == 0:
        raise ValueError("cannot search an empty array")
    idx = int(np.searchsorted(values, target, side="left"))
    return max(0, min(idx, len(values) - 1))
