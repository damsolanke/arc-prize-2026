"""Tests for src/solvers/object_heuristics.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.solvers.object_heuristics import (
    OBJECT_HEURISTICS,
    crop_to_largest_component,
    crop_to_smallest_component,
    fill_holes,
    gravity_down,
    gravity_left,
    gravity_right,
    gravity_up,
    largest_component_color_fill,
    outline_objects,
    smallest_component_color_fill,
)


def _g(a):
    return np.array(a, dtype=np.int8)


def test_gravity_down_stacks_bottom():
    g = _g([
        [0, 1, 0],
        [2, 0, 3],
        [0, 0, 0],
    ])
    out = gravity_down(g)
    assert out.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [2, 1, 3],
    ]


def test_gravity_up_stacks_top():
    g = _g([
        [0, 0, 0],
        [0, 0, 0],
        [2, 1, 3],
    ])
    out = gravity_up(g)
    assert out.tolist() == [
        [2, 1, 3],
        [0, 0, 0],
        [0, 0, 0],
    ]


def test_gravity_left_stacks_left():
    g = _g([[0, 1, 0, 2]])
    assert gravity_left(g).tolist() == [[1, 2, 0, 0]]


def test_gravity_right_stacks_right():
    g = _g([[1, 0, 2, 0]])
    assert gravity_right(g).tolist() == [[0, 0, 1, 2]]


def test_object_heuristics_library_size():
    assert len(OBJECT_HEURISTICS) == 11


def test_fill_holes_fills_enclosed_zero():
    pytest.importorskip("scipy")
    g = _g([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ])
    out = fill_holes(g)
    assert out.tolist() == [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]


def test_crop_to_largest_component():
    pytest.importorskip("scipy")
    # one 2-cell object on left, one 1-cell object on right
    g = _g([
        [2, 2, 0, 0, 3],
        [0, 0, 0, 0, 0],
    ])
    out = crop_to_largest_component(g)
    assert out.shape == (1, 2)
    assert (out == 2).all()


def test_crop_to_smallest_component():
    pytest.importorskip("scipy")
    g = _g([
        [2, 2, 0, 0, 3],
        [0, 0, 0, 0, 0],
    ])
    out = crop_to_smallest_component(g)
    assert out.shape == (1, 1)
    assert (out == 3).all()


def test_largest_component_color_fill_keeps_only_largest():
    pytest.importorskip("scipy")
    g = _g([
        [2, 2, 0, 3],
        [0, 0, 0, 0],
    ])
    out = largest_component_color_fill(g)
    assert (out == 2).sum() == 2
    assert (out == 3).sum() == 0


def test_smallest_component_color_fill():
    pytest.importorskip("scipy")
    g = _g([
        [2, 2, 0, 3],
        [0, 0, 0, 0],
    ])
    out = smallest_component_color_fill(g)
    assert (out == 3).sum() == 1
    assert (out == 2).sum() == 0


def test_outline_objects_preserves_edges():
    g = _g([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])
    out = outline_objects(g)
    # interior (2,2) has no bg neighbor → cleared to 0
    assert out[2, 2] == 0
    # edges (1,1), (1,3), (3,1), (3,3) remain
    assert out[1, 1] == 1
