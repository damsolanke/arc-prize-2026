"""Tests for augmentation utilities (src/ttt/augmentation.py)."""
from __future__ import annotations

import numpy as np

from src.ttt.augmentation import (
    Aug,
    color_permute,
    color_permute_invert,
    d8_apply,
    d8_invert,
    grid_to_text,
    sample_augs,
    text_to_grid,
)


def test_d8_round_trip():
    g = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    for tid in range(8):
        back = d8_invert(d8_apply(g, tid), tid)
        assert np.array_equal(back, g), f"D8 invert failed at tid={tid}"


def test_color_permute_round_trip():
    g = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int8)
    perm = np.arange(10)
    np.random.default_rng(42).shuffle(perm)
    back = color_permute_invert(color_permute(g, perm), perm)
    assert np.array_equal(back, g)


def test_aug_combined_round_trip():
    g = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        aug = sample_augs(1, rng)[0]
        back = aug.invert(aug.apply(g))
        assert np.array_equal(back, g), f"round-trip failed for tid={aug.tid}"


def test_grid_text_round_trip():
    g = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    back = text_to_grid(grid_to_text(g))
    assert back is not None
    assert np.array_equal(back, g)


def test_text_to_grid_rejects_malformed():
    assert text_to_grid("123\n12") is None  # ragged
    assert text_to_grid("abc") is None  # non-digit
    assert text_to_grid("") is None


def test_aug_is_hashable():
    a = Aug(tid=0, perm=tuple(range(10)))
    assert hash(a) == hash(Aug(tid=0, perm=tuple(range(10))))
