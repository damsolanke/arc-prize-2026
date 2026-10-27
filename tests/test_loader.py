from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.eval.loader import EVAL_DIR, load_split, load_task


def _make_task_json(tmp: Path, task_id: str, train, test) -> Path:
    path = tmp / f"{task_id}.json"
    path.write_text(
        json.dumps(
            {
                "train": [{"input": p[0], "output": p[1]} for p in train],
                "test": [{"input": p[0], "output": p[1]} for p in test],
            }
        )
    )
    return path


def test_load_task_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = _make_task_json(
            Path(d),
            "abc",
            train=[([[0, 1], [1, 0]], [[1, 0], [0, 1]])],
            test=[([[0, 0], [0, 0]], [[1, 1], [1, 1]])],
        )
        t = load_task(path)
    assert t.task_id == "abc"
    assert len(t.train) == 1
    assert len(t.test) == 1
    assert t.train[0].input.shape == (2, 2)
    assert t.train[0].input.dtype == np.int8


def test_reject_out_of_range_colors():
    with tempfile.TemporaryDirectory() as d:
        path = _make_task_json(
            Path(d),
            "bad",
            train=[([[0, 11]], [[0, 0]])],  # 11 is out of range
            test=[([[0, 0]], [[0, 0]])],
        )
        with pytest.raises(ValueError):
            load_task(path)


def test_public_eval_loads_120():
    if not EVAL_DIR.exists():
        pytest.skip("arc-agi-v2 not cloned into reference/")
    tasks = list(load_split(EVAL_DIR))
    assert len(tasks) == 120
    for t in tasks:
        assert len(t.train) >= 1
        assert len(t.test) >= 1
