from __future__ import annotations

import numpy as np

from src.eval.loader import Pair, Task
from src.eval.scorer import exact_match, score_submission
from src.solvers import EnsembleSolver, TransformSolver, majority_vote
from src.solvers.heuristics import (
    BASIC_TRANSFORMS,
    D8,
    anti_transpose,
    color_map_from_pair,
    complete_horizontal_symmetry,
    crop_to_nonzero_bbox,
    flip_h,
    flip_v,
    infer_consistent_color_map,
    invert_colors,
    largest_component_isolate,
    least_common_color,
    most_common_color,
    rot90,
    rot180,
    rot270,
    scale_up_2x,
    tile_2x2_self,
    transpose,
)


def _g(a):
    return np.array(a, dtype=np.int8)


def test_d8_has_eight():
    assert len(D8) == 8


def test_basic_transforms_count():
    assert len(BASIC_TRANSFORMS) >= 20


def test_flip_and_rotate():
    g = _g([[1, 2], [3, 4]])
    assert np.array_equal(flip_h(g), _g([[2, 1], [4, 3]]))
    assert np.array_equal(flip_v(g), _g([[3, 4], [1, 2]]))
    assert np.array_equal(rot90(g), _g([[2, 4], [1, 3]]))
    assert np.array_equal(rot180(g), _g([[4, 3], [2, 1]]))
    assert np.array_equal(rot270(g), _g([[3, 1], [4, 2]]))
    assert np.array_equal(transpose(g), _g([[1, 3], [2, 4]]))
    assert np.array_equal(anti_transpose(g), _g([[4, 2], [3, 1]]))


def test_most_common_color():
    assert most_common_color(_g([[0, 0, 1, 2], [0, 3, 0, 0]])) == 0


def test_least_common_color_is_deterministic():
    # least-common among 1,2,3 all tied=1 → argmin picks smallest
    assert least_common_color(_g([[1, 2, 3]])) == 1


def test_color_map():
    src = _g([[1, 2], [2, 1]])
    dst = _g([[3, 4], [4, 3]])
    m = color_map_from_pair(src, dst)
    assert m == {1: 3, 2: 4}


def test_color_map_inconsistent_returns_none():
    src = _g([[1, 1]])
    dst = _g([[2, 3]])
    assert color_map_from_pair(src, dst) is None


def test_infer_consistent_color_map_across_pairs():
    m = infer_consistent_color_map([
        (_g([[1, 2]]), _g([[3, 4]])),
        (_g([[2, 1]]), _g([[4, 3]])),
    ])
    assert m == {1: 3, 2: 4}


def test_infer_consistent_color_map_inconsistent_across_pairs():
    m = infer_consistent_color_map([
        (_g([[1, 2]]), _g([[3, 4]])),
        (_g([[1, 2]]), _g([[5, 4]])),  # conflict on 1
    ])
    assert m is None


def test_invert_colors_swaps_mc_and_lc():
    g = _g([[1, 1, 1, 2]])  # mc=1, lc=2
    out = invert_colors(g)
    assert out.tolist() == [[2, 2, 2, 1]]


def test_crop_to_nonzero_bbox():
    g = _g([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert crop_to_nonzero_bbox(g).tolist() == [[5]]


def test_tile_2x2_self():
    g = _g([[1, 2]])
    assert tile_2x2_self(g).shape == (2, 4)


def test_scale_up_2x():
    g = _g([[1, 2]])
    out = scale_up_2x(g)
    assert out.shape == (2, 4)
    assert out.tolist() == [[1, 1, 2, 2], [1, 1, 2, 2]]


def test_complete_horizontal_symmetry():
    g = _g([[1, 0, 0, 0]])
    out = complete_horizontal_symmetry(g)
    assert out.tolist() == [[1, 0, 0, 1]]


def test_largest_component_isolate():
    g = _g([[1, 1, 0, 2], [1, 0, 0, 0]])  # 3-cell component of 1s, 1-cell of 2
    out = largest_component_isolate(g)
    assert (out == 1).sum() == 3
    assert (out == 2).sum() == 0


def test_solver_picks_flip_h_when_consistent():
    task = Task(
        task_id="t1",
        train=(
            Pair(_g([[1, 2, 3]]), _g([[3, 2, 1]])),
            Pair(_g([[4, 5]]), _g([[5, 4]])),
        ),
        test=(Pair(_g([[7, 8, 9]]), _g([[9, 8, 7]])),),
    )
    pred = TransformSolver().solve(task)
    assert pred.candidates[0][0].tolist() == [[9, 8, 7]]


def test_scorer_second_attempt_matches():
    """2026 rule: submission has attempt_1 and attempt_2 (N_ATTEMPTS=2)."""
    task = Task(
        task_id="t",
        train=(Pair(_g([[1]]), _g([[1]])),),
        test=(Pair(_g([[1]]), _g([[1]])),),
    )
    preds = {"t": [[_g([[0]]), _g([[1]])]]}  # attempt_2 matches
    score = score_submission([task], preds)
    assert score.solved == 1


def test_scorer_neither_attempt_matches():
    task = Task(
        task_id="t",
        train=(Pair(_g([[1]]), _g([[1]])),),
        test=(Pair(_g([[1]]), _g([[1]])),),
    )
    # 3rd candidate would match but N_ATTEMPTS=2 ignores it
    preds = {"t": [[_g([[0]]), _g([[2]]), _g([[1]])]]}
    score = score_submission([task], preds)
    assert score.solved == 0


def test_exact_match_shape_mismatch():
    assert not exact_match(_g([[1]]), _g([[1, 1]]))


def test_ensemble_majority_vote_picks_most_common():
    # solver 1 says [[1]], solver 2 says [[1]], solver 3 says [[2]]
    # rank-weighted vote: [[1]] gets 1.0+1.0=2.0, [[2]] gets 1.0
    result = majority_vote([
        [_g([[1]])],
        [_g([[1]])],
        [_g([[2]])],
    ], top_k=3)
    assert result[0].tolist() == [[1]]


def test_ensemble_solver_aggregates():
    task = Task(
        task_id="t",
        train=(Pair(_g([[1, 2]]), _g([[2, 1]])),),  # just flip_h
        test=(Pair(_g([[3, 4]]), _g([[4, 3]])),),
    )
    s1 = TransformSolver()
    ens = EnsembleSolver(inner=[s1, s1, s1])
    pred = ens.solve(task)
    # All inner solvers should return flip_h on test → voted top is flip_h result
    assert pred.candidates[0][0].tolist() == [[4, 3]]
