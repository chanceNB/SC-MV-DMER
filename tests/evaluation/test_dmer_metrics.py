import importlib.util

import numpy as np
import pytest


def evaluate(records):
    assert importlib.util.find_spec("sc_mv_dmer.evaluation") is not None, "DMER evaluator is not implemented"
    from sc_mv_dmer.evaluation.dmer_metrics import evaluate_songs
    return evaluate_songs(records)


def record(song, p, y, mask=None):
    p, y = np.array(p, dtype=float), np.array(y, dtype=float)
    return {"song_id": song, "prediction": p, "target": y, "mask": np.ones_like(y, dtype=bool) if mask is None else np.array(mask, dtype=bool)}


def test_macro_is_song_equal_weight_and_pooled_is_separate():
    r = evaluate([
        record("short", [[0, 0], [1, 1]], [[0, 0], [1, 1]]),
        record("long", [[0, 0]] * 4, [[-1, -1], [1, 1], [-1, -1], [1, 1]]),
    ])
    assert r["song_count"] == 2 and r["valid_point_counts"] == {"valence": 6, "arousal": 6}
    assert r["macro"]["valence"]["ccc"] == pytest.approx(0.5)
    assert r["macro"]["arousal"]["rmse"] == pytest.approx(0.5)
    assert r["pooled"]["valence"]["rmse"] == pytest.approx(np.sqrt(4 / 6))
    assert r["selection_score"] == pytest.approx(0.5)
    assert any(d["kind"] == "constant_prediction" for d in r["degenerate_diagnostics"])


def test_dimension_masks_preserve_independent_tails():
    r = evaluate([record("s", [[0, 0], [1, 1], [999, 2]], [[0, 0], [1, 1], [np.nan, 2]], [[True, True], [True, True], [False, True]])])
    assert r["valid_point_counts"] == {"valence": 2, "arousal": 3}
    assert r["selection_score"] == pytest.approx(1)


def test_nonfinite_valid_prediction_fails_instead_of_dropping_point():
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate([record("s", [[np.nan, 0], [1, 1]], [[0, 0], [1, 1]])])


def test_zero_valid_song_dimension_makes_macro_invalid():
    r = evaluate([record("missing", [[0, 0], [1, 1]], [[0, 0], [1, 1]], [[False, True], [False, True]]), record("present", [[0, 0], [1, 1]], [[0, 0], [1, 1]])])
    assert r["song_count"] == 2
    assert r["status"] == "INVALID"
    assert r["macro"]["valence"]["ccc"] is None
    assert r["selection_score"] is None
    assert r["invalid_song_dimensions"] == [{"song_id": "missing", "dimension": "valence", "reason": "zero_valid_points"}]


def test_constant_series_behavior_is_explicit():
    r = evaluate([record("same", [[2, 3], [2, 3]], [[2, 1], [2, 1]])])
    assert r["macro"]["valence"] == {"ccc": 1.0, "pcc": 0.0, "rmse": 0.0}
    assert r["macro"]["arousal"] == {"ccc": 0.0, "pcc": 0.0, "rmse": 2.0}
    assert len(r["degenerate_diagnostics"]) >= 2


def test_duplicate_song_evaluation_fails():
    rec = record("s", [[0, 0], [1, 1]], [[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([rec, rec])


def test_fractional_constant_does_not_become_correlated_from_roundoff():
    r = evaluate([record("s", [[0.1, 0.1]] * 3, [[0.1, 0.2]] * 3)])
    assert r["macro"]["valence"]["pcc"] == 0
    assert r["macro"]["valence"]["ccc"] == 1
    assert r["macro"]["arousal"]["ccc"] == 0
