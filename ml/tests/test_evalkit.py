"""Metrics, intervals and splits on hand-built cases (no model, no GPU)."""

import math

import pytest

import evalkit
from common import load_labels, load_setups, video_codes, window_targets


def _v(file, verdict, events=None, duration=20.0):
    v = dict(file=file, duration_sec=duration, verdict=verdict)
    if events:
        v["events"] = events
        v["timecode_sec"] = events[0]["timecode_sec"]
    return v


def _e(code, t, alt=None):
    e = dict(type_code=code, timecode_sec=t, confidence="high", observation="x")
    if alt:
        e["alt_type_codes"] = alt
    return e


VIDEOS = [
    _v("a", "violation", [_e("DOOR_OPEN", 10)]),
    _v("b", "violation", [_e("UNATTENDED_CASH", 8, alt=["DOOR_OPEN"])]),
    _v("c", "violation", [_e("NO_COUNTER", 5), _e("PERSONAL_ITEMS", 40)], duration=60),
    _v("d", "no_violation"),
    _v("e", "no_violation"),
]


def test_video_metrics_baseline_definitions():
    pred = {
        "a": [("DOOR_OPEN", 11.0, 0.9)],  # correct type, 1 s error
        "b": [("DOOR_OPEN", 8.5, 0.6)],  # alternative type only
        "c": [],  # miss
        "d": [("SAFE_OPEN", 3.0, 0.7)],  # false positive
        "e": [],
    }
    m, rows = evalkit.video_metrics(VIDEOS, pred)
    assert m["n_pos"] == 3 and m["n_neg"] == 2
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["recall"] == pytest.approx(2 / 3) and m["fp_rate"] == pytest.approx(0.5)
    assert m["verdict_acc"] == pytest.approx(3 / 5)
    assert m["type_acc_on_pos"] == pytest.approx(1 / 3)  # only "a" has the primary type right
    assert m["type_acc_acceptable"] == pytest.approx(2 / 3)  # "b" counts with alternative types
    assert m["tc_n"] == 1 and m["tc_mae"] == pytest.approx(1.0) and m["tc_within_2s"] == 1.0
    assert m["multi_event_videos"] == 1 and m["multi_event_full_recall"] == 0.0
    assert rows[0]["type_ok"] is True and rows[1]["type_ok"] is False


def test_event_matching_tolerance_and_multi_event():
    pred = {"c": [("NO_COUNTER", 6.0, 0.8), ("PERSONAL_ITEMS", 47.0, 0.7)]}
    m, _ = evalkit.video_metrics([VIDEOS[2]], pred, tols=(1, 2, 10))
    assert m["event_recall_1s"] == pytest.approx(0.5)  # 1 s off counts, 7 s off does not
    assert m["event_recall_2s"] == pytest.approx(0.5)
    assert m["event_recall_10s"] == 1.0 and m["event_precision_10s"] == 1.0
    assert m["multi_event_full_recall"] == 1.0
    matches, up, ut = evalkit.match_events([("NO_COUNTER", 6.0, 0.8), ("NO_COUNTER", 5.5, 0.9)],
                                           VIDEOS[2]["events"], tol=2)
    assert len(matches) == 1 and matches[0][0] == 1  # higher score matched first, second is a duplicate
    assert up == [0] and ut == [1]


def test_wilson_and_bootstrap_contain_point_estimate():
    lo, hi = evalkit.wilson(16, 21)
    assert lo < 16 / 21 < hi and 0.54 < lo < 0.56 and 0.89 < hi < 0.90
    pred = {"a": [("DOOR_OPEN", 11.0, 0.9)], "b": [], "c": [], "d": [], "e": [("SAFE_OPEN", 1.0, 0.6)]}
    ci = evalkit.bootstrap_ci(VIDEOS, pred, n_boot=300, seed=1)
    m, _ = evalkit.video_metrics(VIDEOS, pred)
    for k in ("recall", "fp_rate", "verdict_acc"):
        assert ci[k][0] <= m[k] <= ci[k][1]


def test_paired_bootstrap_detects_strict_improvement():
    a = {"a": [], "b": [], "c": [], "d": [], "e": []}
    b = {"a": [("DOOR_OPEN", 10.0, 0.9)], "b": [("UNATTENDED_CASH", 8.0, 0.9)], "c": [("NO_COUNTER", 5.0, 0.9)],
         "d": [], "e": []}
    pb = evalkit.paired_bootstrap(VIDEOS, a, b, n_boot=200, seed=0)
    assert pb["recall"]["mean"] == pytest.approx(1.0) and pb["recall"]["p_b_better"] == 1.0
    assert evalkit.flips(VIDEOS, a, b) == {"fixed_by_b": ["a", "b", "c"], "broken_by_b": []}


def test_paired_bootstrap_fp_rate_lower_is_better():
    """Flagging every normal raises the FP rate: P(new better) for fp_rate must be 0, not 1."""
    a = {v["file"]: [] for v in VIDEOS}
    b = {v["file"]: ([] if evalkit.is_violation(v) else [("DOOR_OPEN", 1.0, 0.9)]) for v in VIDEOS}
    pb = evalkit.paired_bootstrap(VIDEOS, a, b, n_boot=200, seed=0)
    assert pb["fp_rate"]["mean"] > 0 and pb["fp_rate"]["p_b_better"] == 0.0


def test_score_metrics():
    y = [1, 1, 0, 0]
    assert evalkit.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert evalkit.roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert evalkit.average_precision(y, [0.9, 0.1, 0.8, 0.2]) == pytest.approx((1 + 2 / 4) / 2)  # hits at ranks 1 and 4
    assert evalkit.brier([1, 0], [1.0, 0.0]) == 0.0
    assert evalkit.ece([1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1], bins=5) == pytest.approx(0.1)
    assert math.isfinite(evalkit.log_loss([1, 0], [0.9, 0.2]))


def test_splits_are_disjoint_and_group_aware():
    videos = load_labels()
    setups = load_setups()
    assert len(videos) == 40 and all(v["file"] in setups for v in videos)
    setup_of = lambda v: setups[v["file"]]  # noqa: E731
    rep = evalkit.split_report(evalkit.lovo(videos))
    assert rep["folds"] == 40 and rep["mean_test"] == 1
    logo = list(evalkit.leave_one_group_out(videos, setup_of))
    assert len(logo) == 8
    for train, test in logo:
        assert len({setup_of(v) for v in test}) == 1
        assert not {setup_of(v) for v in train} & {setup_of(v) for v in test}
    rg = list(evalkit.repeated_group_kfold(videos, setup_of, k=4, repeats=3, seed=0))
    assert len(rg) == 12
    for i in range(3):
        covered = [v["file"] for _, test in rg[i * 4:(i + 1) * 4] for v in test]
        assert sorted(covered) == sorted(v["file"] for v in videos)
    train, test, unknown = evalkit.leave_one_type_out(videos, "DOOR_OPEN")
    assert unknown == {"Х-28.mov", "Х-32.mov", "Х-33.mov", "Х-34.mov", "Х-11.mov", "Х-22.mov", "Х-24.mov"}
    assert not any("DOOR_OPEN" in video_codes(v) for v in train) and len(test) == 40


def test_window_targets_alt_policy():
    v = _v("x", "violation", [_e("UNATTENDED_CASH", 8, alt=["DOOR_OPEN", "PERSONAL_ITEMS"])])
    assert window_targets(v, 0, 15, "UNATTENDED_CASH", False) == 1
    assert window_targets(v, 0, 15, "DOOR_OPEN", True) == 0  # first-release behaviour
    assert window_targets(v, 0, 15, "DOOR_OPEN", True, alt_policy="ignore") is None
    assert window_targets(v, 0, 15, "DOOR_OPEN", True, alt_policy="positive") == 1
    assert window_targets(v, 0, 15, "SAFE_OPEN", True, alt_policy="ignore") == 0  # unrelated type stays negative
    # windows after the timecode: positive for states, unknown for events
    assert window_targets(v, 10, 20, "UNATTENDED_CASH", False) is None
    assert window_targets(v, 10, 20, "DOOR_OPEN", True, alt_policy="positive") == 1
    assert window_targets(v, 0, 5, "UNATTENDED_CASH", False) == 0


def test_dataset_alt_type_videos_are_the_documented_six():
    files = sorted(v["file"] for v in load_labels() for e in v.get("events", []) if e.get("alt_type_codes"))
    assert files == ["X-2.mov", "Х-11.mov", "Х-22.mov", "Х-24.mov", "Х-28.mov", "Х-34.mov"]
