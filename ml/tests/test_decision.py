"""Decision policies on synthetic scores: baseline equivalence, Šidák length-invariance, calibration."""

import random

import numpy as np
import pytest

import decision
import evalkit
from app.vj.queries import BY_CODE, QUERIES

CODES = [q.code for q in QUERIES]


def _row(file, start, end, yes: dict[str, float], ground_peak=None):
    gate, ground = {}, {}
    for q in QUERIES:
        p = yes.get(q.code, 0.05)
        probs = [(1 - p) / (len(q.options) - 1)] * len(q.options)
        probs[q.yes_index] = p
        gate[q.code] = probs
        times = [start + 0.5 + i for i in range(int(end - start))]
        peak = ground_peak if ground_peak is not None else (start + end) / 2
        ground[q.code] = dict(times=times, probs=[float(np.exp(-abs(t - peak))) for t in times])
    return dict(file=file, aug="none", start=start, end=end, gate=gate, ground=ground)


def _video(file, verdict, code=None, t=None, duration=20.0):
    v = dict(file=file, duration_sec=duration, verdict=verdict)
    if code:
        v["events"] = [dict(type_code=code, timecode_sec=t, confidence="high", observation="x")]
        v["timecode_sec"] = t
    return v


def _synthetic(n_windows_normal=(1, 1, 1, 7, 7), seed=0):
    """3 violations with a clear peak, normals with iid noisy windows (long normals have 7 windows)."""
    rng = random.Random(seed)
    videos, by_video = [], {}
    for i, code in enumerate(["SAFE_OPEN", "DOOR_OPEN", "UNATTENDED_CASH"]):
        f = f"v{i}"
        videos.append(_video(f, "violation", code, 10, 20))
        by_video[f] = [_row(f, 0, 20, {code: 0.9}, ground_peak=10.5)]
    for j, n in enumerate(n_windows_normal):
        f = f"n{j}"
        videos.append(_video(f, "no_violation", duration=20 + 10 * (n - 1)))
        by_video[f] = [_row(f, 10 * w, 10 * w + 20, {c: rng.uniform(0.0, 0.6) for c in CODES}) for w in range(n)]
    return videos, by_video


def test_baseline_policy_matches_calibrate_predict():
    import calibrate

    videos, by_video = _synthetic()
    th = {"default": 0.5, "per_type": {"DOOR_OPEN": 0.8}, "max_events": 1}
    cal = decision.fit_calibration(videos, by_video)
    for v in videos:
        s_ref, e_ref = calibrate.predict(by_video[v["file"]], th)
        s_new, e_new = decision.predict(by_video[v["file"]], th, decision.BASELINE, cal)
        assert {c: s.score for c, s in s_ref.items()} == {c: s.score for c, s in s_new.items()}
        assert [(c, round(t, 3), s) for c, t, s in e_ref] == [(c, round(t, 3), s) for c, t, s in e_new]


def test_timecode_comes_from_grounding_peak():
    videos, by_video = _synthetic()
    cal = decision.fit_calibration(videos, by_video)
    _, events = decision.predict(by_video["v0"], {"default": 0.5, "per_type": {}, "max_events": 1},
                                 decision.BASELINE, cal)
    assert events[0][0] == "SAFE_OPEN" and abs(events[0][1] - 10.5) < 1e-6


def test_sidak_is_length_invariant_under_null():
    """Max over N iid windows exceeds a fixed threshold more often for large N; the Šidák-combined
    p-value does not. Simulated with uniform window scores, calibrated on a separate normal set."""
    rng = random.Random(1)
    train = [_video(f"t{i}", "no_violation") for i in range(30)]
    by_video = {v["file"]: [_row(v["file"], 0, 20, {c: rng.random() for c in CODES})] for v in train}
    cal = decision.fit_calibration(train, by_video)
    th = {"default": 0.9, "per_type": {}, "max_events": 1}

    def fp_rate(policy, n_windows, trials=400):
        flagged = 0
        for k in range(trials):
            rows = [_row("x", 10 * w, 10 * w + 20, {c: rng.random() for c in CODES}) for w in range(n_windows)]
            scores = decision.type_scores(rows, policy, cal)
            flagged += bool(decision.decide(scores, th))
        return flagged / trials

    fp_max_1, fp_max_8 = fp_rate(decision.Policy(agg="max"), 1), fp_rate(decision.Policy(agg="max"), 8)
    fp_z_1, fp_z_8 = fp_rate(decision.Policy(agg="zsidak"), 1), fp_rate(decision.Policy(agg="zsidak"), 8)
    fp_e_1, fp_e_8 = fp_rate(decision.Policy(agg="sidak"), 1), fp_rate(decision.Policy(agg="sidak"), 8)
    assert fp_max_8 > fp_max_1 + 0.2  # 8 windows x 8 types at threshold .9: almost always flagged
    assert abs(fp_z_8 - fp_z_1) < 0.15  # Gaussian-tail p-values combine over windows: no length effect
    assert fp_e_8 <= fp_e_1 + 0.05  # empirical p-values: no inflation either (but a resolution floor, see docs)


def test_bias_calibration_removes_always_yes_question():
    rng = random.Random(2)
    normals = [_video(f"n{i}", "no_violation") for i in range(20)]
    by_video = {v["file"]: [_row(v["file"], 0, 20, {"DOOR_OPEN": 0.9 + 0.05 * rng.random()})] for v in normals}
    cal = decision.fit_calibration(normals, by_video)
    assert cal.bias["DOOR_OPEN"] > 2  # logit(0.92) ~ 2.4
    raw = decision.type_scores(by_video["n0"], decision.Policy(), cal)["DOOR_OPEN"].score
    adj = decision.type_scores(by_video["n0"], decision.Policy(calib="bias"), cal)["DOOR_OPEN"].score
    assert raw > 0.9 and 0.2 < adj < 0.8


def test_fit_thresholds_objectives_and_shrink():
    videos, by_video = _synthetic()
    cal = decision.fit_calibration(videos, by_video)
    th = decision.fit_thresholds(videos, by_video, decision.BASELINE, cal, objective="f2_type")
    m, _ = decision.evaluate(videos, by_video, lambda v: th, decision.BASELINE, cal)
    assert m["recall"] == 1.0
    th_s = decision.fit_thresholds(videos, by_video, decision.BASELINE, cal, objective="recall_at_fp", alpha=0.0,
                                   shrink=0.1)
    assert all(abs(t - th_s["default"]) <= 0.1 + 1e-9 for t in th_s["per_type"].values())
    m_s, _ = decision.evaluate(videos, by_video, lambda v: th_s, decision.BASELINE, cal)
    assert m_s["fp_rate"] == 0.0
    th_g = decision.fit_thresholds(videos, by_video, decision.BASELINE, cal, per_type=False)
    assert th_g["per_type"] == {}


def test_nested_evaluate_uses_only_training_fold():
    videos, by_video = _synthetic()
    splits = list(evalkit.lovo(videos))
    pred, margins, used = decision.nested_evaluate(videos, by_video, decision.BASELINE, splits)
    assert set(pred) == {v["file"] for v in videos}
    assert all(f in margins for f in pred)
    m, _ = evalkit.video_metrics(videos, pred)
    assert m["recall"] == 1.0


def test_open_set_probe_runs():
    videos, by_video = _synthetic()
    out = decision.open_set_recall(videos, by_video, decision.BASELINE, "SAFE_OPEN")
    assert out["unknown_videos"] == 1 and 0.0 <= out["open_set_recall"] <= 1.0
    assert "SAFE_OPEN" not in out["flagged_as"]


def test_multi_event_outputs_two_types():
    videos, by_video = _synthetic()
    f = "v0"
    by_video[f] = [_row(f, 0, 20, {"SAFE_OPEN": 0.9, "DOOR_OPEN": 0.85})]
    cal = decision.fit_calibration(videos, by_video)
    th = {"default": 0.5, "per_type": {}, "max_events": 1}
    _, e1 = decision.predict(by_video[f], th, decision.Policy(max_events=1), cal)
    _, e2 = decision.predict(by_video[f], th, decision.Policy(max_events=2), cal)
    assert [e[0] for e in e1] == ["SAFE_OPEN"] and [e[0] for e in e2] == ["SAFE_OPEN", "DOOR_OPEN"]


def test_pvalue_policies_use_their_own_grid_and_ignore_bias_shift():
    videos, by_video = _synthetic()
    cal = decision.fit_calibration(videos, by_video)
    th = decision.fit_thresholds(videos, by_video, decision.Policy(agg="zsidak"), cal)
    assert th["default"] in decision.PGRID and th["default"] >= 0.5
    m, _ = decision.evaluate(videos, by_video, lambda v: th, decision.Policy(agg="zsidak"), cal)
    assert m["recall"] == 1.0 and m["fp_rate"] < 0.5
    a = decision.type_scores(by_video["n3"], decision.Policy(agg="zsidak"), cal)
    b = decision.type_scores(by_video["n3"], decision.Policy(agg="zsidak", calib="bias"), cal)
    assert {c: s.score for c, s in a.items()} == {c: s.score for c, s in b.items()}


def test_rank_raw_orders_flagged_types_by_raw_probability():
    videos, by_video = _synthetic()
    f = "v0"
    by_video[f] = [_row(f, 0, 20, {"SAFE_OPEN": 0.7, "DOOR_OPEN": 0.95})]
    cal = decision.fit_calibration(videos, by_video)
    th = {"default": 0.5, "per_type": {"DOOR_OPEN": 0.9}, "max_events": 1}
    _, by_margin = decision.predict(by_video[f], th, decision.Policy(), cal)
    _, by_raw = decision.predict(by_video[f], th, decision.Policy(rank="raw"), cal)
    assert by_margin[0][0] == "SAFE_OPEN"  # margin 0.2 beats 0.05
    assert by_raw[0][0] == "DOOR_OPEN"  # raw 0.95 beats 0.7


def test_service_choose_matches_offline_policy():
    """detector.choose with an exported thresholds_v2 must reproduce decision.predict(zsidak, events_from=max)."""
    from app.vj.detector import choose

    videos, by_video = _synthetic(n_windows_normal=(1, 2, 3, 7, 7, 1, 1, 2), seed=3)
    cal = decision.fit_calibration(videos, by_video)
    pol = decision.Policy(agg="zsidak", events_from="max")
    th = decision.fit_thresholds(videos, by_video, pol, cal)
    base = decision.fit_thresholds(videos, by_video, decision.Policy(), cal)
    exported = dict(policy="zsidak+eventsmax", default=th["default"], per_type=th["per_type"], max_events=1,
                    calibration={c: {"mu": cal.mu[c], "sigma": cal.sigma[c]} for c in cal.mu},
                    base={"default": base["default"], "per_type": base["per_type"]})
    pred, _, _ = decision.nested_evaluate(videos, by_video, pol, [(videos, videos)], fixed_thresholds=th)
    # nested_evaluate refits the base thresholds itself on the same fold, so both sides use `base`
    for v in videos:
        rows = by_video[v["file"]]
        hits = choose([((r["start"], r["end"]), r["gate"]) for r in rows], exported)
        assert [h.code for h in hits] == [e[0] for e in pred[v["file"]]], v["file"]
    legacy = dict(default=0.5, per_type={}, max_events=1)
    for v in videos:
        rows = by_video[v["file"]]
        assert [h.code for h in choose([((r["start"], r["end"]), r["gate"]) for r in rows], legacy)] == \
               [h.code for h in decision.decide(decision.type_scores(rows, decision.Policy(), cal), legacy)]
