"""Логика решения VideoJev-детектора без модели и GPU."""

from app.rules import CATALOG
from app.vj.detector import OBSERVATIONS, decide, gate_scores, pick_timecode
from app.vj.queries import QUERIES, gate_questions, ground_questions


def test_queries_cover_catalog():
    assert {q.code for q in QUERIES} == set(CATALOG) == set(OBSERVATIONS)
    ids = [q["id"] for q in gate_questions() + ground_questions([q.code for q in QUERIES])]
    assert len(ids) == len(set(ids))  # VideoJev требует уникальные id в одном запросе


def test_gate_scores_takes_max_over_windows():
    s = gate_scores([((0, 20), {"SAFE_OPEN": [0.2, 0.7, 0.1]}), ((10, 30), {"SAFE_OPEN": [0.9, 0.1, 0.0]})])
    assert s["SAFE_OPEN"].score == 0.9 and s["SAFE_OPEN"].window == (10, 30)


def test_decide_uses_per_type_threshold_and_limit():
    s = gate_scores([((0, 20), {"SAFE_OPEN": [0.6, 0.4, 0], "DOOR_OPEN": [0.95, 0.05], "NO_COUNTER": [0.7, 0.3]})])
    th = {"default": 0.5, "per_type": {"DOOR_OPEN": 0.99}, "max_events": 1}
    hits = decide(s, th)
    assert [h.code for h in hits] == ["NO_COUNTER"]  # запас 0.2 больше, чем у SAFE_OPEN (0.1)
    assert decide(s, {**th, "default": 0.99}) == []


def test_pick_timecode_event_peak_and_state_onset():
    times = [0.5, 1.5, 2.5, 3.5, 4.5]
    probs = [0.1, 0.3, 0.9, 0.95, 0.2]
    tc, ev = pick_timecode("UNATTENDED_CASH", times, probs)
    assert tc == 3.5 and 1 <= len(ev) <= 3 and 3.5 in ev
    tc, ev = pick_timecode("SAFE_OPEN", times, probs)  # состояние: момент, где кривая дошла до половины пика
    assert tc == 2.5 and ev == [2.5, 3.5]


def test_confidence_from_margin_survives_analyzer_filter():
    """Событие, прошедшее порог гейта, не должно отбрасываться analyzer.MIN_CONFIDENCE.

    Регрессия первого выпуска: пороги типов в проде (ml/artifacts/thresholds.json) доходят до 0.05,
    а в confidence писался сырой score, поэтому срабатывания со score в [порог, 0.5) терялись."""
    import json
    from pathlib import Path

    from app import analyzer
    from app.vj.detector import confidence_from_margin, threshold_for

    th = json.loads((Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "thresholds.json").read_text())
    for q in QUERIES:
        t = threshold_for(th, q.code)
        assert confidence_from_margin(t, t) >= analyzer.MIN_CONFIDENCE
        assert confidence_from_margin(t + 1e-6, t) >= analyzer.MIN_CONFIDENCE
    assert confidence_from_margin(1.0, 0.05) == 1.0
    assert confidence_from_margin(0.3, 0.05) < confidence_from_margin(0.6, 0.05) < confidence_from_margin(0.9, 0.05)
    assert confidence_from_margin(0.2, 0.7) == 0.5  # ниже порога сюда не попадает, но функция не ломается
    assert threshold_for(th, "UNAUTHORIZED_PERSON") == 0.05 and threshold_for(th, "SAFE_OPEN") == 0.45


def test_choose_reproduces_offline_policy_on_real_scores():
    """detector.choose (политика zsidak+eventsmax из ml/artifacts/thresholds_v2.json) должен давать те же
    типы и окна, что ml/decision.predict на тех же оценках окон — иначе сервис снова разойдётся с оценкой."""
    import json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    scores = root / "runs" / "head_v1" / "scores_cv.jsonl"
    th_path = root / "ml" / "artifacts" / "thresholds_v2.json"
    if not scores.is_file():
        import pytest

        pytest.skip("runs/head_v1/scores_cv.jsonl отсутствует (scripts/fetch_runs.sh)")
    sys.path.insert(0, str(root / "ml"))
    import decision
    from app.vj.detector import choose

    th = json.loads(th_path.read_text())
    cal = decision.Calibration(mu={c: v["mu"] for c, v in th["calibration"].items()},
                               sigma={c: v["sigma"] for c, v in th["calibration"].items()})
    pol = decision.Policy(agg="zsidak", events_from="max", max_events=th["max_events"])
    by_video = decision.load_scores(scores)
    flagged = 0
    for file, rows in by_video.items():
        gates = [((r["start"], r["end"]), r["gate"]) for r in rows]
        service_hits = [(h.code, h.window) for h in choose(gates, th)]
        _, events = decision.predict(rows, th, pol, cal)
        if events:  # as in decision.nested_evaluate(events_from="max"): type/timecode from the max rule when it fires
            base = {**th["base"], "max_events": th["max_events"]}
            _, base_events = decision.predict(rows, base, decision.Policy(max_events=th["max_events"]), cal)
            events = base_events or events
        assert [c for c, _ in service_hits] == [e[0] for e in events], file
        flagged += bool(service_hits)
    assert flagged > 0


def test_grab_frame_falls_back_when_msec_seek_fails(tmp_path, monkeypatch):
    """Х-23 из датасета: перемотка CAP_PROP_POS_MSEC после ~11 с не читает кадр, и задача падала с
    «Не удалось прочитать кадр». grab_frame должен дочитать кадр другим путём."""
    import cv2
    import numpy as np

    from app import video

    path = str(tmp_path / "v.avi")
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for i in range(30):
        wr.write(np.full((24, 32, 3), i * 8, np.uint8))
    wr.release()

    real = cv2.VideoCapture

    class NoMsecSeek:
        def __init__(self, p):
            self.cap, self.broken = real(p), False

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_MSEC:
                self.broken = True
                return True
            return self.cap.set(prop, value)

        def read(self):
            if self.broken:
                self.broken = False
                return False, None
            return self.cap.read()

        def __getattr__(self, name):
            return getattr(self.cap, name)

    monkeypatch.setattr(video.cv2, "VideoCapture", NoMsecSeek)
    assert video.grab_frame(path, 2.0).shape == (24, 32, 3)
    assert video.grab_frame(path, 99.0).shape == (24, 32, 3)  # за концом файла — последний кадр


def test_unreadable_evidence_frame_does_not_fail_analysis(monkeypatch):
    from app import analyzer
    from app.video import VideoError

    def boom(path, t):
        raise VideoError("нет кадра")

    monkeypatch.setattr(analyzer, "grab_frame", boom)
    assert analyzer._evidence_frames("x.mov", [1.0, 2.0]) == []


def test_grounding_is_asked_in_the_scored_context():
    """Ответ VideoJev зависит от предыдущих вопросов в упакованном контексте. Таймкод в сервисе должен
    считаться на том же списке вопросов, что ml/score_dataset.py (на нём обучена голова и измерена
    локализация): все гейты, затем grounding по всем типам."""
    from app.vj.detector import scoring_questions

    qs = scoring_questions()
    assert qs[: len(QUERIES)] == gate_questions()
    assert qs[len(QUERIES):] == ground_questions([q.code for q in QUERIES])
    assert len({q["id"] for q in qs}) == len(qs)
