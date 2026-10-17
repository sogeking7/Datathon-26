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
