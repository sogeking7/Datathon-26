"""Пороги гейтов и метрики по критериям ТЗ на результатах score_dataset.py.

python ml/calibrate.py runs/zeroshot/scores.jsonl --out runs/zeroshot/thresholds.json

Метрики (как считает жюри, раздел 9 и FAQ 15–17):
- вердикт по видео: recall на видео с нарушением (пропуск «недопустим»), FP на видео-нормах;
- тип: совпадает ли тип первого события с разметкой;
- таймкод: |наш − размеченный| для видео, где тип угадан.

Пороги подбираются leave-one-video-out: метрики в отчёте посчитаны на видео, которые не
участвовали в подборе порога. Итоговый thresholds.json подобран на всех видео.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import load_labels

from app.vj.detector import decide, gate_scores, pick_timecode
from app.vj.queries import QUERIES

GRID = [round(0.05 * i, 2) for i in range(1, 20)]
CODES = [q.code for q in QUERIES]


def load(path: str, aug: str):
    by_video = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r["aug"] == aug:
            by_video[r["file"]].append(r)
    return by_video


def predict(rows, thresholds):
    scores = gate_scores([((r["start"], r["end"]), r["gate"]) for r in rows])
    hits = decide(scores, thresholds)
    events = []
    for h in hits:
        row = next(r for r in rows if (r["start"], r["end"]) == h.window)
        g = row["ground"][h.code]
        tc, _ = pick_timecode(h.code, g["times"], g["probs"])
        events.append((h.code, tc, h.score))
    return scores, events


def evaluate(videos, by_video, thr_for):
    """thr_for(video) -> thresholds; возвращает метрики и построчный разбор."""
    n_pos = n_neg = tp = fp = type_ok = 0
    tc_err, rows = [], []
    for v in videos:
        scores, events = predict(by_video[v["file"]], thr_for(v))
        pos = v["verdict"] == "violation"
        flagged = bool(events)
        n_pos += pos
        n_neg += not pos
        tp += pos and flagged
        fp += (not pos) and flagged
        label = {e["type_code"]: e["timecode_sec"] for e in v.get("events", [])} if pos else {}
        label_code = ",".join(label) or None
        if pos and flagged and events[0][0] in label:
            type_ok += 1
            tc_err.append(abs(events[0][1] - label[events[0][0]]))
        top = sorted(scores.values(), key=lambda s: -s.score)[:3]
        rows.append(dict(file=v["file"], label=label_code or "-", t=v.get("timecode_sec"),
                         pred=[(c, round(t, 1), round(s, 2)) for c, t, s in events],
                         top=[(s.code, round(s.score, 2)) for s in top]))
    m = dict(videos=len(videos), recall=tp / max(n_pos, 1), fp_rate=fp / max(n_neg, 1),
             verdict_acc=(tp + n_neg - fp) / max(len(videos), 1),
             type_acc_on_pos=type_ok / max(n_pos, 1),
             tc_mae=sum(tc_err) / len(tc_err) if tc_err else None,
             tc_within_2s=sum(e <= 2 for e in tc_err) / len(tc_err) if tc_err else None)
    return m, rows


def fit(videos, by_video, max_events, beta=2.0):
    """Глобальный порог + поправки по типам; цель — F_beta (beta=2: recall важнее, FAQ 16)."""
    def score(th):
        m, _ = evaluate(videos, by_video, lambda v: th)
        p = 1 - m["fp_rate"]  # прокси точности на сбалансированном наборе
        r = m["recall"]
        f = (1 + beta**2) * p * r / max(beta**2 * p + r, 1e-9)
        return f + 0.25 * m["type_acc_on_pos"]

    best = max(({"default": t, "per_type": {}, "max_events": max_events} for t in GRID), key=score)
    # Поправка по типу: только если тип часто «шумит» на нормах и это улучшает цель.
    for code in CODES:
        cands = [best] + [{**best, "per_type": {**best["per_type"], code: t}} for t in GRID]
        best = max(cands, key=score)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("--aug", default="none")
    p.add_argument("--out")
    p.add_argument("--max-events", type=int, default=1)
    a = p.parse_args()
    by_video = load(a.scores, a.aug)
    videos = [v for v in load_labels() if v["file"] in by_video]

    folds = {v["file"]: fit([u for u in videos if u is not v], by_video, a.max_events) for v in videos}
    m_cv, rows = evaluate(videos, by_video, lambda v: folds[v["file"]])
    final = fit(videos, by_video, a.max_events)
    m_fit, _ = evaluate(videos, by_video, lambda v: final)

    for r in rows:
        print(f"{r['file']:10} label={r['label']:24} t={r['t']}  pred={r['pred']}  top={r['top']}")
    print("\nLOVO (честная оценка):", json.dumps(m_cv, ensure_ascii=False))
    print("fit on all (оптимистично):", json.dumps(m_fit, ensure_ascii=False))
    print("thresholds:", json.dumps(final))
    if a.out:
        Path(a.out).write_text(json.dumps(final, indent=1))


if __name__ == "__main__":
    main()
