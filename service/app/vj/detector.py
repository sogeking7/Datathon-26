"""Детектор на VideoJev: окна → вопросы-гейты по типам → пороги → таймкод по grounding.

Решение о наличии признака принимает только гейт (вопрос с вариантами): grounding в VideoJev
всегда возвращает какой-то отрезок, даже если события нет, поэтому он используется только для
поиска таймкода у типов, уже прошедших порог.

Пороги по типам — из VIDEOJEV_THRESHOLDS (ml/calibrate.py). Логика выбора общая для сервиса
и для скриптов ml/.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .queries import BY_CODE, gate_questions, ground_questions

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {"default": 0.5, "per_type": {}, "max_events": 2}


@dataclass
class TypeScore:
    code: str
    score: float  # max по окнам вероятности «да»
    window: tuple[float, float]


def gate_scores(window_results: list[tuple[tuple[float, float], dict[str, list[float]]]]) -> dict[str, TypeScore]:
    """window_results: [((start, end), {code: probabilities гейта})] → лучший score по каждому типу."""
    best: dict[str, TypeScore] = {}
    for win, probs in window_results:
        for code, p in probs.items():
            s = p[BY_CODE[code].yes_index]
            if code not in best or s > best[code].score:
                best[code] = TypeScore(code, s, win)
    return best


def decide(scores: dict[str, TypeScore], thresholds: dict) -> list[TypeScore]:
    per = thresholds.get("per_type", {})
    hits = [s for s in scores.values() if s.score >= per.get(s.code, thresholds["default"])]
    hits.sort(key=lambda s: s.score - per.get(s.code, thresholds["default"]), reverse=True)
    return hits[: thresholds.get("max_events", 2)]


def pick_timecode(code: str, times: list[float], probs: list[float]) -> tuple[float, list[float]]:
    """Таймкод и 1–3 момента для кадров-доказательств по кривой grounding."""
    peak = max(range(len(probs)), key=probs.__getitem__)
    if BY_CODE[code].state:
        # Состояние (сейф/дверь открыты): первый момент, где кривая поднялась до половины пика.
        level = probs[peak] * 0.5
        first = next(i for i, p in enumerate(probs) if p >= level)
        ts = [times[first], times[peak]]
        return times[first], sorted(set(ts))
    near = sorted(range(len(probs)), key=lambda i: (-probs[i], abs(i - peak)))[:3]
    return times[peak], sorted({times[i] for i in near})


def load_thresholds(path: str | None) -> dict:
    if path and Path(path).is_file():
        return {**DEFAULT_THRESHOLDS, **json.loads(Path(path).read_text())}
    return dict(DEFAULT_THRESHOLDS)


class VideoJevDetector:
    name = "videojev"

    def __init__(self):
        self.model_dir = os.environ.get("VIDEOJEV_MODEL", "/models/videojev-4b-fp16")
        self.head_path = os.environ.get("VIDEOJEV_HEAD") or None
        self.thresholds = load_thresholds(os.environ.get("VIDEOJEV_THRESHOLDS"))

    def run(self, path, info, progress):
        from ..analyzer import Detection
        from . import runtime

        vj = runtime.get(self.model_dir, self.head_path)
        t0 = time.perf_counter()
        with vj.lock:
            t_lock = time.perf_counter()
            sample = vj.decode(path)
            t_dec = time.perf_counter()
            spans = vj.windows(float(sample["sampled_duration"]))
            gates = []
            for i, (a, b) in enumerate(spans):
                w = vj.prepare_window(sample, a, b)
                res = vj.ask(w, gate_questions())
                gates.append(((a, b), {r.id: r.probabilities for r in res}))
                vj.free(w)
                progress(0.8 * (i + 1) / len(spans))
            hits = decide(gate_scores(gates), self.thresholds)
            t_gate = time.perf_counter()
            out = []
            for h in hits:
                # Окно готовим заново: держать пиксели всех окон в RAM дороже, чем пересобрать одно.
                w = vj.prepare_window(sample, *h.window)
                (g,) = vj.ask(w, ground_questions([h.code]))
                tc, ev = pick_timecode(h.code, g.times, g.probabilities)
                vj.free(w)
                out.append(Detection(
                    code=h.code, timecode_sec=tc, confidence=h.score, evidence_ts=ev,
                    observation=OBSERVATIONS[h.code],
                ))
            progress(1.0)
            log.info("videojev %s: очередь %.1fs, декод %.1fs, гейты %.1fs (%d окон), grounding %.1fs",
                     Path(path).name, t_lock - t0, t_dec - t_lock, t_gate - t_dec, len(spans),
                     time.perf_counter() - t_gate)
            return out


# Что видно на видео — без предположений о том, чего на видео нет.
OBSERVATIONS = {
    "SAFE_OPEN": "Дверца сейфа остаётся открытой, пока кассир работает или находится вне сейфа.",
    "UNATTENDED_CASH": "Кассир покидает рабочее место, на столе остаются наличные или сумка с ценностями.",
    "DOOR_OPEN": "После входа или выхода дверь кассового помещения остаётся открытой.",
    "NO_COUNTER": "Кассир пересчитывает банкноты вручную, счётчик банкнот не используется.",
    "UNAUTHORIZED_PERSON": "Второй человек заходит за стол кассира и находится рядом во время работы с наличными.",
    "CASH_EXPOSED_TO_VISITOR": "В помещение входит посетитель, пока на столе открыто лежат наличные.",
    "CIT_PROCEDURE": "Инкассатор передаёт сумку кассиру через стол; проверка сумки и пломбы в кадре не видна.",
    "PERSONAL_ITEMS": "На рабочем столе кассира находятся личные вещи, напиток или смартфон.",
}


def warmup() -> None:
    """Загружает модель и прогоняет короткий синтетический ролик (компиляция ядер Triton)."""
    import logging
    import tempfile

    import cv2
    import numpy as np

    from . import runtime

    log = logging.getLogger(__name__)
    det = VideoJevDetector()
    try:
        runtime.get(det.model_dir, det.head_path)
        path = tempfile.mktemp(suffix=".mp4")
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (464, 848))
        for i in range(150):
            w.write(np.full((848, 464, 3), (i * 3) % 255, np.uint8))
        w.release()
        det.run(path, None, lambda p: None)
        log.info("VideoJev готов")
    except Exception:
        log.exception("Прогрев VideoJev не удался")
