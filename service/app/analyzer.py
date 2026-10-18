"""Анализ видео: детекторы → события → ответ в формате API.

Детектор получает путь к видео и возвращает список Detection. Основной — VideoJevDetector
(app/vj), включается DETECTOR=videojev; BaselineDetector-заглушка нужна для тестов без GPU.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .config import settings
from .rules import CATALOG, format_rules
from .video import VideoError, VideoInfo, grab_frame, probe, to_jpeg_base64

log = logging.getLogger(__name__)

ProgressFn = Callable[[float], None]


@dataclass
class Detection:
    code: str  # ключ из rules.CATALOG
    timecode_sec: float
    observation: str  # что именно видно на видео, без предположений
    confidence: float
    evidence_ts: list[float] = field(default_factory=list)  # 1–3 таймкода кадров


class Detector(Protocol):
    name: str

    def run(self, path: str, info: VideoInfo, progress: ProgressFn) -> list[Detection]: ...


class BaselineDetector:
    """Заглушка: ничего не находит. Нужна, чтобы проверить сквозной путь API → очередь → ответ."""

    name = "baseline"

    def run(self, path: str, info: VideoInfo, progress: ProgressFn) -> list[Detection]:
        progress(1.0)
        return []


def _detectors() -> list[Detector]:
    # DETECTOR=videojev включает модель (нужны GPU и requirements-gpu.txt); по умолчанию заглушка,
    # чтобы тесты и UI работали без GPU.
    if settings.detector == "videojev":
        from .vj.detector import VideoJevDetector

        return [VideoJevDetector()]
    return [BaselineDetector()]


DETECTORS: list[Detector] = _detectors()

# Ниже этого порога событие не попадает в ответ (ложные срабатывания штрафуются).
# Контракт для детекторов: confidence — уже калиброванная уверенность, а не сырой score модели;
# событие, прошедшее порог детектора, обязано иметь confidence >= MIN_CONFIDENCE
# (см. vj.detector.confidence_from_margin), иначе решение детектора молча отменяется здесь.
MIN_CONFIDENCE = 0.5
# События одного типа ближе этого интервала схлопываются в одно.
MERGE_WINDOW_SEC = 5.0


def _merge(dets: list[Detection]) -> list[Detection]:
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: (d.code, d.timecode_sec)):
        prev = out[-1] if out else None
        if prev and prev.code == d.code and d.timecode_sec - prev.timecode_sec < MERGE_WINDOW_SEC:
            if d.confidence > prev.confidence:
                out[-1] = d
            continue
        out.append(d)
    return sorted(out, key=lambda d: d.timecode_sec)


def _evidence_frames(path: str, ts_list: list[float]) -> list[dict]:
    """Кадры-доказательства; нечитаемый кадр пропускается, а не роняет весь анализ (вердикт уже принят)."""
    frames = []
    for t in ts_list:
        try:
            frames.append({"timestamp_sec": round(t, 1), "image_base64": to_jpeg_base64(grab_frame(path, t))})
        except VideoError as e:
            log.warning("кадр-доказательство пропущен: %s", e)
    return frames


def analyze(path: str, progress: ProgressFn = lambda p: None) -> dict:
    info = probe(path)
    found: list[Detection] = []
    for i, det in enumerate(DETECTORS):
        def sub(p: float, i: int = i) -> None:
            progress((i + p) / len(DETECTORS))

        found.extend(det.run(path, info, sub))

    events = _merge([d for d in found if d.confidence >= MIN_CONFIDENCE and d.code in CATALOG])
    violations = []
    for d in events:
        vt = CATALOG[d.code]
        ts_list = (d.evidence_ts or [d.timecode_sec])[:3]
        violations.append(
            {
                "type": vt.title,
                "type_code": vt.code,
                "timecode_sec": round(d.timecode_sec, 1),
                "description": f"{d.observation} {vt.rationale} Основание: {format_rules(vt.code)}.",
                "confidence": round(d.confidence, 2),
                "rules": [
                    {"document": r.document, "clause": r.clause, "quote": r.quote, "url": r.url}
                    for r in vt.rules
                ],
                "evidence_frames": _evidence_frames(path, ts_list),
            }
        )
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "video": {"duration_sec": round(info.duration_sec, 1), "fps": round(info.fps, 2)},
    }
