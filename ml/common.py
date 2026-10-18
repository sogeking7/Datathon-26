"""Общее для скриптов ml/: путь к рантайму сервиса, разметка, группы съёмки, аугментации."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))

LABELS = ROOT / "analysis" / "labels.json"
SETUPS = ROOT / "analysis" / "setups.json"


# Побайтовые дубли в датасете организатора (md5 совпадает). Х-4/Х-47 — один ролик с двумя
# размеченными нарушениями (57 и 68 с), Х-5/Х-48 — одна норма. Сливаем, иначе leave-one-out
# держит копию отложенного видео в обучении.
DUPLICATES = {"Х-47.mov": "Х-4.mov", "Х-48.mov": "Х-5.mov"}


def load_labels(path: Path = LABELS) -> list[dict]:
    """[{file, duration_sec, verdict, timecode_sec, events: [{type_code, timecode_sec, ...}]}]"""
    videos = json.loads(Path(path).read_text())["videos"]
    by_file = {v["file"]: v for v in videos}
    for dup, orig in DUPLICATES.items():
        by_file[orig]["events"] = by_file[orig].get("events", []) + by_file[dup].get("events", [])
    return [v for v in videos if v["file"] not in DUPLICATES]


def load_setups(path: Path = SETUPS) -> dict[str, str]:
    """file -> идентификатор сессии съёмки (одна камера, один фон); см. ml/analysis/scene_groups.py.

    Нужен для group-aware кросс-валидации (leave-one-setup-out): ролик-нарушение и его «норма-двойник»
    сняты одной камерой подряд, и при leave-one-video-out двойник всегда остаётся в обучении.
    """
    data = json.loads(Path(path).read_text())["setups"]
    return {f: sid for sid, files in data.items() for f in files}


def video_codes(video: dict, with_alt: bool = True) -> set[str]:
    """Коды типов, видимых на видео: основной тип каждого события и (опционально) альтернативные."""
    codes = set()
    for e in video.get("events", []):
        codes.add(e["type_code"])
        if with_alt:
            codes.update(e.get("alt_type_codes", []))
    return codes


# --- аугментации: имитация другого ракурса (ТЗ: финальное видео снято под другим углом) ---


def _affine(frames, angle=0.0, scale=1.0, shift=(0.0, 0.0), flip=False, persp=0.0, seed=0):
    import cv2
    import numpy as np

    t, h, w, _ = frames.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += (shift[0] * w, shift[1] * h)
    m = np.vstack([m, [0, 0, 1]])
    if persp:
        rng = np.random.default_rng(seed)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-persp, persp, size=(4, 2)).astype(np.float32) * [w, h]
        m = cv2.getPerspectiveTransform(src, dst.astype(np.float32)) @ m
    out = np.empty_like(frames)
    for i in range(t):
        f = frames[i][:, ::-1] if flip else frames[i]
        out[i] = cv2.warpPerspective(np.ascontiguousarray(f), m, (w, h), borderMode=cv2.BORDER_REFLECT)
    return out


AUGS = {
    "none": None,
    "hflip": lambda f: _affine(f, flip=True),
    "rot_zoom": lambda f: _affine(f, angle=8, scale=1.15, shift=(0.03, -0.02)),
    "persp": lambda f: _affine(f, angle=-6, scale=1.1, persp=0.08, seed=1),
    "flip_persp": lambda f: _affine(f, angle=5, scale=1.1, flip=True, persp=0.06, seed=2),
}


def window_targets(video: dict, start: float, end: float, code: str, state: bool,
                   alt_policy: str = "negative") -> int | None:
    """Метка «признак типа code виден в окне»: 1 / 0 / None (не учитывать).

    Таймкоды — по событиям этого типа (обычно одно на видео). Окно с таймкодом — положительное. Окна целиком до него —
    отрицательные. Окна после таймкода: для состояний (сейф/дверь открыты) — положительные,
    для событий — неизвестно (событие могло ещё длиться), их пропускаем.

    alt_policy — как размечать окна видео, где code указан только как альтернативный тип
    (labels.json: alt_type_codes, 6 роликов). "negative" — как основной отрицательный пример (поведение
    первого выпуска: дверь на Х-24 видимо открыта, но DOOR_OPEN учится как 0); "ignore" — не учитывать
    такие окна (None); "positive" — считать таймкод события таймкодом альтернативного типа.
    """
    events = video.get("events", [])
    times = [e["timecode_sec"] for e in events if e["type_code"] == code]
    if not times:
        alt_times = [e["timecode_sec"] for e in events if code in e.get("alt_type_codes", [])]
        if alt_times and alt_policy == "ignore":
            return None
        if alt_times and alt_policy == "positive":
            times = alt_times
        else:
            return 0
    if any(start <= t < end for t in times):
        return 1
    if all(end <= t for t in times):
        return 0
    return 1 if state else None
