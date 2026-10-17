"""Общее для скриптов ml/: путь к рантайму сервиса, разметка, аугментации."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))

LABELS = ROOT / "analysis" / "labels.json"


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


def window_targets(video: dict, start: float, end: float, code: str, state: bool) -> int | None:
    """Метка «признак типа code виден в окне»: 1 / 0 / None (не учитывать).

    Таймкоды — по событиям этого типа (обычно одно на видео). Окно с таймкодом — положительное. Окна целиком до него —
    отрицательные. Окна после таймкода: для состояний (сейф/дверь открыты) — положительные,
    для событий — неизвестно (событие могло ещё длиться), их пропускаем.
    """
    times = [e["timecode_sec"] for e in video.get("events", []) if e["type_code"] == code]
    if not times:
        return 0
    if any(start <= t < end for t in times):
        return 1
    if all(end <= t for t in times):
        return 0
    return 1 if state else None
