"""Утилиты для чтения видео и извлечения кадров-доказательств."""

import base64
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


class VideoError(Exception):
    pass


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError("Не удалось открыть видеофайл")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise VideoError("В видео нет кадров")
        return VideoInfo(
            fps=fps,
            frame_count=count,
            duration_sec=count / fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iter_frames(path: str, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Последовательно отдаёт (timestamp_sec, BGR-кадр) с частотой ~sample_fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError("Не удалось открыть видеофайл")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx / fps, frame
            idx += 1
    finally:
        cap.release()


def grab_frame(path: str, timestamp_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000)
        ok, frame = cap.read()
        if not ok:
            raise VideoError(f"Не удалось прочитать кадр на {timestamp_sec:.1f} с")
        return frame
    finally:
        cap.release()


def to_jpeg_base64(frame: np.ndarray, max_side: int = 960, quality: int = 85) -> str:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VideoError("Не удалось закодировать кадр в JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
