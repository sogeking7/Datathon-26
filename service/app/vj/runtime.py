"""Запуск VideoJev-4B на Tesla T4 (16 ГБ, Turing).

Что пришлось обойти относительно официального рецепта (H200, BF16, FlashAttention 2):
- T4 не умеет BF16, а remote code жёстко включает autocast(bfloat16). Используем копию модели,
  где bfloat16 заменён на float16 (готовит ml/prepare_model.py), и грузим веса в FP16.
- FlashAttention 2 на Turing нет → attn_implementation="sdpa". При этом transformers включает
  enable_gqa, который memory-efficient ядро SDPA не поддерживает, и SDPA падает в math-путь
  с матрицей N² → OOM уже на ~7 тыс. токенов. Отключаем GQA-путь (KV разворачиваются явно)
  и разрешаем только EFFICIENT_ATTENTION.

Видео декодируется один раз; длинные ролики режутся на перекрывающиеся окна, каждое окно
кодируется отдельно (контекст 16k токенов ≈ 20 с при 2 fps без сильного даунскейла).
"""

import math
import threading
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

FPS = 2.0
CONTEXT = 16384
WINDOW_SEC = 20.0
STRIDE_SEC = 10.0


def _patch_sdpa() -> None:
    import transformers.integrations.sdpa_attention as sa

    sa.use_gqa_in_sdpa = lambda *a, **k: False


@dataclass
class Window:
    start: float
    end: float
    pixels: torch.Tensor
    record: dict
    features: torch.Tensor | None = None


@dataclass
class Scored:
    """Результат одного вопроса в одном окне; времена — в секундах от начала видео."""

    id: str  # код типа (question["code"]) или id вопроса
    task: str
    probabilities: list[float]
    times: list[float] = field(default_factory=list)  # для grounding: середины ячеек
    hidden: dict | None = None  # для обучения головы: pairs/query скрытые состояния


class VideoJev:
    def __init__(self, model_dir: str | Path, head_path: str | Path | None = None, device: str = "cuda:0"):
        from transformers import AutoModel, AutoProcessor

        model_dir = str(model_dir)
        src = (Path(model_dir) / "modeling_videojev.py").read_text()
        if "dtype=torch.bfloat16" in src:
            raise RuntimeError(f"{model_dir}: нужна FP16-копия модели, см. ml/prepare_model.py")
        _patch_sdpa()
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, dtype=torch.float16,
            device_map={"": device}, attn_implementation="sdpa",
        )
        self.model.prepare_for_inference()
        if head_path:
            self.model.head.load_state_dict(torch.load(head_path, map_location=device))
        pkg = type(self.model).__module__.rsplit(".", 1)[0]
        self._proc = import_module(f"{pkg}.processing_videojev")
        self._video = import_module(f"{pkg}.videojev_video")
        self._temporal = import_module(f"{pkg}.videojev_temporal")
        self.lock = threading.Lock()  # одна GPU — один анализ за раз
        self._captured: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.model.head.register_forward_hook(self._capture)
        self.capture_hidden = False

    def _capture(self, module, args, output):
        if self.capture_hidden:
            pairs, queries = args
            self._captured.append((pairs.detach().half().cpu(), queries.detach().half().cpu()))

    # --- видео -----------------------------------------------------------------------------

    def decode(self, path: str, fps: float = FPS, transform=None) -> dict:
        """Кадры RGB на сетке fps. transform(frames: np.ndarray) -> np.ndarray — аугментация."""
        import av

        with av.open(str(path)) as c:
            s = c.streams.video[0]
            duration = float(s.duration * s.time_base) if s.duration is not None else c.duration / 1e6
        sample = self._video.read_video(dict(video_path=str(path), duration=duration), fps, max_frames=None)
        if transform is not None:
            sample["video"] = transform(sample["video"])
        return sample

    @staticmethod
    def windows(duration: float, win: float = WINDOW_SEC, stride: float = STRIDE_SEC,
                fps: float = FPS) -> list[tuple[float, float]]:
        if duration <= win * 1.3:
            return [(0.0, duration)]
        starts = list(np.arange(0.0, max(duration - win, 0.0) + 1e-6, stride))
        if starts[-1] + win < duration:
            # Начало окна — строго на сетке кадров: процессор сверяет таймстемпы с индексами кадров.
            starts.append(math.floor((duration - win) * fps) / fps)
        return [(float(s), float(min(s + win, duration))) for s in starts]

    def prepare_window(self, sample: dict, start: float, end: float, context: int = CONTEXT) -> Window:
        """То же, что processing_videojev.prepare_video, но для отрезка уже декодированных кадров."""
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
        from transformers.video_utils import VideoMetadata

        times = np.asarray(sample["times"])
        sel = (times >= start - 1e-6) & (times < end - 1e-6)
        frames = sample["video"][sel]
        local = times[sel] - start
        n, height, width, _ = frames.shape
        vp = self.processor.video_processor
        budget = context * 4 // 5

        def tokens(cap):
            h, w = smart_resize(n, height, width, temporal_factor=vp.temporal_patch_size,
                                factor=vp.patch_size * vp.merge_size,
                                min_pixels=vp.size["shortest_edge"], max_pixels=n * cap)
            return (n + 1) // 2 * (h // 32) * (w // 32)

        low, high = max(1024, math.ceil(vp.size["shortest_edge"] / n)), 640 * 640
        if tokens(low) > budget:
            raise ValueError("Окно не помещается в контекст даже при минимальном разрешении")
        while low < high:
            mid = (low + high + 1) // 2
            low, high = (mid, high) if tokens(mid) <= budget else (low, mid - 1)
        meta = VideoMetadata(total_num_frames=n, fps=sample["fps"], frames_indices=list(range(n)))
        native = vp(videos=[frames], video_metadata=[meta], do_sample_frames=False,
                    size=dict(shortest_edge=vp.size["shortest_edge"], longest_edge=n * low), return_tensors="pt")
        grid = native["video_grid_thw"][0].tolist()
        score_times, bounds = self._temporal.pair_timeline(local, end - start)
        record = dict(frames=n, fps=sample["fps"], grid_thw=grid, tokens=int(np.prod(grid)) // 4,
                      score_times=score_times.tolist(), score_bounds=bounds.tolist(),
                      sampled_duration=end - start)
        return Window(start, end, native["pixel_values_videos"], record)

    # --- скоринг ---------------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, w: Window) -> None:
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            w.features = self.model.encode_video(w.pixels, torch.tensor([w.record["grid_thw"]]))

    @torch.inference_mode()
    def ask(self, w: Window, questions: list[dict], context: int = CONTEXT, keep_hidden: bool = False) -> list[Scored]:
        if w.features is None:
            self.encode(w)
        self._captured.clear()
        self.capture_hidden = keep_hidden
        out: list[Scored] = []
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                for batch, rows in self._proc.prepare_query_groups(self.processor, w.record, questions, context):
                    o = self.model(**batch, visual_features=w.features)
                    for i, row in enumerate(rows):
                        k = int(o.score_mask[i].sum())
                        s = Scored(id=row.get("code", row["id"]), task=row["task"],
                                   probabilities=o.probabilities[i, :k].float().cpu().tolist())
                        if row["task"] == "grounding":
                            s.times = [w.start + t for t in w.record["score_times"]]
                        if keep_hidden:
                            pairs, queries = self._captured[-1]
                            s.hidden = dict(pairs=pairs[i, :k].clone(), query=queries[i].clone())
                        out.append(s)
        finally:
            self.capture_hidden = False
        return out

    def free(self, w: Window) -> None:
        w.features = w.pixels = None
        torch.cuda.empty_cache()


_instance: VideoJev | None = None
_instance_lock = threading.Lock()


def get(model_dir: str, head_path: str | None = None) -> VideoJev:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = VideoJev(model_dir, head_path)
        return _instance
