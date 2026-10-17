"""Очередь задач: состояние хранится в JSON на диске, обработка в пуле потоков."""

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import analyzer
from .config import settings
from .video import VideoError

log = logging.getLogger(__name__)


class JobStore:
    def __init__(self, root: Path, workers: int):
        self.jobs_dir = root / "jobs"
        self.uploads_dir = root / "uploads"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._fail_interrupted()

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _write(self, job: dict) -> None:
        tmp = self._path(job["job_id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False))
        tmp.replace(self._path(job["job_id"]))

    def _update(self, job_id: str, **fields) -> dict:
        with self._lock:
            job = self.get(job_id) or {"job_id": job_id}
            job.update(fields)
            self._write(job)
            return job

    def _fail_interrupted(self) -> None:
        # Задачи, которые обрабатывались при остановке сервиса, уже не завершатся.
        for p in self.jobs_dir.glob("*.json"):
            job = json.loads(p.read_text())
            if job.get("status") == "processing":
                job.update(status="failed", error="Обработка прервана перезапуском сервиса")
                self._write(job)

    def get(self, job_id: str) -> dict | None:
        p = self._path(job_id)
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def new_upload_path(self, suffix: str) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        return job_id, self.uploads_dir / f"{job_id}{suffix}"

    def submit(self, job_id: str, video_path: Path, filename: str) -> dict:
        job = self._update(
            job_id, status="processing", progress=0.0, filename=filename, created_at=time.time()
        )
        self._pool.submit(self._run, job_id, video_path)
        return job

    def _run(self, job_id: str, video_path: Path) -> None:
        started = time.time()
        self._update(job_id, started_at=started)

        def progress(p: float) -> None:
            if time.time() - started > settings.job_timeout_sec:
                raise TimeoutError
            self._update(job_id, progress=round(min(max(p, 0.0), 1.0), 3))

        try:
            result = analyzer.analyze(str(video_path), progress)
            self._update(job_id, status="done", progress=1.0, finished_at=time.time(), **result)
        except TimeoutError:
            self._update(job_id, status="failed", error="Превышено время обработки", finished_at=time.time())
        except VideoError as e:
            self._update(job_id, status="failed", error=str(e), finished_at=time.time())
        except Exception:
            log.exception("job %s failed", job_id)
            self._update(job_id, status="failed", error="Внутренняя ошибка анализа", finished_at=time.time())


store = JobStore(settings.data_dir, settings.workers)
