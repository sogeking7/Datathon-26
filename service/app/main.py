import logging
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .config import settings
from .jobs import store
from .rules import CATALOG

logging.basicConfig(level=logging.INFO)
if settings.api_key == "dev-key-change-me":
    logging.warning("API_KEY не задан, используется ключ по умолчанию. Задайте API_KEY перед публикацией.")

app = FastAPI(title="Контроль кассовой дисциплины", version="0.1.0")


@app.on_event("startup")
def _warmup() -> None:
    # Первая загрузка модели и компиляция Triton-ядер занимают минуты; делаем это до первого
    # запроса жюри, а не внутри 5-минутного лимита на видео.
    if settings.detector == "videojev":
        import threading

        from .vj.detector import warmup

        threading.Thread(target=warmup, name="videojev-warmup", daemon=True).start()
STATIC = Path(__file__).parent / "static"
CHUNK = 1024 * 1024


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


async def _accept_upload(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(400, f"Неподдерживаемый формат {suffix or '(нет)'}; ожидается .mov")
    job_id, dest = store.new_upload_path(suffix)
    limit = settings.max_upload_mb * CHUNK
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Файл больше {settings.max_upload_mb} МБ")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Пустой файл")
    job = store.submit(job_id, dest, file.filename or dest.name)
    return {"job_id": job_id, "status": job["status"]}


def _public(job: dict) -> dict:
    out = {"job_id": job["job_id"], "status": job["status"]}
    if job["status"] == "processing":
        out["progress"] = job.get("progress", 0.0)
    elif job["status"] == "done":
        out["verdict"] = job["verdict"]
        out["violations"] = job["violations"]
    else:
        out["error"] = job.get("error")
    return out


def _get_job(job_id: str) -> dict:
    job = store.get(job_id) if job_id.isalnum() else None
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return _public(job)


# --- API для жюри (по ключу) ---

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def api_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    return _get_job(job_id)


# --- Эндпоинты веб-интерфейса (без ключа, ТЗ не требует авторизации в UI) ---

@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    return _get_job(job_id)


@app.get("/api/v1/violation-types")
def violation_types():
    return [
        {
            "code": t.code,
            "title": t.title,
            "rationale": t.rationale,
            "rules": [{"document": r.document, "clause": r.clause, "quote": r.quote, "url": r.url} for r in t.rules],
        }
        for t in CATALOG.values()
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
