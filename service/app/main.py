import logging
import secrets
from pathlib import Path

from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .config import settings
from .jobs import store
from .rules import CATALOG

logging.basicConfig(level=logging.INFO)
if settings.api_key == "dev-key-change-me":
    logging.warning("API_KEY не задан, используется ключ по умолчанию. Задайте API_KEY перед публикацией.")

DESCRIPTION = """
Сервис находит на видеозаписях кассовой зоны нарушения кассовой дисциплины (VideoJev-4B на GPU) и возвращает
вердикт, тип нарушения, таймкод, кадры-доказательства и ссылку на пункт нормативного акта НБРК.

## Как пользоваться

1. Нажмите **Authorize** и введите ключ в заголовок `X-API-Key` (нужен для `/api/v1/analyze` и `/api/v1/jobs`).
2. `POST /api/v1/analyze` — загрузите видео (поле формы `file`). В ответ придёт `job_id`.
3. `GET /api/v1/jobs/{job_id}` — опрашивайте раз в 1–2 с, пока `status` = `processing`.
   Затем придёт `done` с `verdict` и `violations` или `failed` с `error`.

```bash
curl -H "X-API-Key: $KEY" -F "file=@video.mov" http://<host>/api/v1/analyze
curl -H "X-API-Key: $KEY" http://<host>/api/v1/jobs/<job_id>
```

## Ограничения

- Форматы: `.mov` (основной), `.mp4`, `.m4v`, `.avi`, `.mkv`; до 300 МБ.
- Время обработки на Tesla T4: ролик 10–20 с — 15–35 с, ролик 140 с — около 2.5 мин; лимит задачи 280 с
  (считая ожидание в очереди).
- В ответе не больше одного нарушения на ролик; `confidence` — запас оценки над порогом типа, отображённый в [0.5, 1].
- Справочник типов нарушений и нормативной базы — `GET /api/v1/violation-types` (без ключа).
"""

TAGS = [
    {"name": "Анализ видео", "description": "Загрузка видео и получение результата (нужен `X-API-Key`)."},
    {"name": "Справочники", "description": "Типы нарушений и нормативная база, проверка работоспособности."},
]

app = FastAPI(title="Контроль кассовой дисциплины", version="0.1.0", description=DESCRIPTION, openapi_tags=TAGS)


# --- схемы ответов (только для документации и валидации; форма ответа не меняется) ---

class Rule(BaseModel):
    document: str = Field(description="Нормативный акт")
    clause: str = Field(description="Пункт акта", examples=["п. 20 ч. 2"])
    quote: str = Field(description="Цитата из пункта")
    url: str = Field(description="Ссылка на акт на adilet.zan.kz")


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(description="Момент кадра, секунды от начала видео", examples=[4.2])
    image_base64: str = Field(description="Кадр в JPEG, base64")


class Violation(BaseModel):
    type: str = Field(description="Название нарушения", examples=["Сейф (хранилище наличных) оставлен открытым"])
    type_code: str = Field(description="Код типа (см. /api/v1/violation-types)", examples=["SAFE_OPEN"])
    timecode_sec: float = Field(description="Таймкод нарушения, секунды от начала видео", examples=[4.2])
    description: str = Field(description="Что видно на записи, почему это нарушение и основание")
    confidence: float = Field(ge=0, le=1, description="Уверенность в [0.5, 1]", examples=[0.84])
    rules: list[Rule]
    evidence_frames: list[EvidenceFrame] = Field(description="1–3 кадра-доказательства")


class JobAccepted(BaseModel):
    job_id: str = Field(examples=["25a08c784ae3"])
    status: Literal["processing"] = "processing"


class Job(BaseModel):
    job_id: str = Field(examples=["25a08c784ae3"])
    status: Literal["processing", "done", "failed"]
    progress: float | None = Field(None, description="Только при status = processing, от 0 до 1")
    verdict: Literal["violation", "no_violation"] | None = Field(None, description="Только при status = done")
    violations: list[Violation] | None = Field(None, description="Только при status = done; пустой при no_violation")
    error: str | None = Field(None, description="Только при status = failed")


class ViolationType(BaseModel):
    code: str = Field(examples=["SAFE_OPEN"])
    title: str
    rationale: str
    rules: list[Rule]


class Error(BaseModel):
    detail: str


class Health(BaseModel):
    status: Literal["ok"]


AUTH_ERRORS = {401: {"model": Error, "description": "Неверный или отсутствующий X-API-Key"}}


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


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Ключ API, выданный жюри")


def require_key(x_api_key: str | None = Security(api_key_header)) -> None:
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

@app.post(
    "/api/v1/analyze", dependencies=[Depends(require_key)], response_model=JobAccepted, tags=["Анализ видео"],
    summary="Загрузить видео на анализ",
    responses={**AUTH_ERRORS, 400: {"model": Error, "description": "Неподдерживаемый формат или пустой файл"},
               413: {"model": Error, "description": "Файл больше 300 МБ"}},
)
async def api_analyze(file: UploadFile = File(..., description="Видеофайл (.mov)")):
    """Принимает видео и ставит его в очередь. Результат — по `job_id` через `GET /api/v1/jobs/{job_id}`."""
    return await _accept_upload(file)


@app.get(
    "/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)], response_model=Job,
    response_model_exclude_none=True, tags=["Анализ видео"], summary="Статус и результат анализа",
    responses={**AUTH_ERRORS, 404: {"model": Error, "description": "Задача не найдена"}},
)
def api_job(job_id: str):
    """`processing` (с `progress`) → `done` (`verdict`, `violations`) или `failed` (`error`)."""
    return _get_job(job_id)


# --- Эндпоинты веб-интерфейса (без ключа, ТЗ не требует авторизации в UI) ---

@app.post("/ui/analyze", include_in_schema=False)
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept_upload(file)


@app.get("/ui/jobs/{job_id}", include_in_schema=False)
def ui_job(job_id: str):
    return _get_job(job_id)


@app.get("/api/v1/violation-types", response_model=list[ViolationType], tags=["Справочники"],
         summary="Типы нарушений и нормативная база")
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


@app.get("/health", response_model=Health, tags=["Справочники"], summary="Проверка работоспособности")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
