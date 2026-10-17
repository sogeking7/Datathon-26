# Контроль кассовой дисциплины — сервис

Веб-интерфейс и API для поиска визуальных признаков нарушений кассовой дисциплины на видео.
Всё работает локально, облачные AI-сервисы не используются.

## Запуск

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
cp .env.example .env   # задайте API_KEY
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker: `docker build -t kassa . && docker run -p 8000:8000 --env-file .env -v $PWD/data:/data kassa`

Интерфейс: `http://localhost:8000/`, документация API: `/docs`.

## API (раздел 10 ТЗ)

```bash
curl -X POST http://localhost:8000/api/v1/analyze -H "X-API-Key: $API_KEY" -F "file=@video.mov"
# {"job_id": "3f2a9c1e0b4d", "status": "processing"}
curl http://localhost:8000/api/v1/jobs/3f2a9c1e0b4d -H "X-API-Key: $API_KEY"
```

Ответ `done` соответствует схеме ТЗ (`verdict`, `violations[].type/timecode_sec/description/evidence_frames`).
Дополнительные поля: `type_code`, `confidence`, `rules` (документ, пункт, цитата, ссылка).
`GET /api/v1/violation-types` возвращает каталог типов нарушений и норм.

## Устройство

- `app/main.py`: эндпоинты; `/api/v1/*` по ключу, `/ui/*` для веб-интерфейса.
- `app/jobs.py`: очередь задач (пул потоков, состояние в `data/jobs/*.json`, таймаут 280 с).
- `app/analyzer.py`: запуск детекторов, фильтр по уверенности, схлопывание дублей, сборка ответа.
- `app/rules.py`: каталог типов нарушений с проверенными цитатами Правил НБРК № 231 и № 14.
- `app/video.py`: чтение видео и кадры-доказательства в JPEG/base64.

Чтобы добавить детектор, реализуйте `run(path, info, progress) -> list[Detection]` и добавьте его в `analyzer.DETECTORS`.

## Тесты

```bash
.venv/bin/python -m pytest -q
```
