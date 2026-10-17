# ForteBank Datathon 26 — локальный AI-сервис проверки кассовой дисциплины по видео

Сервис принимает видеозапись кассового рабочего места (вид сверху) и сообщает проверяющему:
есть ли визуальные признаки возможного нарушения, **когда** (таймкод), **что видно** (кадры-доказательства
и описание) и **какое правило** может быть затронуто (документ НБРК/АРРФР, пункт, ссылка на adilet.zan.kz).
Результат — подсказка для человека, а не юридическое заключение.

* Всё выполняется **локально** (RTX 3050 4 ГБ): детектор RT-DETRv2 + VLM Qwen3-VL (Apache-2.0). Без облачных API;
  после загрузки весов интернет не нужен (`HF_HUB_OFFLINE=1`).
* Анализ только визуальный, без звука и без распознавания лиц: люди различаются только по позиции и поведению.
* Видео не сохраняются: файл удаляется сразу после обработки, результаты хранятся в памяти 6 ч.

Документация: [архитектура и модели](docs/architecture.md) · [типы нарушений](taxonomy.md) ·
[нормативные источники](docs/normative_sources.md) · [реестр правил](data/rules_registry.json) ·
[журнал экспериментов](docs/experiment_log.md) · [оценка качества](docs/evaluation.md) · [план питча](docs/pitch.md)

## 1. Запуск (Windows, нативно)

```powershell
# 1) окружение (вне OneDrive, на диске с местом)
python -m venv E:\fortebank_env
E:\fortebank_env\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
E:\fortebank_env\Scripts\python.exe -m pip install -r requirements.txt

# 2) веса (один раз, нужен интернет)
$env:HF_HOME="E:\fortebank_env\hf"; E:\fortebank_env\Scripts\python.exe scripts\download_weights.py

# 3) сервис (офлайн); при первом запуске генерируется и печатается API-ключ (runtime\api_key.txt)
powershell -ExecutionPolicy Bypass -File scripts\start_service.ps1 -Port 8000
```

Интерфейс: `http://localhost:8000/` · проверка: `GET /health`.

Linux + Docker (NVIDIA Container Toolkit): веса скачать в `./models` (`HF_HOME=./models python scripts/download_weights.py`),
затем `FORTE_API_KEY=<ключ> docker compose up -d`. *Docker-образ подготовлен, но на машине команды не проверялся (нет Docker).*

### Публичный URL
Используется ngrok (агент `E:\fortebank_env\bin\ngrok.exe`, бесплатный аккаунт команды):
```powershell
E:\fortebank_env\bin\ngrok.exe config add-authtoken <токен из dashboard.ngrok.com>   # один раз
E:\fortebank_env\bin\ngrok.exe http 8000                                               # публичный https-адрес
```
Туннель только пересылает HTTP-запросы; обработка видео и все модели остаются на этой машине.
Cloudflare quick tunnel (`scripts\start_service.ps1 -Tunnel`) в нашей сети заблокирован (порт 7844 сбрасывается),
поэтому выбран ngrok (работает через 443). Известная особенность бесплатного ngrok: если большой файл отправлен
**без** ключа, ответ 401 сервиса может прийти клиенту искажённым (100/обрыв); с маленьким телом и с верным ключом всё штатно.

### Переменные окружения
| Переменная | По умолчанию | Назначение |
|---|---|---|
| `FORTE_API_KEY` | — (обязательна) | ключ для заголовка `X-API-Key` |
| `FORTE_VLM` | `Qwen/Qwen3-VL-2B-Instruct` | VLM (альтернатива `Qwen/Qwen3-VL-4B-Instruct`) |
| `FORTE_JOB_DEADLINE_SEC` | 270 | жёсткий лимит на задачу от момента загрузки; по истечении — лучший частичный результат |
| `FORTE_MAX_UPLOAD_MB` | 300 | максимальный размер файла |
| `HF_HOME` | — | каталог с весами |
| `FORTE_DEBUG` | 0 | добавить в ответ отладочный блок (для разработки) |

## 2. API (для жюри)

```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
# -> {"job_id": "3f2a9c1e0b7d"}
curl https://<host>/api/v1/jobs/3f2a9c1e0b7d -H "X-API-Key: <ключ>"
```

Ответ по готовности (обязательные поля — строго по ТЗ; дополнительные поля только добавлены):
```json
{
  "job_id": "3f2a9c1e0b7d", "status": "done", "verdict": "violation",
  "violations": [{
    "type": "Ценности/документы оставлены без присмотра на рабочем месте",
    "timecode_sec": 9.0,
    "description": "На видео работник отходит от рабочего места, на столе остаются банкноты; ... Правило: ...",
    "evidence_frames": [{"timestamp_sec": 9.0, "image_base64": "/9j/..."}],
    "type_id": "T01_UNATTENDED_VALUABLES", "confidence": 0.99,
    "rule": {"grounding": "medium", "references": [{"document": "...", "clause": "п. 20, ч. 2", "url": "https://adilet.zan.kz/rus/docs/V1900019680", "verified": true}]}
  }],
  "partial": false, "video_duration_sec": 19.8, "processing_time_sec": 24.1
}
```
* `status`: `processing` → `done` | `failed`. Каждая задача обязательно завершается; повреждённый/неподдерживаемый файл → `failed` с сообщением.
* Без ключа или с неверным ключом → `401`. Неподдерживаемое расширение → `415`, пустой файл → `400`, слишком большой → `413`.
* Задачи обрабатываются очередью на одном GPU; несколько одновременных загрузок допустимы.

## 3. Ограничения
См. раздел «Ограничения и известные ошибки» в [docs/architecture.md](docs/architecture.md) и результаты оценки в
[docs/evaluation.md](docs/evaluation.md).
