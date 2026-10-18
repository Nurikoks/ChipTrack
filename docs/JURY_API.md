# Проверка ChipTRick через API

Рабочая папка: `C:\Users\GGG\Desktop\datathon\ChipTrack-work`.
Текущий локальный адрес: http://127.0.0.1:8000.
Проверка готовности: http://127.0.0.1:8000/health (`status: ready`).
Публичный адрес текущего запуска: https://player-leg-routers-transmit.trycloudflare.com. Актуальный адрес — в `runtime/public-url.txt`; управление и ограничения описаны в [PUBLIC_ACCESS.md](PUBLIC_ACCESS.md). Жюри на другом компьютере нужно использовать публичный адрес, а не `127.0.0.1`.

## Контракт

1. `POST /api/v1/analyze`: заголовок `X-API-Key`, multipart-поле `file`, файл MOV. Ответ содержит `job_id`.
2. `GET /api/v1/jobs/{job_id}` с тем же заголовком: `processing`, `done` или `failed`.
3. При `done`: `verdict` равен `violation` или `no_violation`; для каждого события — `type`, `timecode_sec`, `description`, 1–3 JPEG в `evidence_frames[].image_base64` и время кадра `timestamp_sec`.
4. Дополнительные поля включают версию модели, уверенность и сведения об источнике правила.

Для жюри нужно передать публичный HTTPS-адрес сервиса, API-ключ и эти два маршрута. Пароль администратора передавать не требуется. Ключ хранится локально в `runtime/api-key.txt`; не включайте его в публичную презентацию или скриншот.

## Пример PowerShell на этом компьютере

Запустите PowerShell из рабочей папки. Следующий пример загружает исходный X-1, а не тестовое видео жюри:

```powershell
Set-Location 'C:\Users\GGG\Desktop\datathon\ChipTrack-work'
$chipBase = 'http://127.0.0.1:8000'
$chipKey = (Get-Content -LiteralPath '.\runtime\api-key.txt' -Raw).Trim()
$chipVideo = (Resolve-Path -LiteralPath '.\videos\X-1.mov').Path
$chipJob = curl.exe --silent --show-error --fail -X POST "$chipBase/api/v1/analyze" -H "X-API-Key: $chipKey" -F "file=@$chipVideo" | ConvertFrom-Json
$chipJob
```

Через несколько секунд запросите состояние; повторяйте, пока статус `processing`:

```powershell
$chipResult = Invoke-RestMethod -Uri "$chipBase/api/v1/jobs/$($chipJob.job_id)" -Headers @{'X-API-Key'=$chipKey}
$chipResult | Select-Object job_id,status,verdict,model_version,error
$chipResult.violations | Select-Object type,timecode_sec,description
```

Полный JSON с кадрами можно сохранить, не печатая длинные base64 в консоль:

```powershell
$chipResult | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath '.\runtime\reports\jury-response.json' -Encoding UTF8
```

Для удалённого жюри `$chipBase` заменяется реальным публичным адресом, путь видео — файлом жюри, а ключ передаётся отдельно. Обработка должна завершиться за 300 секунд **от загрузки, включая ожидание в очереди**. Нынешняя внутренняя отсечка обработки не гарантирует этот срок при большой очереди.

## Воспроизводимый локальный аудит

```powershell
.\.venv\Scripts\python.exe tools\jury_api_audit.py
```

Скрипт читает ключ из файла, проверяет авторизацию и ошибки входа, затем одновременно загружает пять исходных видео. Он проверяет сроки, структуру JSON и декодирование JPEG. Он не включает загрузки в обучение и не меняет модель.
Результаты: `runtime/reports/api-audit/report.json`, полные ответы — рядом, каждый под своим `job_id`.
Это функциональная проверка на известных роликах, а не независимая оценка качества на скрытом наборе.

## Ограничения конкурсного запуска

- Публичный HTTPS-доступ подготовлен и проверен через интернет-маршрут с этого ПК на искусственном видео. Независимый тест с отдельного устройства жюри ещё не выполнен.
- При недостаточности доказательств без найденных событий API возвращает `failed` / `insufficient_evidence`. Это не успешный ответ «нарушений нет» и может быть засчитано как необработанное видео.
- Полученный кейс запрещает обучение на тестовых видео жюри. Возможность включить API-загрузки в обучение, ранее запрошенная пользователем, требует отдельного конкурсного ограничения или явного уточнения организаторов.
- Пять коротких исходных видео в одной очереди не подтверждают выдерживание срока для любой длительности и нагрузки.
