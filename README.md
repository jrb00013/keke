# Keke — Excel Datasheet Tool

Keke is a web-based tool for uploading, analyzing, cleaning, charting, and exporting spreadsheet data (Excel, CSV, JSON, Parquet). The UI is served by Node/Express; data processing runs in Python (pandas).

## What works today

- **Upload** with server-issued session IDs and on-disk session storage
- **Analyze**, **clean**, **transform**, **formulas**, **export**
- **Charts** in the browser (Chart.js) plus Excel chart download
- **ML** (scikit-learn): predict, cluster, anomalies, correlation
- **FreeRTOS kernel** (Python): tasks, semaphores, mutex (`house_lock`), message queues, watchdog
- **AI endpoints** are disabled by default (`AI_ENABLED=false`)

## Quick start

```bash
git clone <repo-url>
cd keke

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install

cp env.example .env
# AI_ENABLED is false by default — no API keys required

npm start
# Open http://localhost:3000
```

Sessions are stored under `data/sessions/`. Uploaded files stay on disk for the session lifetime.

## API flow

1. `POST /api/excel/upload` → returns `session_id` and `file_info`
2. Use `session_id` on routes like `/api/excel/:sessionId/analyze/:sheetName`

## Project layout

```
api/                 Express server + Python processors
  session_store.py   Disk-backed session persistence
  excel_processor.py Core spreadsheet logic
  ml_processor.py    sklearn models
data/sessions/       Uploaded data (gitignored)
legacy/              Unused RTOS experiment code
k8s/                 Kubernetes manifests (optional)
```

## Configuration

See `env.example`. Important variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 3000 | HTTP port |
| `AI_ENABLED` | false | LLM routes (503 when false) |
| `KEKE_SESSION_DIR` | `data/sessions` | Session storage path |

## FreeRTOS API

```bash
curl http://localhost:3000/api/rtos/status
curl -X POST http://localhost:3000/api/rtos/watchdog/feed
curl -X POST http://localhost:3000/api/rtos/excel/enqueue \
  -H 'Content-Type: application/json' \
  -d '{"file_path":"/path/to/file.xlsx","operations":[{"type":"remove_duplicates"}]}'
curl http://localhost:3000/api/rtos/excel/results
```

Primitives: `excel_jobs` / `excel_results` queues, `excel_processing` semaphore (3), `house_lock` mutex, `system_watchdog` (30s, feed via API or during job processing).

## Tests

```bash
source .venv/bin/activate
pytest tests/test_session_store.py tests/test_api_routes.py tests/test_excel_processor.py -q
pytest tests/test_freertos_integration.py -q
```

## License

MIT — see [LICENSE](LICENSE).
