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
git clone https://github.com/jrb00013/keke.git
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
  rtos_bridge.py     HTTP routes for the FreeRTOS kernel
  metrics.js         Prometheus exposition served at /metrics
data/sessions/       Uploaded data (gitignored)
legacy/              FreeRTOS kernel; freertos_integration.py is imported by
                     api/rtos_bridge.py and the C sources are reference only
k8s/                 Kubernetes manifests (optional)
monitoring/          Prometheus config and alert rules
```

## Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for Docker, Kubernetes, monitoring
and configuration. In short:

```bash
make install && make start     # local
docker compose up              # container
make test && make lint         # verify
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
make test      # npx jest + python -m pytest (agentci excluded)
npx jest       # Node only
make test-py   # Python unit/integration only (-m "not agentci")
```

### Agentic CI (local Ollama)

Optional LLM-as-judge tests live under `tests/agentci/` and are **excluded** from
`make test-py` / default CI so a missing Ollama never fails required checks.

```bash
# Ollama must be reachable (default http://127.0.0.1:11434).
# Example: docker descent-ollama on :11436 with model qwen2.5:3b
export OLLAMA_HOST=http://127.0.0.1:11436
export OLLAMA_JUDGE_MODEL=qwen2.5:3b
export OPENAI_BASE_URL=http://127.0.0.1:11436/v1

# openai package required for the Assistant exercise (optional otherwise)
pip install 'openai>=1.0.0'

make test-agentci
```

`api/assistant.py` honors `OPENAI_BASE_URL` for any OpenAI-compatible endpoint
(including Ollama `/v1`). When the base URL is localhost and no API key is set,
the client uses `api_key=ollama`.

## License

MIT — see [LICENSE](LICENSE).
