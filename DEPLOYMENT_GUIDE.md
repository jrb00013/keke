# Keke Deployment Guide

How to run, test, containerise and deploy Keke. Every command below has been
executed against this repository.

> **Scope note.** Keke is an AI-assisted Excel datasheet tool. The Node API, the
> Python processing helpers, the FreeRTOS bridge and the observability stack are
> all working. The datastores (Redis, MongoDB, PostgreSQL) and cloud storage
> providers are **present but not yet wired into request handling** — see
> [Known limitations](#known-limitations). There is **no authentication layer**;
> do not expose the service to an untrusted network.

---

## Contents

- [Requirements](#requirements)
- [Local development](#local-development)
- [Tests and linters](#tests-and-linters)
- [Docker](#docker)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Kubernetes](#kubernetes)
- [CI/CD](#cicd)
- [Known limitations](#known-limitations)

---

## Requirements

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.11 | The image is pinned to `python:3.11-slim`. |
| Node.js | 18 | The image is pinned to `node:18-bookworm-slim`. |
| Docker + Compose v2 | any recent | `docker-compose` (v1) is not supported. |
| kubectl | any recent | Only for the Kubernetes path. |

The Python wheels in `requirements.txt` are all pre-built for CPython 3.11, so
no compiler or BLAS/ATLAS headers are needed. A local `virtualenv` is enough.

---

## Local development

```bash
git clone https://github.com/jrb00013/keke.git
cd keke

python3 -m venv .venv && source .venv/bin/activate
make install            # npm install + pip install -r requirements.txt
make start              # AI disabled by default
```

`run.py` is the canonical entrypoint and offers the same capabilities:

```bash
python3 run.py install                 # install Node and Python dependencies
python3 run.py start --mode app-only   # app only (default)
python3 run.py start --mode with-db    # app + datastores
python3 run.py start --mode with-monitoring
python3 run.py start --mode full       # everything
python3 run.py status
python3 run.py stop
python3 run.py restart
python3 run.py test
```

Useful `make` targets:

| Target | Effect |
| --- | --- |
| `make help` | List every target. |
| `make install` / `make setup` | Install dependencies. |
| `make start` / `make dev` | Start the app / start with hot reload. |
| `make test` | Both test suites. |
| `make lint` | ESLint + flake8 + Black check. |
| `make format` | Apply Black in place. |
| `make validate` | Deployment-config validators (see below). |
| `make smoke` | End-to-end smoke test against a running server. |
| `make up` / `make down` | Docker up / down. |
| `make up-databases` / `up-monitoring` / `up-proxy` / `up-all` | Opt-in stacks. |
| `make logs` | Follow container logs. |
| `make clean` | Remove build/test artifacts, keeping `.venv` and `node_modules`. |

### Health and metrics

```bash
curl http://localhost:3000/health    # { status, timestamp, uptime, version }
curl http://localhost:3000/metrics   # Prometheus text exposition
```

---

## Tests and linters

```bash
make test           # npx jest  +  python -m pytest
npx jest            # Node suite (metrics, python_runner)
python -m pytest    # Python suite

make lint           # npm run lint (ESLint)
                    # + flake8 --select=E9,F63,F7,F82 (syntax/undefined names)
                    # + black --check api/
```

CI additionally runs the deployment-config validators, which you can run
locally with `make validate`:

| Command | Checks |
| --- | --- |
| `python scripts/validate_k8s.py` | Every `k8s/*.yaml` parses; no duplicate objects; Service selectors match exactly one workload; every referenced Secret/ConfigMap key exists; **no env var is set that the application does not read**. |
| `docker compose config -q` | The Compose file renders with every profile enabled. |

Two limits worth knowing: `docker compose config` does **not** verify that a
bind-mount source exists on the host (Docker silently creates a directory
instead), and it does not run the image build. Build at least once before
trusting a compose change.

---

## Docker

The application is self-contained, so `docker compose up` starts only the app.
Everything else is opt-in through profiles:

```bash
docker compose up                        # app only
docker compose --profile databases up    # + redis, mongodb, postgres
docker compose --profile monitoring up   # + prometheus, grafana
docker compose --profile proxy up        # + nginx reverse proxy
docker compose --profile logging up      # + elasticsearch, logstash, kibana
docker compose --profile databases --profile monitoring up   # combine
```

```bash
docker compose build keke-app            # build the image
docker compose ps
docker compose logs -f keke-app
docker compose down                      # add -v to drop volumes
```

The image is a three-stage build: Node dependencies, Python dependencies, then
a slim runtime that copies both in and runs as the non-root `appuser`.

The published host port is configurable if something else already uses 3000:

```bash
KEKE_HOST_PORT=3121 docker compose up
```

The container always listens on 3000 internally; only the host side moves.

---

## Configuration

Copy `env.example` to `.env` in the repository root. `api/server.js` loads it
at startup, so no export is needed.

```bash
cp env.example .env
```

Only these variables are read by the application:

| Variable | Default | Read by | Purpose |
| --- | --- | --- | --- |
| `PORT` | `3000` | `api/server.js` | HTTP listen port. |
| `AI_ENABLED` | `false` | `api/server.js`, `api/api_routes.js` | Enables the `/api/ai/*` routes. |
| `ALLOWED_ORIGINS` | *(empty)* | `api/server.js` | Comma-separated CORS allowlist. Empty means same-origin only. |
| `MAX_FILE_SIZE` | `52428800` | `api/api_routes.js` | Upload size limit in bytes. |
| `KEKE_SESSION_DIR` | `./data/sessions` | `api/api_routes.js` | Where uploaded workbooks are stored. |
| `PYTHON_TIMEOUT_MS` | `120000` | `api/python_runner.js` | Per-invocation Python timeout. The child is `SIGKILL`ed and the route returns 504. |
| `KEKE_RTOS_SOCKET` | `data/rtos.sock` | `api/rtos_bridge.py` | FreeRTOS bridge socket path. |
| `KEKE_ENV` | `development` | `run.py` | Selects the startup environment. |
| `OPENAI_API_KEY` | – | Python AI helpers | Enables the OpenAI provider. |
| `ANTHROPIC_API_KEY` | – | Python AI helpers | Enables the Anthropic provider. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET` | – | `api/session_store.py` | S3 storage. |
| `DROPBOX_TOKEN` | – | `api/session_store.py` | Dropbox storage. |
| `GOOGLE_DRIVE_TOKEN`, `GOOGLE_DRIVE_FOLDER_ID` | – | `api/session_store.py` | Google Drive storage. |

AI routes return a clear "AI not configured" error when `AI_ENABLED` is off or
no key is present, rather than failing opaquely.

Rate limiting is applied to `/api/*` at **100 requests per 15 minutes per IP**.

---

## Monitoring

```bash
docker compose --profile monitoring up -d
```

- Prometheus: <http://localhost:9090> — the `keke-app` target scrapes
  `keke-app:3000/metrics` and should report `health: up`.
- Grafana: <http://localhost:3001> — provisioned datasource and dashboard;
  default credentials `admin` / `admin`, which you should change.

### Metrics exposed by the app

| Metric | Type | Labels |
| --- | --- | --- |
| `keke_http_requests_total` | counter | `method`, `route`, `status` |
| `keke_http_request_duration_seconds` | histogram | `method`, `route` |
| `keke_python_calls_total` | counter | `script`, `outcome` (`success`/`error`/`timeout`) |
| `keke_python_call_duration_seconds` | histogram | `script` |
| `keke_process_resident_memory_bytes` | gauge | – |
| `keke_process_uptime_seconds` | gauge | – |
| `keke_build_info` | gauge | `version`, `node_version` |

`route` labels are normalised: session UUIDs collapse to `:sessionId` and the
route label set is capped at 200 values, with overflow collapsing to `other`, so
crafted sheet names cannot grow the series count without bound.

### Alerting rules

`monitoring/keke_rules.yml` contains six rules that only reference metrics the
app actually exports (`KekeAppDown`, `KekeHighErrorRate`, `KekeSlowResponses`,
`KekeProcessMemoryHigh`, `KekePythonTimeouts`, `KekePythonErrorRate`).

`monitoring/keke_rules.exporters.yml` holds rules that need exporters which are
**not** deployed (`node_exporter`, `kube-state-metrics`, `redis_exporter`,
`postgres_exporter`, `mongodb_exporter`, `blackbox_exporter`). It is not loaded by
default and is not referenced from `rule_files`. Deploy the matching exporters
first, then add the file to `rule_files` in `monitoring/prometheus.yml`.

No Alertmanager is deployed, so alerts appear in the Prometheus UI only. Add an
Alertmanager service and re-enable the commented `alerting:` block in
`monitoring/prometheus.yml` to route them.

---

## Kubernetes

```bash
python scripts/validate_k8s.py    # always run this first
```

Apply in this order — the script prints the commands:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml          # copied from secrets.yaml.example
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/database-deployments.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/monitoring.yaml
```

`./deploy.sh` and `./scripts/deploy.sh <environment>` wrap the same sequence and
also apply `k8s/monitoring.yaml` and `k8s/ingress.yaml`.

The Secret is **not** committed and **not** applied by CI. Create it first:

```bash
cp k8s/secrets.yaml.example k8s/secrets.yaml
$EDITOR k8s/secrets.yaml
kubectl apply -f k8s/secrets.yaml
```

Only `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `POSTGRES_PASSWORD` and
`MONGODB_PASSWORD` are consumed. The AI keys are referenced with
`optional: true`, so the pod starts without them.

Access the app:

```bash
kubectl port-forward service/keke-service 3000:3000 -n keke
```

Before applying the Ingress, replace `keke.yourdomain.com` and
`monitoring.yourdomain.com` in `k8s/ingress.yaml` with real hostnames.

---

## CI/CD

`.github/workflows/ci-cd.yml` runs on pushes to `main` and `develop` and on
releases:

| Job | Trigger | What it does |
| --- | --- | --- |
| `test` | every push | ESLint, Jest, flake8, Black, pytest, Compose config, `validate_k8s.py` |
| `security` | non-PR | `npm audit`, Trivy, pip safety |
| `build` | push or release | Builds and pushes the image (needs `test` and `security`) |
| `deploy-staging` | push to `develop` | Applies manifests to the staging cluster |
| `deploy-production` | release | Applies manifests to the production cluster |
| `rollback` | on failure | Reverts to the previous image |

The deploy jobs use the `staging` and `production` GitHub Environments, so
approval gates and environment secrets are configured there. They require the
following repository secrets: `KUBE_CONFIG`, `STAGING_NAMESPACE`,
`PRODUCTION_NAMESPACE`, and registry credentials.

---

## Known limitations

These are real gaps, not oversights in the documentation:

- **No authentication or authorisation.** There is no JWT, session or RBAC
  layer. `helmet` security headers, CORS restrictions and rate limiting are the
  only protections. The app must not be exposed to the public internet as-is.
- **Datastores are not wired in.** Redis, MongoDB and PostgreSQL deploy and
  start, but request handling does not use them. The `/api/cloud/*` routes are
  the only consumers of the datastore configuration, and they are off unless
  credentials are supplied.
- **Cluster monitoring is not configured.** The manifests ship no ServiceAccount
  or RBAC, so Prometheus cannot use `kubernetes_sd_configs`; it scrapes only the
  application. Grant list/watch and add the discovery jobs if you need it.
- **Formula evaluation is a stub.** `applyFormulas` supports `SUM(A:B)`,
  `AVERAGE(A:B)`, `COUNT(A:B)` and single cell references such as `A1`. Anything
  else returns a per-column error rather than a wrong result.
- **The HPA needs `metrics-server`.** `k8s/hpa.yaml` targets CPU utilisation,
  which requires a metrics server installed in the cluster.
