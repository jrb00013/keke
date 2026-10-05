# Keke - Excel Datasheet Tool
#
# Thin wrapper around the commands CI runs, plus the Docker Compose stacks.
# Anything here should be safe to run from a clean checkout after
# `make install`.

SHELL := /bin/bash
PY := python3
VENV := .venv
BIN := $(VENV)/bin
COMPOSE := docker compose
PROFILES := databases,monitoring,proxy,logging

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)

# ---------------------------------------------------------------- setup ----

.PHONY: install
install: $(BIN)/python ## Install Node and Python dependencies
	npm ci
	$(BIN)/pip install -r requirements.txt

.PHONY: setup
setup: ## Alias for install
	@$(MAKE) install

# ------------------------------------------------------------------ run ----

.PHONY: start
start: ## Start the app (AI disabled by default)
	npm start

.PHONY: dev
dev: ## Start the app with hot reload
	npm run dev

.PHONY: smoke
smoke: ## Run the end-to-end smoke test against a running server
	@echo "Requires a running server (PORT ?= 3000)"
	./scripts/smoke_test.sh $${PORT:-3000}

# ----------------------------------------------------------------- test ----

.PHONY: test
test: test-js test-py ## Run all tests

.PHONY: test-js
test-js: ## Run the Node test suite
	npm test

.PHONY: test-py
test-py: ## Run the Python test suite
	$(BIN)/python -m pytest tests/ -q

.PHONY: lint
lint: lint-js lint-py ## Run all linters

.PHONY: lint-js
lint-js: ## Lint JavaScript
	npm run lint

.PHONY: lint-py
lint-py: ## Lint Python (syntax/undefined names, then formatting)
	$(BIN)/flake8 api/ --count --select=E9,F63,F7,F82 --show-source --statistics
	$(BIN)/black --check api/

.PHONY: format
format: ## Format Python in place
	$(BIN)/black api/

# ------------------------------------------------------------ validate -----

.PHONY: validate
validate: validate-k8s validate-compose ## Run the deploy-config validators

.PHONY: validate-k8s
validate-k8s: ## Check the Kubernetes manifests are internally consistent
	$(BIN)/python scripts/validate_k8s.py

.PHONY: validate-compose
validate-compose: ## Check docker-compose.yml renders (all profiles)
	COMPOSE_PROFILES=$(PROFILES) $(COMPOSE) config -q

# --------------------------------------------------------------- docker ----

.PHONY: up
up: ## Start the application in Docker
	$(COMPOSE) up -d --build keke-app

.PHONY: up-databases
up-databases: ## Start the app plus redis, mongodb and postgres
	$(COMPOSE) --profile databases up -d --build

.PHONY: up-monitoring
up-monitoring: ## Start the app plus prometheus and grafana
	$(COMPOSE) --profile monitoring up -d

.PHONY: up-proxy
up-proxy: ## Start the app plus the nginx reverse proxy
	$(COMPOSE) --profile proxy up -d

.PHONY: up-all
up-all: ## Start every optional stack (databases, monitoring, proxy, ELK)
	$(COMPOSE) --profile $(PROFILES) up -d

.PHONY: down
down: ## Stop and remove the containers
	$(COMPOSE) --profile $(PROFILES) down --remove-orphans

.PHONY: logs
logs: ## Follow the application logs
	$(COMPOSE) logs -f keke-app

# ------------------------------------------------------------ kubernetes ---

.PHONY: k8s-check
k8s-check: validate-k8s ## Alias for validate-k8s

# ----------------------------------------------------------------- clean ---

.PHONY: clean
clean: ## Remove build/test artifacts (keeps .venv and node_modules)
	rm -rf .pytest_cache .coverage coverage.xml htmlcov __pycache__ logs
	find api tests scripts -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
