.PHONY: help dev-api dev-ui build up down logs seed fmt check

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

dev-api:  ## Run the API against SQLite with the simulated fleet
	cd backend && DGXCTL_DATABASE_URL="sqlite+aiosqlite:///./dev.db" DGXCTL_DRIVER=sim \
	  DGXCTL_AUTH_MODE=dev ../.venv/bin/uvicorn app.main:app --reload --port 8000

dev-ui:   ## Run the Vite dev server (proxies /api to localhost:8000)
	cd frontend && npm run dev

build:    ## Build the production UI bundle into backend/static
	cd frontend && npm run build

up:       ## Start the whole control plane with docker compose
	docker compose up -d --build

down:     ## Stop it
	docker compose down

logs:     ## Tail the API log
	docker compose logs -f api

check:    ## Typecheck the frontend and syntax-check the backend
	cd frontend && npx tsc -b --pretty false
	cd backend && python3 -m compileall -q app

test:     ## Run the verification suite against the simulated fleet
	cd backend && ../.venv/bin/python -m pytest -q

test-v:   ## Same, with test names
	cd backend && ../.venv/bin/python -m pytest -v
