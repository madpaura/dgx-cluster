# Two stages: build the UI, then serve it from the API container so there is a
# single port to expose and no CORS to configure.
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "sqlalchemy[asyncio]>=2.0.36" \
      "asyncpg>=0.30" "pydantic>=2.9" "pydantic-settings>=2.6" "asyncssh>=2.18" \
      "httpx>=0.27" "authlib>=1.3" "itsdangerous>=2.2" "greenlet>=3.1" "mcp>=2.2" \
 && pip install --no-cache-dir pytest pytest-asyncio aiosqlite

COPY backend/app ./app
# The verification suite ships with the image: `./setup.sh test` proves the
# build you are about to run on real hardware is sound, using the simulated
# fleet. Small, and the only way to check the deployed artifact rather than
# a developer's checkout.
COPY backend/tests ./tests
COPY backend/pytest.ini ./
COPY --from=ui /backend/static ./static

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
