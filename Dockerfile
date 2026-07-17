# LLMJudge local / staging image (DevStack: Gateway + workers, shared in-process MQ)
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY alembic.ini ./
COPY alembic ./alembic

RUN pip install --no-cache-dir -e ".[postgres,redis]"

ENV LLMJUDGE_PERSISTENCE=sql \
    LLMJUDGE_DATABASE_URL=postgresql+asyncpg://llmj:llmj@postgres:5432/llmjudge \
    LLMJUDGE_SCHEMA_MODE=alembic \
    LLMJUDGE_REDIS_URL=redis://redis:6379/0 \
    LLMJUDGE_FORCE_SAMPLE=1 \
    LLMJUDGE_API_KEYS=dev-key:tenant-dev \
    LLMJUDGE_DISPATCH_CONFIG=/app/configs/dispatch.yaml \
    LLMJUDGE_HOST=0.0.0.0 \
    LLMJUDGE_PORT=8080

EXPOSE 8080

CMD ["llmj-stack"]
