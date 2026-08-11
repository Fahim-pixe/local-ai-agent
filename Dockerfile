FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 10001 agent && \
    mkdir -p /workspace && \
    chown -R agent:agent /app /workspace

USER agent

EXPOSE 8000

CMD ["uvicorn", "local_ai_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
