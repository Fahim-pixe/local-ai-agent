FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 sandbox

WORKDIR /workspace/project
USER sandbox

ENTRYPOINT ["/bin/sh", "-lc"]
