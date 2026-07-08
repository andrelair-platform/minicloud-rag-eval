FROM python:3.11-slim

WORKDIR /app

# Install build deps, then the package and its dependencies
COPY pyproject.toml .
COPY rag_eval/ ./rag_eval/

RUN pip install --no-cache-dir -e . && \
    pip cache purge

RUN useradd --uid 1000 --no-create-home --shell /sbin/nologin appuser
USER 1000

CMD ["python", "-m", "rag_eval.cli"]
