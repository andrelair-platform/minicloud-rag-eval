# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app

# uv — ~10× faster dependency resolution/install than pip (build-time only).
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY rag_eval/ ./rag_eval/

# Install deps with a BuildKit cache mount: uv reuses downloaded wheels across builds
# (the cache lives OUTSIDE the image layer, so the final image stays small — the win
# `--no-cache-dir`/`pip cache purge` used to give — AND rebuilds are fast). uv's fast
# resolver is the main cold-build speed-up for the heavy ragas/langchain stack.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e . && \
    find /usr/local/lib/python3.11/site-packages/ragas -name "base.py" -path "*/llms/*" \
      -exec sed -i 's|from langchain_community.chat_models.vertexai import ChatVertexAI|ChatVertexAI = None  # patched: removed in langchain-community>=0.3|g' {} \; && \
    python3 -c "from ragas.llms import LangchainLLMWrapper; print('ragas import OK')"

# Pre-download tiktoken encoding so ai-namespace egress block doesn't affect faithfulness metric
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN mkdir -p /app/.tiktoken && \
    python3 -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); print('tiktoken cached OK')" && \
    chmod -R 755 /app/.tiktoken

RUN useradd --uid 1000 --no-create-home --shell /sbin/nologin appuser
USER 1000

CMD ["python", "-m", "rag_eval.cli"]
