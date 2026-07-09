FROM python:3.11-slim

WORKDIR /app

# Install build deps, then the package and its dependencies
COPY pyproject.toml .
COPY rag_eval/ ./rag_eval/

RUN pip install --no-cache-dir -e . && \
    pip cache purge && \
    find /usr/local/lib/python3.11/site-packages/ragas -name "base.py" -path "*/llms/*" \
      -exec sed -i 's|from langchain_community.chat_models.vertexai import ChatVertexAI|ChatVertexAI = None  # patched: removed in langchain-community>=0.3|g' {} \; && \
    python3 -c "from ragas.llms import LangchainLLMWrapper; print('ragas import OK')"

RUN useradd --uid 1000 --no-create-home --shell /sbin/nologin appuser
USER 1000

CMD ["python", "-m", "rag_eval.cli"]
