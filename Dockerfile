FROM python:3.11-slim

WORKDIR /app

# Install build deps, then the package and its dependencies
COPY pyproject.toml .
COPY rag_eval/ ./rag_eval/

RUN pip install --no-cache-dir -e . && \
    pip cache purge && \
    python3 -c "import ragas.llms.base as _b; from pathlib import Path; _p=Path(_b.__file__); _c=_p.read_text(); _old='from langchain_community.chat_models.vertexai import ChatVertexAI'; _new='ChatVertexAI = None'; _p.write_text(_c.replace(_old, _new)) if _old in _c else None; print('ragas/llms/base.py patched' if _old in _c else 'no patch needed')"

RUN useradd --uid 1000 --no-create-home --shell /sbin/nologin appuser
USER 1000

CMD ["python", "-m", "rag_eval.cli"]
