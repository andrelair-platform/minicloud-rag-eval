FROM python:3.11-slim

WORKDIR /app

# Install build deps, then the package and its dependencies
COPY pyproject.toml .
COPY rag_eval/ ./rag_eval/

RUN pip install --no-cache-dir -e . && \
    pip cache purge && \
    python3 -c "
import ragas.llms.base as b
from pathlib import Path
p = Path(b.__file__)
code = p.read_text()
old = 'from langchain_community.chat_models.vertexai import ChatVertexAI'
new = 'ChatVertexAI = None  # patched: not available in langchain-community>=0.3'
if old in code:
    p.write_text(code.replace(old, new))
    print('Patched ragas/llms/base.py: removed VertexAI import')
"

RUN useradd --uid 1000 --no-create-home --shell /sbin/nologin appuser
USER 1000

CMD ["python", "-m", "rag_eval.cli"]
