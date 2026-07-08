"""Bootstrap eval dataset from ragdb chunks using Ragas synthetic testset generation."""

import json
import os
import sys
from pathlib import Path

import psycopg2
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator


def _litellm_base() -> str:
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _load_chunks_from_ragdb() -> list[Document]:
    conn = psycopg2.connect(
        host=os.environ.get("PG_HOST", "localhost"),
        port=int(os.environ.get("PG_PORT", "5432")),
        dbname=os.environ.get("PG_DBNAME", "ragdb"),
        user=os.environ.get("PG_USER", "aiplatform"),
        password=os.environ["PG_PASSWORD"],
    )
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content, meta
        FROM document_chunk
        WHERE length(content) > 100
        ORDER BY id
        LIMIT 1000
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"[generate-dataset] Loaded {len(rows)} chunks from ragdb")
    return [
        Document(page_content=row[0], metadata=row[1] if row[1] else {})
        for row in rows
    ]


def generate_dataset() -> None:
    out_path = Path(os.environ.get("OUTPUT_PATH", "/output/eval_dataset_draft.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_size = int(os.environ.get("TESTSET_SIZE", "100"))
    base = _litellm_base()
    api_key = os.environ["LITELLM_API_KEY"]

    llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", base_url=base, api_key=api_key, timeout=120)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", base_url=base, api_key=api_key)
    )

    docs = _load_chunks_from_ragdb()
    if len(docs) < 10:
        print("[generate-dataset] Too few chunks — cannot generate testset", file=sys.stderr)
        sys.exit(1)

    print(f"[generate-dataset] Generating {test_size} synthetic Q&A pairs…")
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    try:
        testset = generator.generate_with_langchain_docs(docs, testset_size=test_size)
    except AttributeError:
        # Ragas 0.2.x renamed method to generate()
        testset = generator.generate(docs=docs, test_size=test_size)

    df = testset.to_pandas()

    entries = []
    for i, row in df.iterrows():
        entries.append(
            {
                "id": f"synthetic-{i:03d}",
                "query": str(row.get("user_input", row.get("question", ""))),
                "ground_truth": str(row.get("reference", row.get("ground_truth", ""))),
                "source_doc": str(row.get("source_doc", "")),
                "domain": "unknown",
            }
        )

    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    print(f"[generate-dataset] Wrote {len(entries)} entries → {out_path}")
    print("\nNext step: review eval_dataset_draft.json, assign domains, keep best 50,")
    print("rename to eval_dataset.json, and commit to minicloud-gitops/manifests/ai/eval/")
