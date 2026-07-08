"""Ragas LLM-as-judge metrics: faithfulness, answer_relevancy, context_relevancy, context_recall."""

import os
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _build_llm() -> LangchainLLMWrapper:
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=os.environ.get("JUDGE_MODEL", "gpt-4o"),
            base_url=base,
            api_key=os.environ["LITELLM_API_KEY"],
            timeout=120,
            max_retries=2,
        )
    )


def _build_embeddings() -> LangchainEmbeddingsWrapper:
    base = os.environ["LITELLM_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=base,
            api_key=os.environ["LITELLM_API_KEY"],
        )
    )


def score_batch(
    queries: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> list[dict[str, float]]:
    """Run all 4 Ragas metrics on a batch. Returns per-sample score dicts."""
    llm = _build_llm()
    emb = _build_embeddings()

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=gt,
        )
        for q, a, c, gt in zip(queries, answers, contexts, ground_truths)
    ]
    dataset = EvaluationDataset(samples=samples)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_relevancy, context_recall],
        llm=llm,
        embeddings=emb,
    )

    df = results.to_pandas()
    scores = []
    for _, row in df.iterrows():
        scores.append(
            {
                "faithfulness": round(float(row.get("faithfulness", 0.0)), 4),
                "answer_relevancy": round(float(row.get("answer_relevancy", 0.0)), 4),
                "context_relevancy": round(float(row.get("context_relevancy", 0.0)), 4),
                "context_recall": round(float(row.get("context_recall", 0.0)), 4),
            }
        )
    return scores


def score_faithfulness_single(query: str, answer: str, contexts: list[str]) -> float:
    """Lightweight single-sample faithfulness check for online sampling."""
    llm = _build_llm()
    sample = SingleTurnSample(user_input=query, response=answer, retrieved_contexts=contexts)
    dataset = EvaluationDataset(samples=[sample])
    results = evaluate(dataset=dataset, metrics=[faithfulness], llm=llm)
    df = results.to_pandas()
    return round(float(df.iloc[0].get("faithfulness", 0.0)), 4)
