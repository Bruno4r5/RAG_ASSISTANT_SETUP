#!/usr/bin/env python
"""evaluate_ragas_totalenergies.py
=================================

Evaluate the TotalEnergies ESG report that was pre‑processed into
``vector_store/`` JSON chunks (via *ingest_docs.py*) against the
curated Q‑A dataset ``curated_esg_dataset_totalenergies_v2.csv``.

The script will

1. **Load** or build a Llama‑Index `VectorStoreIndex` from the JSON
   files in ``--index_dir`` (default: ``./vector_store``). If a
   persisted index already exists in that directory it is re‑used; if
   not, the JSON files themselves are indexed and then persisted for
   future runs.
2. **Query** the index with every *question* in the curated dataset
   (column names must include ``question`` and ``answer``).
3. **Capture** the RAG answer *and* the source chunks (contexts).
4. **Save** raw RAG outputs to
   ``<output_prefix>_rag_results.json``.
5. **Convert** those outputs into a RAGAS `EvaluationDataset` and run
   the standard metrics *(faithfulness, answer‑relevancy,
   context‑precision, context‑recall)*.
6. **Export**
   * the detailed metric table → ``<output_prefix>_ragas_metrics.xlsx``
   * the four‑column table you asked for →
     ``<output_prefix>_rag_outputs.xlsx``

---
"""
import os
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
import logging
from tqdm import tqdm

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LlamaIndexLLMWrapper
from ragas import EvaluationDataset
from ragas.integrations.llama_index import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "openai_config_template.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def build_or_load_index(index_dir: str, embeddings):
    """Load a persisted Llama‑Index if one exists, otherwise build it
    directly from all ``*.json`` files inside *index_dir* and then
    persist it for next time."""
    idx_path = Path(index_dir)
    try:
        storage_ctx = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_ctx, embed_model=embeddings)
        logger.info("Loaded existing index from %s", index_dir)
        return index
    except Exception as exc:
        logger.warning("No persisted index in %s (or failed to load – %s). "
                       "Rebuilding from JSON files …", index_dir, exc)

    documents = SimpleDirectoryReader(
        index_dir,
        recursive=True,
        required_exts=[".json"],
    ).load_data()
    logger.info("Indexed %d processed JSON chunks", len(documents))

    index = VectorStoreIndex.from_documents(documents, embed_model=embeddings)
    index.storage_context.persist(persist_dir=index_dir)
    logger.info("Persisted freshly‑built index to %s", index_dir)
    return index

def main(args):
    # ------------------------------------------------------------------
    # 0) Credentials -----------------------------------------------------
    # ------------------------------------------------------------------
    config = load_config()
    
    # Set OpenAI API key from config
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    # ------------------------------------------------------------------
    # 1) Models ----------------------------------------------------------
    # ------------------------------------------------------------------
    generator_llm = OpenAI(model=config['completion_model'])
    embeddings = OpenAIEmbedding(model=config['embedding_model'])

    # ------------------------------------------------------------------
    # 2) Index / Query Engine -------------------------------------------
    # ------------------------------------------------------------------
    logger.info("Loading or building vector index …")
    index = build_or_load_index(args.index_dir, embeddings)
    query_engine = index.as_query_engine(similarity_top_k=args.top_k)

    # ------------------------------------------------------------------
    # 3) Dataset ---------------------------------------------------------
    # ------------------------------------------------------------------
    df = pd.read_csv(args.dataset_path)
    mandatory = {"ground_truth_answer", "question"}
    if not mandatory.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns {mandatory}, got {set(df.columns)}")

    # ------------------------------------------------------------------
    # 4) Run RAG ---------------------------------------------------------
    # ------------------------------------------------------------------
    results = []
    logger.info("Running RAG for %d questions …", len(df))
    for _idx, row in tqdm(df.iterrows(), total=len(df)):
        q = str(row["question"])
        truth = str(row["ground_truth_answer"])

        rag_response = query_engine.query(q)
        answer_text = str(rag_response)
        contexts = [sn.node.get_content() for sn in getattr(rag_response, "source_nodes", [])]

        results.append({
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer_text,
            "reference": truth,
        })

    raw_path = Path(f"{args.output_prefix}_rag_results.json")
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), "utf-8")
    logger.info("Raw RAG outputs saved → %s", raw_path)

    # ------------------------------------------------------------------
    # 5) RAGAS evaluation ----------------------------------------------
    # ------------------------------------------------------------------
    ragas_dataset = EvaluationDataset.from_list(results)
    evaluator_llm = LlamaIndexLLMWrapper(generator_llm)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    logger.info("Scoring with RAGAS … this may take a while.")
    eval_result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=ragas_dataset,
    )

    logger.info("Aggregated metrics →\n%s", eval_result)

    metrics_df = eval_result.to_pandas()
    metrics_path = Path(f"{args.output_prefix}_ragas_metrics.xlsx")
    metrics_df.to_excel(metrics_path, index=False)
    logger.info("Per‑question metric table saved → %s", metrics_path)

    # ------------------------------------------------------------------
    # 6) Export four‑column table --------------------------------------
    # ------------------------------------------------------------------
    concise_df = pd.DataFrame({
        "question": [r["user_input"] for r in results],
        "ground_truth_answer": [r["reference"] for r in results],
        "retrieved_contexts": ["\n\n---\n\n".join(r["retrieved_contexts"]) for r in results],
        "response": [r["response"] for r in results],
    })
    concise_path = Path(f"{args.output_prefix}_rag_outputs.xlsx")
    concise_df.to_excel(concise_path, index=False)
    logger.info("Question / GT‑answer / contexts / response saved → %s", concise_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TotalEnergies ESG report with RAGAS.")
    parser.add_argument(
        "--dataset_path",
        default="./dataset/curated_esg_dataset_totalenergies_v2.csv",
        help="CSV containing 'question' and 'answer' columns.")
    parser.add_argument(
        "--index_dir",
        default="./vector_store",
        help="Directory with processed JSON chunks or a persisted index.")
    parser.add_argument(
        "--completion_model",
        default="gpt-4o-mini",
        help="OpenAI model for generation & evaluation (chat completion).")
    parser.add_argument(
        "--embedding_model",
        default="text-embedding-3-small",
        help="OpenAI embedding model for vector index.")
    parser.add_argument(
        "--top_k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--output_prefix",
        default="totalenergies",
        help="Filename prefix for all saved artefacts.")
    args = parser.parse_args()

    main(args)
