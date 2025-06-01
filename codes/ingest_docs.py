"""
-----------------
Build or update a JSON‑based llama‑index store from a
folder of source documents.

Run it once to create the index, and again whenever we add new PDFs /
TXTs / DOCXs – it will embed only the files that aren’t already in the
store.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import yaml
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode, Document 

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from typing import List

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an ESG assistant expert. "
    "Answer **only** with information found in the provided context. "
    "Cite the page number after every sentence (e.g., [p 12]). "
    "If the context is insufficient, reply with “I don’t know”."
)

def load_openai_config(config_path: str = "config/openai_config_template.yaml") -> None:
    """Populate env‑vars from a YAML file if present (falls back to .env)."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        os.environ.setdefault("OPENAI_API_KEY", cfg.get("openai_api_key", ""))
        if cfg.get("base_url"):
            os.environ.setdefault("OPENAI_BASE_URL", cfg["base_url"])


def configure_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Configure global llama‑index Settings (replaces old ServiceContext)."""
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI("gpt-4o-mini", temperature=0, system_prompt=SYSTEM_PROMPT)
    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

# ---------------------------------------------------------------------------
# Chunking helper that injects metadata
# ---------------------------------------------------------------------------

def docs_to_nodes(docs: Sequence[Document]) -> List[BaseNode]:
    """Chunk documents and attach `file_name` and `page` metadata."""
    parser: SentenceSplitter = Settings.node_parser 
    nodes: List[BaseNode] = []
    for doc in docs:
        file_path = doc.metadata.get("file_path", "")
        file_name = Path(file_path).name if file_path else ""
        page_label = doc.metadata.get("page_label") or doc.metadata.get("page_number")
        for n in parser.get_nodes_from_documents([doc]):
            n.metadata["file_name"] = file_name
            n.metadata["file_path"] = file_path 
            if page_label:
                n.metadata["page"] = page_label  
            nodes.append(n)
    return nodes

# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def build_index(docs: Sequence, persist_dir: str) -> VectorStoreIndex:
    """Create a new VectorStoreIndex and persist it."""
    nodes = docs_to_nodes(docs)
    index = VectorStoreIndex(nodes, show_progress=True)
    index.storage_context.persist(persist_dir=persist_dir)
    return index


# ---------------------------------------------------------------------------
# Main CLI flow
# ---------------------------------------------------------------------------

def main() -> None:
    default_source = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(
        description="Build or update a persistent llama‑index JSON store (no FAISS)."
    )
    parser.add_argument(
        "--source_dir", default=str(default_source), help="Folder with PDFs / TXTs"
    )
    parser.add_argument(
        "--persist_dir", default="./vector_store", help="Folder that holds index JSON"
    )
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument(
        "--rebuild", action="store_true", help="Ignore existing store and rebuild"
    )
    args = parser.parse_args()

    load_openai_config()
    configure_settings(args.chunk_size, args.chunk_overlap)

    docs_path = Path(args.source_dir).resolve()
    if not docs_path.exists():
        sys.exit(f"❌  Source directory not found: {docs_path}")

    persist_path = Path(args.persist_dir).resolve()

    # ------------------------------------------------------------------
    # Either load an existing store or build a new one
    # ------------------------------------------------------------------
    if persist_path.exists() and not args.rebuild:
        print(f"📄  Loading existing store at {persist_path} …")
        storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_ctx)
    else:
        print("🛠  Building a fresh store …")
        if not list(docs_path.rglob("*.pdf")) and not list(docs_path.rglob("*.txt")) and not list(docs_path.rglob("*.docx")):
            sys.exit("❌  No source files found – add PDFs/TXTs/DOCXs or specify --source_dir.")
        raw_docs = SimpleDirectoryReader(str(docs_path)).load_data()
        index = build_index(raw_docs, persist_dir=str(persist_path))

    # ------------------------------------------------------------------
    # Incremental update: embed any new documents not yet in the store.
    # ------------------------------------------------------------------
    node_parser: SentenceSplitter = Settings.node_parser  
    docstore = index.storage_context.docstore
    ref_infos = docstore.get_all_ref_doc_info() or {}
    known_paths = {
        info.metadata.get("file_path")
        for info in ref_infos.values()
        if isinstance(info.metadata, dict) and info.metadata.get("file_path")
    }
    new_files = [
        p
        for p in docs_path.rglob("*.*")
        if p.suffix.lower() in {".pdf", ".txt", ".docx"} and str(p) not in known_paths
    ]

    if new_files:
        print(f"➕  Found {len(new_files)} new file(s); embedding …")
        new_docs = SimpleDirectoryReader(input_files=[str(p) for p in new_files]).load_data()
        new_nodes = docs_to_nodes(new_docs)
        index.insert_nodes(new_nodes)
        index.storage_context.persist(persist_dir=str(persist_path))
        print("✅  Update complete – store persisted.")
    else:
        print("✅  No new documents detected – nothing to do.")


if __name__ == "__main__":
    main()
