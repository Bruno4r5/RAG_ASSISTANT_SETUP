"""
-------------
Streamlit front-end for our ESG RAG assistant.

* Loads a persistent llama-index JSON store from `./vector_store`.
* If the store doesn't exist, it builds one from `./data`.
* Users can upload PDFs/TXTs/DOCXs at runtime; the app chunks &
  embeds them on-the-fly, inserts new nodes, and re-persists the store.
* Chat UI shows answers with citations.

Run locally: streamlit run src/rag_app.py

"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List

from ingest_docs import docs_to_nodes

import streamlit as st
import yaml
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI



# ---------------------------------------------------------------------------
# Helper utilities (mirrors ingest_docs.py)
# ---------------------------------------------------------------------------

def load_openai_config(config_path: str = "config/openai_config_template.yaml") -> None:
    """Populate env-vars from YAML if present (falls back to .env)."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        os.environ.setdefault("OPENAI_API_KEY", cfg.get("openai_api_key", ""))
        if cfg.get("base_url"):
            os.environ.setdefault("OPENAI_BASE_URL", cfg["base_url"])


def configure_settings(chunk_size: int = 1024, chunk_overlap: int = 50) -> None:
    """Configure global llama-index Settings."""
    # Add a system prompt that focuses on the selected document only
    system_prompt = (
        "You are an ESG assistant expert. "
        "Answer **only** with information found in the provided context. "
        "Cite the page number after every sentence (e.g., [p 12]). "
        "If the context is insufficient, reply with 'I don't know'."
    )
    
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = OpenAI("gpt-4o-mini", temperature=0, system_prompt=system_prompt)
    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

# ---------------------------------------------------------------------------
# Index loading / building
# ---------------------------------------------------------------------------

PERSIST_DIR = Path("vector_store").resolve()
DATA_DIR = Path("data").resolve()
UPLOAD_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Clear all cached resources
st.cache_resource.clear()

@st.cache_resource(show_spinner=False)
def get_index(persist_exists: bool) -> VectorStoreIndex:
    load_openai_config()
    configure_settings()
    if persist_exists:
        storage_ctx = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        return load_index_from_storage(storage_ctx)
    else:
        with st.spinner("Building initial index …"):
            raw_docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()
            nodes = docs_to_nodes(raw_docs)           
            index = VectorStoreIndex(nodes, show_progress=True)
            index.storage_context.persist(persist_dir=str(PERSIST_DIR))
            return index


# ---------------------------------------------------------------------------
# Upload handling
# ---------------------------------------------------------------------------

def handle_upload(files: List[st.runtime.uploaded_file_manager.UploadedFile], index: VectorStoreIndex):
    if not files:
        return

    parser: SentenceSplitter = Settings.node_parser  
    new_paths: List[str] = []

    for f in files:
        dest = UPLOAD_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        new_paths.append(str(dest))

    with st.spinner(f"Embedding {len(new_paths)} file(s) …"):
        new_docs = SimpleDirectoryReader(input_files=new_paths).load_data()
        nodes = docs_to_nodes(new_docs)
        index.insert_nodes(nodes)
        index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    st.success("File(s) added to knowledge base!", icon="✅")


# Debug function to inspect metadata in nodes
def inspect_index_metadata(index):
    docstore = index.storage_context.docstore
    all_docs = docstore.docs
    metadata_samples = []
    
    for doc_id, doc in list(all_docs.items())[:50]:  
        if hasattr(doc, 'metadata') and doc.metadata:
            metadata_samples.append({
                'doc_id': str(doc_id)[:10] + "...",
                'file_name': doc.metadata.get('file_name', 'Unknown'),
                'file_path': doc.metadata.get('file_path', 'Unknown'),
                'page': doc.metadata.get('page', 'Unknown')
            })
    
    return metadata_samples

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ESG Assistant", page_icon="🌱", layout="wide")
st.title("🌱 ESG RAG Assistant")

# Index load
index = get_index(PERSIST_DIR.exists())

# Sidebar – upload new docs and select file
with st.sidebar:
    st.header("➕ Add documents")
    uploaded = st.file_uploader(
        "Upload PDFs, TXTs, DOCXs",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )
    if st.button("Add to index", disabled=not uploaded):
        handle_upload(uploaded, index)

    # Build the list of files with their paths
    path_to_file_map = {}
    for dir_path in [DATA_DIR, UPLOAD_DIR]:
        if dir_path.exists():
            for file_path in dir_path.glob("*.*"):
                if file_path.is_file():
                    # Store full path with a display name that shows the directory
                    display_name = f"{file_path.name} ({dir_path.name})"
                    path_to_file_map[display_name] = str(file_path)
    
    selected_display = st.selectbox("Select document to query", sorted(path_to_file_map.keys()))
    selected_path = path_to_file_map.get(selected_display, "")
    
    # Debug checkbox
    show_debug = st.checkbox("Show debug info")

# Debug information about metadata
if "show_debug" in locals() and show_debug:
    st.subheader("Debug Information")
    metadata_samples = inspect_index_metadata(index)
    st.write("Sample metadata in index:")
    st.json(metadata_samples)

# Create a properly filtered query engine for the selected document
if selected_path:
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        metadata_filter_dict={"file_path": selected_path}
    )
    
    # Extract just the filename for display
    selected_filename = Path(selected_path).name
    
    # Add indicator of which document is being queried
    st.caption(f"Currently querying: **{selected_filename}** (Full path: {selected_path})")
else:
    st.warning("No document selected. Please select a document to query.")
    query_engine = None

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Ask a sustainability question…")
if prompt and query_engine: 
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking …"):
        response = query_engine.query(prompt)

    answer = response.response
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Enhanced source display to show filtering is working
    with st.expander("Sources"):
        if not response.source_nodes:
            st.warning("No sources found in the selected document.")
        else:
            for i, sn in enumerate(response.source_nodes, 1):
                file_name = sn.metadata.get("file_name") or Path(sn.metadata.get("file_path", "?")).name
                page = sn.metadata.get("page")
                title = f"{file_name} – page {page}" if page is not None else file_name
                st.markdown(f"**{i}.** *{title}*  ")
                st.write(sn.node.text.strip()[:1000] + ("…" if len(sn.node.text) > 1000 else ""))