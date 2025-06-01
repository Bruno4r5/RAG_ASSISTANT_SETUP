# ESG-ASSISTANT-RAG-SETUP-
# 🌱 ESG RAG Assistant

A Streamlit-based Retrieval-Augmented Generation (RAG) application for querying documents using LlamaIndex and OpenAI's GPT models.

## ⚠️ Project Status

This is a work-in-progress project and is not considered finished. The application is functional but still has some known issues that need to be addressed:

- **Document filtering**: The system should allow querying only the selected document, but the filtering mechanism still has some troubles and needs to be solved
- **Incremental improvements**: Various features and optimizations are ongoing
- **No timeline for completion**: This project is developed as needed with no specific completion date

## 🚀 Features

- **Persistent Vector Store**: Uses LlamaIndex with JSON-based storage (Not in the repo but will be created by running the codes)
- **Multi-format Support**: Handles PDFs, TXT, and DOCX files
- **Real-time Document Upload**: Add new documents through the web interface
- **Document-specific Querying**: Select specific documents to query (⚠️ *filtering currently has issues*)
- **Citation Support**: Answers include page number citations
- **Incremental Updates**: Only processes new documents when rebuilding the index

## 🛠️ Installation

1. Clone the repository:
```
git clone <your-repo-url>
cd esg-rag-assistant
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your OpenAI API configuration:
   - Add your OpenAI API key:
   ```yaml
   openai_api_key: "your-api-key-here"
   ```

4. Add your ESG documents to the `data/` folder (PDFs, TXT, or DOCX files)

## 🚦 Usage

### Running the Streamlit App

```
streamlit run src/rag_app.py
```

The app will:
- Load existing vector store from `./vector_store` (if available)
- Build a new index from `./data` folder (if no existing store)
- Provide a web interface for document upload and querying

### Building/Updating the Index Manually

```
python src/ingest_docs.py \
  --source_dir ./data \
  --persist_dir ./vector_store \
  --chunk_size 1024 \
  --chunk_overlap 50
```

Options:
- `--rebuild`: Force rebuild the entire index
- `--chunk_size`: Size of text chunks (default: 1024)
- `--chunk_overlap`: Overlap between chunks (default: 50)

## 📁 Project Structure

```
├── src/
│   ├── rag_app.py          # Main Streamlit application
│   └── ingest_docs.py      # Document ingestion and index building
├── data/                   # Place your documents here
├── vector_store/          # Persistent index storage (auto-generated)
├── config/
│   └── openai_config_template.yaml
└── requirements.txt
```

## 🔧 Configuration

### LLM Settings
- **Model**: GPT-4o-mini (configurable in code)
- **Embedding**: text-embedding-3-small
- **Temperature**: 0 (for consistent responses)
- **System Prompt**: Focused on expertise with citation requirements

### Chunking Parameters
- **Default Chunk Size**: 1024 tokens
- **Default Overlap**: 50 tokens
- Configurable via command line or code modification

## 🐛 Known Issues

### Document Filtering Problems
The most significant known issue is with document-specific querying:

- **Expected Behavior**: When a user selects a specific document in the sidebar, queries should only return results from that document
- **Current Issue**: The filtering mechanism doesn't work reliably, and results may include content from other documents in the index
- **Impact**: Users may receive answers that mix content from multiple documents when they expect results from only the selected document

### Potential Causes
- Metadata inconsistencies in the vector store
- Filter implementation issues in the query engine

### Workarounds
- Use the debug mode to inspect metadata
- Be aware that results may come from multiple documents
- Consider the source information in the expandable "Sources" section


## 🤝 Contributing

This project is in active development. If you'd like to contribute:

1. Focus on the document filtering issue as it's the main priority
2. Test thoroughly with multiple documents
3. Document any issues or improvements you find
4. Submit pull requests with clear descriptions


## ⚡ Quick Start Example

1. Add some PDFs reports to the `data/` folder
2. Run `streamlit run src/rag_app.py`
3. Select a document from the sidebar dropdown
4. Ask questions like:
   - "What are the main sustainability goals mentioned?"
   - "What carbon reduction targets are set?"
   - "How does the company approach social responsibility?"

**Note**: Remember that document filtering currently has issues, so verify the sources of your answers.

---

*This project is provided as-is for learning and experimentation purposes. Use in production environments is not recommended until the filtering issues are resolved.*
