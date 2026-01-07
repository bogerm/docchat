# DocChat 🐥

**DocChat** is an intelligent document chat application powered by Docling, LangChain, LangGraph, and multi-agent RAG (Retrieval-Augmented Generation). It enables users to upload documents and ask questions about their content with verification and relevance checking.

## Features ✨

- **📄 Multi-format Document Support**: Process PDF, DOCX, TXT, and Markdown files
- **🔍 Hybrid Retrieval**: Combines BM25 (keyword-based) and vector-based retrieval for optimal search results
- **🤖 Multi-Agent Workflow**: 
  - **Relevance Checker**: Validates if questions are relevant to documents
  - **Research Agent**: Generates answers based on retrieved context
  - **Verification Agent**: Validates answers against source documents
- **💾 Smart Caching**: Caches processed documents for faster subsequent queries
- **🎨 Interactive UI**: Beautiful Gradio-based web interface with examples
- **🔄 Session Management**: Maintains document state across multiple queries

## Architecture

The application follows a modular architecture with the following components:

```
DocChat/
├── agents/              # Multi-agent workflow components
│   ├── workflow.py      # Main LangGraph workflow orchestrator
│   ├── research_agent.py    # Generates initial answers
│   ├── verification_agent.py # Validates answers
│   └── relevance_checker.py  # Checks question-document relevance
├── document_processor/  # Document handling and chunking
│   └── file_handler.py  # Processes documents with caching
├── retriever/          # Retrieval pipeline
│   └── builder.py      # Builds hybrid BM25 + vector retrievers
├── config/             # Configuration management
│   ├── constants.py    # Constants and file restrictions
│   ├── settings.py     # Pydantic settings with environment support
│   └── models.py       # LLM and embedding factory functions
├── utils/              # Utility functions
│   └── logging.py      # Custom logging setup
├── chroma_db/          # Vector database storage
└── document_cache/     # Cached processed documents
```

## Workflow

```
User Query + Documents
        ↓
Check Relevance
        ↓ (if relevant)
Research Agent → Generate Answer
        ↓
Verification Agent → Validate Answer
        ↓
Return Answer + Verification Report
```

1. **Relevance Check**: Questions are validated against documents
2. **Retrieval**: Hybrid retriever (BM25 + vector search) fetches relevant context
3. **Research**: LLM generates initial answer with context
4. **Verification**: Answer is validated against source documents
5. **Response**: Final answer with verification report is returned

## Setup

### Prerequisites

- Python >= 3.13
- Git

### Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd docchat
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or: source .venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies**:
```bash
pip install -e ".[dev]"
```

### Configuration

Create a `.env` file in the project root to configure API keys and settings:

```env
# IBM Watson AI (default provider)
IBM_WATSON_URL=https://us-south.ml.cloud.ibm.com
IBM_WATSON_PROJECT_ID=your-project-id
IBM_WATSON_API_KEY=your-api-key

# OpenAI (optional)
OPENAI_API_KEY=your-openai-key

# Ollama (optional, local)
OLLAMA_BASE_URL=http://localhost:11434

# Model selection (ibm_watson, openai, ollama)
DEFAULT_MODEL_PROVIDER=ibm_watson

# Specific model configurations
RESEARCH_MODEL_ID=meta-llama/llama-3-2-90b-vision-instruct
RESEARCH_MODEL_TEMPERATURE=0.3
VERIFICATION_MODEL_ID=ibm/granite-4-h-small
VERIFICATION_MODEL_TEMPERATURE=0.0

# Vector store
CHROMA_DB_PATH=./chroma_db
VECTOR_SEARCH_K=10

# Caching
CACHE_DIR=document_cache
CACHE_EXPIRE_DAYS=7

# Logging
LOG_LEVEL=INFO
```

## Running the Application

Start the DocChat web interface:

```bash
python main.py
```

The application will launch at `http://127.0.0.1:5000` and share a public URL via ngrok.

### Usage

1. **Upload Documents**: Use the file upload widget to select PDF, DOCX, TXT, or Markdown files
2. **Ask Questions**: Enter your question in the text field
3. **Load Examples** (optional): Select a pre-configured example and load it
4. **Submit**: Click "Submit 🚀" to process

The application will:
- Process and cache documents
- Retrieve relevant sections
- Generate an answer
- Verify the answer against source documents
- Display both the answer and verification report

## Built-in Examples

The application includes example questions and documents:

- **Google 2024 Environmental Report**: Query about data center efficiency metrics
- **DeepSeek-R1 Technical Report**: Summarize model performance evaluations

## Development

### Testing

Run the test suite:

```bash
pytest test/
```

Key test files:
- `test_document_processor.py` - Document processing tests
- `test_embedding_factory.py` - Embedding generation tests
- `test_model_factory.py` - Model initialization tests
- `relevance_agent_functional_test.py` - End-to-end workflow tests

### Project Dependencies

**Core Libraries**:
- **docling** - Document extraction and conversion
- **gradio** - Web UI framework
- **langchain** - LLM framework and utilities
- **langgraph** - Multi-agent workflow orchestration
- **chromadb** - Vector database for embeddings
- **rank-bm25** - BM25 retrieval algorithm

**LLM Providers**:
- **langchain-ibm** - IBM Watson AI integration
- **langchain-openai** - OpenAI integration
- **langchain-ollama** - Ollama integration
- **ibm-watsonx-ai** - IBM Watsonx platform

**Utilities**:
- **pydantic-settings** - Configuration management
- **loguru** - Advanced logging
- **pytest** - Testing framework

## Configuration Reference

### File Size Limits
- Maximum file size: Set in `config/constants.py`
- Total upload limit: Set in `config/constants.py`
- Supported formats: `.pdf`, `.docx`, `.txt`, `.md`

### Retrieval Settings
- `VECTOR_SEARCH_K`: Number of documents to retrieve (default: 10)
- `HYBRID_RETRIEVER_WEIGHTS`: [BM25_weight, Vector_weight] (default: [0.4, 0.6])

### Model Settings
- **Research Model**: Used for generating answers (default: Llama 3.2 90B)
  - Temperature: 0.3 (lower = more focused)
  - Max tokens: 3000
  
- **Verification Model**: Used for answer validation (default: Granite 4H Small)
  - Temperature: 0.0 (deterministic)
  - Max tokens: 2000

### Caching
- Documents are cached using SHA-256 hashing
- Cache validity: 7 days (configurable)
- Cache location: `./document_cache`

## Troubleshooting

### API Key Issues
- Ensure API keys are set in `.env` file
- Check the configured provider matches your credentials
- Restart the application after updating `.env`

### Document Processing Errors
- Verify file format is supported (PDF, DOCX, TXT, MD)
- Check file size limits in `config/constants.py`
- See application logs for detailed error messages

### No Relevant Documents Found
- Try rephrasing your question
- Ensure documents are uploaded
- Check the relevance checker is not filtering valid documents

### Vector Database Issues
- Delete `./chroma_db` to reset vector store
- Ensure `CHROMA_DB_PATH` is writable
- Check sufficient disk space available

## Performance Considerations

- **Caching**: Document processing is cached; subsequent queries are faster
- **Batch Processing**: Multiple documents are processed sequentially
- **Vector Store**: Uses persistent Chroma database for consistency
- **Hybrid Retrieval**: Balances keyword (BM25) and semantic (vector) search

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IBM_WATSON_PROJECT_ID` | - | IBM Watsonx project ID |
| `IBM_WATSON_API_KEY` | - | IBM Watson API key |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `DEFAULT_MODEL_PROVIDER` | `ibm_watson` | LLM provider (ibm_watson, openai, ollama) |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector database location |
| `CACHE_DIR` | `./document_cache` | Processed document cache |
| `CACHE_EXPIRE_DAYS` | `7` | Cache validity period |
| `LOG_LEVEL` | `INFO` | Logging level |

## Contributing

1. Create a new branch for features or fixes
2. Make changes and add tests
3. Run `pytest` to validate
4. Submit a pull request

## License

(Specify your license here)

## Support

For issues or questions, please create an issue on the repository or contact the development team.

---

**Built with ❤️ using Docling, LangChain, and LangGraph**
