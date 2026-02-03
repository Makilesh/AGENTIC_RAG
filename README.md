# Agentic RAG System 

**Production-grade Retrieval Augmented Generation with Multi-Agent Orchestration**

An intelligent document Q&A system featuring semantic chunking, hybrid search, self-corrective mechanisms, and LangGraph-based workflow orchestration.

##  Key Features

- **Multi-Agent Architecture**: 7 specialized agents orchestrated via LangGraph
- **Self-Corrective Loop**: Automatic query rewriting when retrieval quality is low
- **Semantic Chunking**: Embedding-based chunking (not naive fixed-size)
- **True Hybrid Search**: Dense (semantic) + Sparse (keyword) vectors with Reciprocal Rank Fusion
  - Dense vectors (384-dim): Semantic similarity via sentence-transformers
  - Sparse vectors (BM25-style): Keyword matching with deterministic vocabulary
  - RRF fusion: Combines both signals for optimal retrieval quality
- **Multi-Format Support**: PDF, DOCX, PPTX, Excel, TXT
- **Table-to-Text**: Sophisticated Excel processing with natural language descriptions
- **LLM Fallback**: Gemini → Ollama automatic failover

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Query                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Streamlit Interface                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              LangGraph Orchestrator (State Machine)             │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐       │
│  │  Query   │Retrieval │Retrieval │ Quality  │  Query   │       │
│  │ Analyzer │  Router  │ Executor │ Assessor │ Rewriter │       │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘       │
│       │          │          │          │          │             │
│       └──────────┴──────────┴──────────┴──────────┘             │
│                         ▲        │                              │
│                         │        ▼                              │
│  ┌─────────────────┬────┴────────────────┐                      │
│  │     Answer      │      Validator      │                      │
│  │   Synthesizer   │                     │                      │
│  └─────────────────┴─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          Milvus Vector DB (HNSW + Sparse Inverted Index)        │
│                Dense + Sparse Vectors with RRF                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Requirements

- Python 3.10+
- Docker & Docker Compose
- Ollama (for fallback LLM)
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone <repo_url>
cd agentic_rag_system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Start Milvus (Docker)

```bash
cd docker
docker-compose up -d milvus etcd minio
cd ..
```

### Start Ollama (Optional Fallback)

```bash
ollama pull qwen2.5:14b
ollama serve
```

### Run Application

```bash
streamlit run app/streamlit_app.py
```

Access at: http://localhost:8501

## 📁 Project Structure

```
agentic_rag_system/
├── src/
│   ├── agents/           # 7 specialized agents
│   ├── data_processing/  # Document processors & chunking
│   ├── vector_db/        # Milvus client & hybrid search
│   ├── llm/              # LiteLLM wrapper & prompts
│   ├── workflow/         # LangGraph orchestrator
│   └── utils/            # Config, logging, metrics
├── app/
│   ├── streamlit_app.py  # Main UI
│   └── components/       # UI components
├── config/               # YAML configurations
├── docker/               # Docker files
└── data/                 # Sample documents
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `MILVUS_HOST` | Milvus server host | localhost |
| `MILVUS_PORT` | Milvus server port | 19530 |
| `OLLAMA_BASE_URL` | Ollama API URL | http://localhost:11434 |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |

## 🧪 Sample Queries

1. "What is the Q4 revenue for North America?"
2. "Summarize the main findings from the research report"
3. "What are the key recommendations in slide 3?"
4. "Compare revenue across all regions"

## Agent Workflow

| Agent | Responsibility |
|-------|---------------|
| **Query Analyzer** | Parse intent, extract entities, classify query |
| **Retrieval Router** | Decide indices and search strategy |
| **Retrieval Executor** | Execute hybrid search against Milvus |
| **Quality Assessor** | Evaluate context relevance (0-1 score) |
| **Query Rewriter** | Rewrite query if quality < 0.7 (max 2 times) |
| **Answer Synthesizer** | Generate answer with citations |
| **Validator** | Final quality check for hallucinations |

## Self-Corrective Loop

```
Query → Retrieve → Assess Quality
         ↑              ↓
         │         Score < 0.7?
         │              ↓ Yes
         └──── Rewrite Query (max 2x)
                        ↓ No / Max reached
                   Synthesize Answer
```

## Technology Stack

- **Framework**: LangGraph
- **Vector DB**: Milvus 2.4+ (HNSW + Sparse Inverted Index)
- **LLM**: Gemini 2.5 Flash + Ollama Qwen 2.5
- **Embeddings**: Sentence Transformers (dense) + BM25-style (sparse)
- **UI**: Streamlit
- **Document Processing**: PyMuPDF, python-docx, python-pptx, pandas

## License

MIT License

## Acknowledgments

Built for Vegam AI Engineer Assignment demonstrating sophisticated agentic AI workflows.
Thank you for the oppurtunity
