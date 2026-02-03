FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --timeout 120 \
    pydantic>=2.5.0 \
    python-dotenv>=1.0.0 \
    PyYAML>=6.0.0 \
    tenacity>=8.2.0 \
    aiohttp>=3.9.0

RUN pip install --no-cache-dir --timeout 300 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 180 \
    sentence-transformers>=2.2.0 \
    tiktoken>=0.5.0 \
    numpy>=1.24.0

RUN pip install --no-cache-dir --timeout 120 \
    PyMuPDF>=1.23.0 \
    python-docx>=1.1.0 \
    python-pptx>=0.6.23 \
    pandas>=2.1.0 \
    openpyxl>=3.1.0

RUN pip install --no-cache-dir --timeout 180 \
    pymilvus>=2.3.0 \
    litellm>=1.20.0 \
    rank-bm25>=0.2.2

RUN pip install --no-cache-dir --timeout 180 \
    langgraph>=0.0.40 \
    langchain>=0.1.0 \
    langchain-community>=0.0.20

RUN pip install --no-cache-dir --timeout 120 \
    streamlit>=1.30.0 \
    fastapi>=0.108.0 \
    uvicorn>=0.25.0

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/

RUN mkdir -p data/sample_documents data/processed

ENV PYTHONPATH=/app:/app/src

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
