"""
Configuration module for Agentic RAG System.

This module handles loading and managing configuration from environment variables
and YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()


class MilvusConfig(BaseModel):
    """Milvus database configuration."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=19530)
    collection_name: str = Field(default="agentic_rag_documents")
    timeout: int = Field(default=30)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    
    model_name: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384)


class LLMConfig(BaseModel):
    """LLM configuration for primary and fallback models."""
    
    primary_model: str = Field(default="gemini/gemini-2.0-flash-exp")
    fallback_model: str = Field(default="ollama/qwen2.5:14b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=2000)
    timeout: int = Field(default=30)


class ChunkingConfig(BaseModel):
    """Semantic chunking configuration."""
    
    default_chunk_size: int = Field(default=600)
    min_chunk_size: int = Field(default=200)
    max_chunk_size: int = Field(default=1000)
    overlap_tokens: int = Field(default=100)
    similarity_threshold: float = Field(default=0.5)


class SearchConfig(BaseModel):
    """Search configuration for hybrid search."""
    
    default_top_k: int = Field(default=10)
    min_similarity_score: float = Field(default=0.5)
    dense_weight: float = Field(default=0.7)
    sparse_weight: float = Field(default=0.3)
    rrf_k: int = Field(default=60)


class QualityConfig(BaseModel):
    """Quality thresholds for agent decisions."""
    
    quality_threshold: float = Field(default=0.7)
    max_rewrite_iterations: int = Field(default=2)
    confidence_threshold: float = Field(default=0.8)


class AppConfig(BaseModel):
    """Main application configuration."""
    
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    log_level: str = Field(default="INFO")
    max_file_size_mb: int = Field(default=200)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
    """
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def get_config() -> AppConfig:
    """
    Load and return the application configuration.
    
    Combines environment variables and YAML configuration files.
    Environment variables take precedence over YAML config.
    
    Returns:
        AppConfig instance with all configuration values.
    """
    # Base directory for config files
    config_dir = Path(__file__).parent.parent.parent / "config"
    
    # Load YAML configurations
    milvus_yaml = load_yaml_config(config_dir / "milvus_config.yaml")
    llm_yaml = load_yaml_config(config_dir / "llm_config.yaml")
    
    # Build configuration from environment variables (priority) and YAML
    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", milvus_yaml.get("connection", {}).get("host", "localhost")),
        port=int(os.getenv("MILVUS_PORT", milvus_yaml.get("connection", {}).get("port", 19530))),
        collection_name=os.getenv(
            "MILVUS_COLLECTION_NAME", 
            milvus_yaml.get("collection", {}).get("name", "agentic_rag_documents")
        ),
        timeout=milvus_yaml.get("connection", {}).get("timeout", 30)
    )
    
    embedding_config = EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", 384))
    )
    
    llm_config = LLMConfig(
        primary_model=llm_yaml.get("primary", {}).get("model", "gemini/gemini-2.0-flash-exp"),
        fallback_model=llm_yaml.get("fallback", {}).get("model", "ollama/qwen2.5:14b"),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", 
            llm_yaml.get("fallback", {}).get("base_url", "http://localhost:11434")
        ),
        temperature=llm_yaml.get("primary", {}).get("parameters", {}).get("temperature", 0.1),
        max_tokens=llm_yaml.get("primary", {}).get("parameters", {}).get("max_tokens", 2000),
        timeout=llm_yaml.get("primary", {}).get("timeout", 30)
    )
    
    chunking_config = ChunkingConfig(
        default_chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", 600)),
        min_chunk_size=int(os.getenv("MIN_CHUNK_SIZE", 200)),
        overlap_tokens=int(os.getenv("CHUNK_OVERLAP", 100)),
        similarity_threshold=0.5
    )
    
    search_config = SearchConfig(
        default_top_k=int(os.getenv("DEFAULT_TOP_K", 10)),
        min_similarity_score=float(os.getenv("MIN_SIMILARITY_SCORE", 0.5)),
        dense_weight=milvus_yaml.get("search", {}).get("hybrid", {}).get("dense_weight", 0.7),
        sparse_weight=milvus_yaml.get("search", {}).get("hybrid", {}).get("sparse_weight", 0.3),
        rrf_k=milvus_yaml.get("search", {}).get("hybrid", {}).get("rrf_k", 60)
    )
    
    quality_config = QualityConfig(
        quality_threshold=float(os.getenv("QUALITY_THRESHOLD", 0.7)),
        max_rewrite_iterations=int(os.getenv("MAX_REWRITE_ITERATIONS", 2)),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
    )
    
    return AppConfig(
        milvus=milvus_config,
        embedding=embedding_config,
        llm=llm_config,
        chunking=chunking_config,
        search=search_config,
        quality=quality_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", 200))
    )


# Global configuration instance
config = get_config()


def get_gemini_api_key() -> Optional[str]:
    """Get the Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")


def get_ollama_model() -> str:
    """Get the Ollama model name from environment variables."""
    return os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
