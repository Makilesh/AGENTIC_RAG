"""
Utility modules for Agentic RAG System.

This package provides configuration management, logging, and metrics collection.
"""

from .config import config, get_config, get_gemini_api_key, get_ollama_model
from .logger import (
    get_logger,
    get_agent_logger,
    get_data_processing_logger,
    get_vector_db_logger,
    get_llm_logger,
    get_workflow_logger,
    log_agent_decision,
    log_retrieval_operation,
    log_llm_call,
)
from .metrics import (
    metrics_collector,
    QueryMetrics,
    DocumentMetrics,
    Timer,
)


__all__ = [
    # Config
    "config",
    "get_config",
    "get_gemini_api_key",
    "get_ollama_model",
    # Logger
    "get_logger",
    "get_agent_logger",
    "get_data_processing_logger",
    "get_vector_db_logger",
    "get_llm_logger",
    "get_workflow_logger",
    "log_agent_decision",
    "log_retrieval_operation",
    "log_llm_call",
    # Metrics
    "metrics_collector",
    "QueryMetrics",
    "DocumentMetrics",
    "Timer",
]
