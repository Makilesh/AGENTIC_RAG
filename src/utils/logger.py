"""
Logging module for Agentic RAG System.

Provides structured logging with JSON metadata support for tracking
agent decisions, retrieval operations, and LLM calls.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import config


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    Formats log messages as: timestamp | level | module | message | metadata_json
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a structured string."""
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Extract metadata if present
        metadata = getattr(record, 'metadata', None)
        metadata_str = json.dumps(metadata) if metadata else "{}"
        
        # Build the log line
        log_line = f"{timestamp} | {record.levelname} | {record.module} | {record.getMessage()} | {metadata_str}"
        
        # Add exception info if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"
            
        return log_line


class AgentLogger(logging.Logger):
    """
    Custom logger with metadata support for agent operations.
    
    Extends the standard Logger to support structured metadata logging
    for tracking agent decisions, retrieval operations, and LLM calls.
    """
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize the AgentLogger."""
        super().__init__(name, level)
        
    def _log_with_metadata(
        self, 
        level: int, 
        msg: str, 
        metadata: Optional[Dict[str, Any]] = None,
        *args, 
        **kwargs
    ) -> None:
        """Log a message with optional metadata."""
        extra = kwargs.get('extra', {})
        extra['metadata'] = metadata or {}
        kwargs['extra'] = extra
        super().log(level, msg, *args, **kwargs)
    
    def info_with_metadata(
        self, 
        msg: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an INFO message with metadata."""
        self._log_with_metadata(logging.INFO, msg, metadata)
        
    def debug_with_metadata(
        self, 
        msg: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a DEBUG message with metadata."""
        self._log_with_metadata(logging.DEBUG, msg, metadata)
        
    def warning_with_metadata(
        self, 
        msg: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a WARNING message with metadata."""
        self._log_with_metadata(logging.WARNING, msg, metadata)
        
    def error_with_metadata(
        self, 
        msg: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an ERROR message with metadata."""
        self._log_with_metadata(logging.ERROR, msg, metadata)


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[str] = None
) -> AgentLogger:
    """
    Set up and return a configured logger instance.
    
    Args:
        name: Name of the logger (typically module name).
        log_file: Optional path to a log file.
        level: Log level (DEBUG, INFO, WARNING, ERROR). 
               Defaults to config setting.
               
    Returns:
        Configured AgentLogger instance.
    """
    # Register custom logger class
    logging.setLoggerClass(AgentLogger)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level
    log_level = getattr(logging, level or config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = JSONFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> AgentLogger:
    """
    Get or create a logger for the given name.
    
    Args:
        name: Name of the logger.
        
    Returns:
        AgentLogger instance.
    """
    return setup_logger(name)


# Pre-configured loggers for different components
def get_agent_logger(agent_name: str) -> AgentLogger:
    """Get a logger configured for agent operations."""
    return get_logger(f"agents.{agent_name}")


def get_data_processing_logger() -> AgentLogger:
    """Get a logger configured for data processing operations."""
    return get_logger("data_processing")


def get_vector_db_logger() -> AgentLogger:
    """Get a logger configured for vector database operations."""
    return get_logger("vector_db")


def get_llm_logger() -> AgentLogger:
    """Get a logger configured for LLM operations."""
    return get_logger("llm")


def get_workflow_logger() -> AgentLogger:
    """Get a logger configured for workflow orchestration."""
    return get_logger("workflow")


# Convenience function for logging agent decisions
def log_agent_decision(
    logger: AgentLogger,
    agent_name: str,
    input_state: Dict[str, Any],
    output_decision: Dict[str, Any],
    reasoning: str
) -> None:
    """
    Log an agent's decision with full context.
    
    Args:
        logger: The logger instance to use.
        agent_name: Name of the agent making the decision.
        input_state: The input state the agent received.
        output_decision: The decision/output the agent produced.
        reasoning: The agent's reasoning for its decision.
    """
    logger.info_with_metadata(
        f"{agent_name}: {reasoning}",
        metadata={
            "agent_name": agent_name,
            "input_state_keys": list(input_state.keys()),
            "output_decision": output_decision,
            "reasoning": reasoning
        }
    )


# Convenience function for logging retrieval operations
def log_retrieval_operation(
    logger: AgentLogger,
    query: str,
    search_strategy: str,
    num_results: int,
    top_scores: list,
    latency_ms: float
) -> None:
    """
    Log a retrieval operation with performance metrics.
    
    Args:
        logger: The logger instance to use.
        query: The search query.
        search_strategy: The strategy used (hybrid, dense, sparse).
        num_results: Number of results returned.
        top_scores: List of top similarity scores.
        latency_ms: Latency in milliseconds.
    """
    logger.info_with_metadata(
        f"Retrieval completed: {num_results} results in {latency_ms:.2f}ms",
        metadata={
            "query": query[:100],  # Truncate long queries
            "search_strategy": search_strategy,
            "num_results": num_results,
            "top_scores": top_scores[:5],
            "latency_ms": latency_ms
        }
    )


# Convenience function for logging LLM calls
def log_llm_call(
    logger: AgentLogger,
    model_used: str,
    prompt_length: int,
    response_length: int,
    latency_ms: float,
    is_fallback: bool = False
) -> None:
    """
    Log an LLM API call with performance metrics.
    
    Args:
        logger: The logger instance to use.
        model_used: The model that was used.
        prompt_length: Length of the prompt in characters.
        response_length: Length of the response in characters.
        latency_ms: Latency in milliseconds.
        is_fallback: Whether this was a fallback call.
    """
    logger.info_with_metadata(
        f"LLM call to {model_used}: {latency_ms:.2f}ms",
        metadata={
            "model": model_used,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "latency_ms": latency_ms,
            "is_fallback": is_fallback
        }
    )
