"""
Metrics module for Agentic RAG System.

Provides performance tracking and metrics collection for monitoring
system health, agent performance, and retrieval quality.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    
    query_id: str
    original_query: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Agent timings (in seconds)
    agent_timings: Dict[str, float] = field(default_factory=dict)
    
    # Retrieval metrics
    retrieval_latency: float = 0.0
    num_documents_retrieved: int = 0
    top_similarity_scores: List[float] = field(default_factory=list)
    
    # Quality metrics
    context_quality_score: float = 0.0
    confidence_score: float = 0.0
    
    # Rewrite metrics
    num_rewrites: int = 0
    quality_improvement: float = 0.0
    
    # LLM metrics
    llm_calls: int = 0
    total_llm_latency: float = 0.0
    fallback_used: bool = False
    
    def total_latency(self) -> float:
        """Calculate total query processing latency."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class DocumentMetrics:
    """Metrics for document ingestion."""
    
    file_name: str
    file_type: str
    file_size_bytes: int
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Processing metrics
    num_chunks: int = 0
    avg_chunk_size: float = 0.0
    processing_time: float = 0.0
    
    # Status
    success: bool = True
    error_message: Optional[str] = None


class MetricsCollector:
    """
    Thread-safe metrics collector for the Agentic RAG System.
    
    Collects and aggregates metrics for queries, document processing,
    and system health monitoring.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global metrics access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the metrics collector."""
        if self._initialized:
            return
            
        self._initialized = True
        self._query_metrics: List[QueryMetrics] = []
        self._document_metrics: List[DocumentMetrics] = []
        self._agent_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_time": 0.0, "avg_time": 0.0}
        )
        self._system_stats: Dict[str, Any] = {
            "total_queries": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_query_latency": 0.0,
            "avg_retrieval_quality": 0.0,
            "fallback_usage_rate": 0.0,
        }
        self._lock = threading.Lock()
    
    def start_query(self, query_id: str, query: str) -> QueryMetrics:
        """
        Start tracking metrics for a new query.
        
        Args:
            query_id: Unique identifier for the query.
            query: The original query text.
            
        Returns:
            QueryMetrics instance for this query.
        """
        metrics = QueryMetrics(
            query_id=query_id,
            original_query=query,
            start_time=datetime.now()
        )
        return metrics
    
    def complete_query(self, metrics: QueryMetrics, success: bool = True) -> None:
        """
        Complete query tracking and store metrics.
        
        Args:
            metrics: The QueryMetrics instance to complete.
            success: Whether the query was successful.
        """
        metrics.end_time = datetime.now()
        
        with self._lock:
            self._query_metrics.append(metrics)
            self._system_stats["total_queries"] += 1
            
            if success:
                self._system_stats["successful_queries"] += 1
            else:
                self._system_stats["failed_queries"] += 1
            
            # Update averages
            self._update_averages()
    
    def record_agent_timing(
        self, 
        agent_name: str, 
        execution_time: float
    ) -> None:
        """
        Record execution time for an agent.
        
        Args:
            agent_name: Name of the agent.
            execution_time: Execution time in seconds.
        """
        with self._lock:
            stats = self._agent_stats[agent_name]
            stats["count"] += 1
            stats["total_time"] += execution_time
            stats["avg_time"] = stats["total_time"] / stats["count"]
    
    def start_document_processing(
        self, 
        file_name: str, 
        file_type: str, 
        file_size_bytes: int
    ) -> DocumentMetrics:
        """
        Start tracking document processing.
        
        Args:
            file_name: Name of the file.
            file_type: Type of the file (pdf, docx, etc.).
            file_size_bytes: Size of the file in bytes.
            
        Returns:
            DocumentMetrics instance.
        """
        return DocumentMetrics(
            file_name=file_name,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            start_time=datetime.now()
        )
    
    def complete_document_processing(
        self, 
        metrics: DocumentMetrics,
        num_chunks: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Complete document processing tracking.
        
        Args:
            metrics: The DocumentMetrics instance to complete.
            num_chunks: Number of chunks created.
            success: Whether processing was successful.
            error_message: Error message if failed.
        """
        metrics.end_time = datetime.now()
        metrics.num_chunks = num_chunks
        metrics.success = success
        metrics.error_message = error_message
        metrics.processing_time = (
            metrics.end_time - metrics.start_time
        ).total_seconds()
        
        with self._lock:
            self._document_metrics.append(metrics)
            self._system_stats["total_documents"] += 1
            self._system_stats["total_chunks"] += num_chunks
    
    def _update_averages(self) -> None:
        """Update aggregate statistics (called with lock held)."""
        if self._query_metrics:
            # Average query latency
            total_latency = sum(
                m.total_latency() for m in self._query_metrics
            )
            self._system_stats["avg_query_latency"] = (
                total_latency / len(self._query_metrics)
            )
            
            # Average retrieval quality
            qualities = [
                m.context_quality_score 
                for m in self._query_metrics 
                if m.context_quality_score > 0
            ]
            if qualities:
                self._system_stats["avg_retrieval_quality"] = (
                    sum(qualities) / len(qualities)
                )
            
            # Fallback usage rate
            fallback_count = sum(
                1 for m in self._query_metrics if m.fallback_used
            )
            self._system_stats["fallback_usage_rate"] = (
                fallback_count / len(self._query_metrics)
            )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get current system statistics.
        
        Returns:
            Dictionary of system statistics.
        """
        with self._lock:
            return dict(self._system_stats)
    
    def get_agent_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get agent performance statistics.
        
        Returns:
            Dictionary of agent statistics.
        """
        with self._lock:
            return dict(self._agent_stats)
    
    def get_recent_queries(self, limit: int = 10) -> List[QueryMetrics]:
        """
        Get recent query metrics.
        
        Args:
            limit: Maximum number of queries to return.
            
        Returns:
            List of recent QueryMetrics.
        """
        with self._lock:
            return list(self._query_metrics[-limit:])
    
    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get document processing statistics.
        
        Returns:
            Dictionary of document statistics.
        """
        with self._lock:
            if not self._document_metrics:
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "avg_processing_time": 0.0,
                    "success_rate": 0.0,
                }
            
            successful = sum(1 for m in self._document_metrics if m.success)
            total_time = sum(m.processing_time for m in self._document_metrics)
            
            return {
                "total_documents": len(self._document_metrics),
                "total_chunks": sum(m.num_chunks for m in self._document_metrics),
                "avg_processing_time": total_time / len(self._document_metrics),
                "success_rate": successful / len(self._document_metrics),
                "by_type": self._get_stats_by_type(),
            }
    
    def _get_stats_by_type(self) -> Dict[str, Dict[str, Any]]:
        """Get document statistics grouped by file type."""
        by_type = defaultdict(list)
        for m in self._document_metrics:
            by_type[m.file_type].append(m)
        
        result = {}
        for file_type, metrics in by_type.items():
            result[file_type] = {
                "count": len(metrics),
                "total_chunks": sum(m.num_chunks for m in metrics),
                "avg_processing_time": sum(
                    m.processing_time for m in metrics
                ) / len(metrics),
            }
        return result


class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer() as timer:
            # code to time
        print(f"Took {timer.elapsed} seconds")
    """
    
    def __init__(self):
        """Initialize the timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        """Stop the timer and calculate elapsed time."""
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000


# Global metrics collector instance
metrics_collector = MetricsCollector()
