"""Retrieval Executor Agent."""

from typing import Any, Dict, List, Optional
from ..vector_db import HybridSearcher, SearchResult
from ..utils import get_agent_logger, log_agent_decision, Timer

logger = get_agent_logger("retrieval_executor")


class RetrievalExecutorAgent:
    
    def __init__(self, searcher: HybridSearcher):
        self.searcher = searcher
        self.name = "Retrieval Executor"
    
    def execute(
        self,
        query: str,
        retrieval_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute search based on retrieval plan.
        
        Returns:
            List of retrieved documents with scores and metadata.
        """
        strategy = retrieval_plan.get("search_strategy", "hybrid")
        top_k = retrieval_plan.get("top_k", 10)
        filters = retrieval_plan.get("metadata_filters", {})
        
        # Build filter expression
        filter_expr = None
        if filters.get("source_type"):
            filter_expr = f'metadata_json["source_type"] == "{filters["source_type"]}"'
        
        with Timer() as timer:
            try:
                response = self.searcher.search(
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    filters=filter_expr,
                    min_score=0.0  # RRF scores are small (0.01-0.02), don't filter here
                )
                
                results = [
                    {
                        "id": r.id,
                        "text": r.text,
                        "score": r.score,
                        "metadata": r.metadata,
                        "rank": r.rank
                    }
                    for r in response.results
                ]
                
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                results = []
        
        log_agent_decision(
            logger, self.name,
            {"query": query[:50], "strategy": strategy},
            {"num_results": len(results), "top_score": results[0]["score"] if results else 0},
            f"Retrieved {len(results)} documents in {timer.elapsed_ms():.0f}ms"
        )
        
        return results
