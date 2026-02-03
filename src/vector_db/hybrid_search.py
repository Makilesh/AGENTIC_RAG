"""
Hybrid Search module for Agentic RAG System.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ..utils import config, get_vector_db_logger, Timer, log_retrieval_operation
from .milvus_client import MilvusRAGClient

logger = get_vector_db_logger()


@dataclass
class SearchResult:
    """
    Represents a single search result.
    
    Attributes:
        id: Document ID in Milvus.
        text: Text content of the chunk.
        score: Combined or individual score.
        dense_score: Score from dense search (if available).
        sparse_score: Score from sparse search (if available).
        rrf_score: Reciprocal Rank Fusion score (if hybrid).
        metadata: Document metadata.
        rank: Final rank in result list.
    """
    id: int
    text: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rrf_score": self.rrf_score,
            "metadata": self.metadata or {},
            "rank": self.rank
        }


@dataclass
class SearchResponse:
    """
    Complete search response.
    
    Attributes:
        results: List of SearchResult objects.
        query: Original search query.
        search_strategy: Strategy used (dense, sparse, hybrid).
        latency_ms: Search latency in milliseconds.
        total_results: Total number of results.
    """
    results: List[SearchResult]
    query: str
    search_strategy: str
    latency_ms: float
    total_results: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "search_strategy": self.search_strategy,
            "latency_ms": self.latency_ms,
            "total_results": self.total_results
        }


class HybridSearcher:
    """
    Hybrid search implementation combining dense and sparse vectors.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from
    semantic search (dense) and keyword search (sparse).
    
    Attributes:
        client: MilvusRAGClient instance.
        rrf_k: RRF constant (default 60).
        dense_weight: Weight for dense search in hybrid mode.
        sparse_weight: Weight for sparse search in hybrid mode.
    """
    
    def __init__(
        self,
        client: MilvusRAGClient,
        rrf_k: int = 60,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize the hybrid searcher.
        
        Args:
            client: MilvusRAGClient instance.
            rrf_k: RRF constant (typically 60).
            dense_weight: Weight for dense search results.
            sparse_weight: Weight for sparse search results.
        """
        self.client = client
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        logger.info(
            f"HybridSearcher initialized with rrf_k={rrf_k}, "
            f"dense_weight={dense_weight}, sparse_weight={sparse_weight}"
        )
    
    def _compute_rrf_score(
        self,
        dense_rank: Optional[int],
        sparse_rank: Optional[int]
    ) -> float:
        """
        Compute Reciprocal Rank Fusion score.
        
        RRF_score = sum(1 / (k + rank)) for each ranking where document appears.
        
        Args:
            dense_rank: Rank in dense search results (1-indexed, None if not present).
            sparse_rank: Rank in sparse search results (1-indexed, None if not present).
            
        Returns:
            Combined RRF score.
        """
        score = 0.0
        
        if dense_rank is not None:
            score += self.dense_weight * (1.0 / (self.rrf_k + dense_rank))
        
        if sparse_rank is not None:
            score += self.sparse_weight * (1.0 / (self.rrf_k + sparse_rank))
        
        return score
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[str] = None,
        min_score: float = 0.0
    ) -> SearchResponse:
        """
        Perform TRUE hybrid search combining dense and sparse vectors with RRF.
        
        RE-ENABLED for Milvus 2.4+: Now uses actual RRF fusion of dense and sparse results.
        
        Args:
            query: Search query text.
            top_k: Number of final results to return.
            filters: Optional Milvus filter expression.
            min_score: Minimum score threshold for results.
            
        Returns:
            SearchResponse with RRF-fused results.
        """
        with Timer() as timer:
            # Execute BOTH searches (dense semantic + sparse keyword)
            # Retrieve more results for better fusion quality
            fetch_k = min(top_k * 2, 50)
            
            # Dense search (semantic similarity)
            dense_results = self.client.search_dense(
                query=query,
                top_k=fetch_k,
                filters=filters,
                min_score=0.0  # No filtering yet, let RRF decide
            )
            
            # Sparse search (keyword matching)
            sparse_results = self.client.search_sparse(
                query=query,
                top_k=fetch_k,
                filters=filters
            )
            
            # Build rank maps for RRF computation
            dense_rank_map = {}
            dense_data_map = {}
            for i, r in enumerate(dense_results, 1):
                dense_rank_map[r["id"]] = (i, r["score"])
                dense_data_map[r["id"]] = r
            
            sparse_rank_map = {}
            sparse_data_map = {}
            for i, r in enumerate(sparse_results, 1):
                sparse_rank_map[r["id"]] = (i, r["score"])
                sparse_data_map[r["id"]] = r
            
            # Get all unique document IDs from both searches
            all_doc_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())
            
            # Compute RRF scores for all documents
            rrf_results = []
            for doc_id in all_doc_ids:
                # Get ranks (None if not in that search)
                dense_rank = dense_rank_map.get(doc_id, (None, 0))[0]
                sparse_rank = sparse_rank_map.get(doc_id, (None, 0))[0]
                
                # Compute RRF score
                rrf_score = self._compute_rrf_score(dense_rank, sparse_rank)
                
                # Get document data (prefer dense, fallback to sparse)
                doc_data = dense_data_map.get(doc_id) or sparse_data_map.get(doc_id)
                
                # Create SearchResult with all score components
                result = SearchResult(
                    id=doc_id,
                    text=doc_data["text"],
                    score=rrf_score,  # Use RRF as primary score
                    dense_score=dense_rank_map.get(doc_id, (None, 0))[1],
                    sparse_score=sparse_rank_map.get(doc_id, (None, 0))[1],
                    rrf_score=rrf_score,
                    metadata=doc_data["metadata"],
                    rank=0  # Will be set after sorting
                )
                
                # Apply min_score filter on RRF score
                if result.score >= min_score:
                    rrf_results.append(result)
            
            # Sort by RRF score descending
            rrf_results.sort(key=lambda x: x.score, reverse=True)
            
            # Take top_k and assign final ranks
            final_results = rrf_results[:top_k]
            for rank, result in enumerate(final_results, 1):
                result.rank = rank
        
        # Log the operation with detailed metrics
        logger.info(
            f"Hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(final_results)} fused results in {timer.elapsed_ms():.2f}ms"
        )
        
        return SearchResponse(
            results=final_results,
            query=query,
            search_strategy="hybrid",  # Now truly hybrid!
            latency_ms=timer.elapsed_ms(),
            total_results=len(final_results)
        )
    
    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 10,
        filters: Optional[str] = None,
        min_score: float = 0.0
    ) -> SearchResponse:
        """
        Perform search with specified strategy.
        
        Args:
            query: Search query text.
            strategy: Search strategy ('hybrid', 'dense', 'sparse').
            top_k: Number of results to return.
            filters: Optional Milvus filter expression.
            min_score: Minimum score threshold.
            
        Returns:
            SearchResponse with search results.
        """
        # Hybrid search uses RRF fusion
        if strategy == "hybrid":
            return self.search_hybrid(query, top_k, filters, min_score)
        
        # Direct dense or sparse search
        with Timer() as timer:
            if strategy == "dense":
                results = self.client.search_dense(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    min_score=min_score
                )
            elif strategy == "sparse":
                results = self.client.search_sparse(
                    query=query,
                    top_k=top_k,
                    filters=filters
                )
            else:
                raise ValueError(f"Unknown search strategy: {strategy}")
            
            # Convert to SearchResult objects
            search_results = []
            for rank, result in enumerate(results, 1):
                sr = SearchResult(
                    id=result["id"],
                    text=result["text"],
                    score=result["score"],
                    dense_score=result["score"] if strategy == "dense" else None,
                    sparse_score=result["score"] if strategy == "sparse" else None,
                    metadata=result["metadata"],
                    rank=rank
                )
                search_results.append(sr)
        
        # Log the operation
        log_retrieval_operation(
            logger,
            query=query,
            search_strategy=strategy,
            num_results=len(search_results),
            top_scores=[r.score for r in search_results[:5]],
            latency_ms=timer.elapsed_ms()
        )
        
        return SearchResponse(
            results=search_results,
            query=query,
            search_strategy=strategy,
            latency_ms=timer.elapsed_ms(),
            total_results=len(search_results)
        )
    
    def search_with_metadata_filter(
        self,
        query: str,
        source_type: Optional[str] = None,
        file_name: Optional[str] = None,
        strategy: str = "hybrid",
        top_k: int = 10
    ) -> SearchResponse:
        """
        Search with common metadata filters.
        
        Args:
            query: Search query text.
            source_type: Filter by source type (pdf, docx, etc.).
            file_name: Filter by file name.
            strategy: Search strategy.
            top_k: Number of results.
            
        Returns:
            SearchResponse with filtered results.
        """
        # Build filter expression
        filter_parts = []
        
        if source_type:
            filter_parts.append(f'metadata_json["source_type"] == "{source_type}"')
        
        if file_name:
            filter_parts.append(f'metadata_json["file_name"] == "{file_name}"')
        
        filters = " && ".join(filter_parts) if filter_parts else None
        
        return self.search(
            query=query,
            strategy=strategy,
            top_k=top_k,
            filters=filters
        )


def create_hybrid_searcher(client: Optional[MilvusRAGClient] = None) -> HybridSearcher:
    """
    Create a HybridSearcher with default configuration.
    
    Args:
        client: Optional MilvusRAGClient. Creates new one if None.
        
    Returns:
        Configured HybridSearcher instance.
    """
    if client is None:
        client = MilvusRAGClient()
        client.connect()
    
    return HybridSearcher(
        client=client,
        rrf_k=config.search.rrf_k,
        dense_weight=config.search.dense_weight,
        sparse_weight=config.search.sparse_weight
    )
