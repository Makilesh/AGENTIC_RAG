"""
Milvus Client for Agentic RAG System.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymilvus import (
    Collection,
    MilvusClient,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ..utils import config, get_vector_db_logger, Timer
from ..data_processing import Chunk
from .schema_manager import SchemaManager

logger = get_vector_db_logger()


class MilvusRAGClient:
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """
        Initialize the Milvus client.
        
        Args:
            host: Milvus server host. Defaults to config value.
            port: Milvus server port. Defaults to config value.
            collection_name: Collection name. Defaults to config value.
            embedding_model: Pre-loaded embedding model. If None, loads default.
        """
        self.host = host or config.milvus.host
        self.port = port or config.milvus.port
        self.collection_name = collection_name or config.milvus.collection_name
        
        # Initialize embedding model
        self.embedding_model = embedding_model or SentenceTransformer(
            config.embedding.model_name
        )
        
        # Schema manager
        self.schema_manager = SchemaManager(self.collection_name)
        
        # Client state
        self._client: Optional[MilvusClient] = None
        self._collection: Optional[Collection] = None
        self._connected = False
        
        # BM25 state for sparse vectors
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []
        
        logger.info(
            f"MilvusRAGClient initialized for {self.host}:{self.port}/{self.collection_name}"
        )
    
    def connect(self) -> bool:
        """
        Connect to Milvus server.
        
        Returns:
            True if connection successful.
        """
        try:
            uri = f"http://{self.host}:{self.port}"
            self._client = MilvusClient(uri=uri)
            
            # Also use connections for Collection API
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
                timeout=config.milvus.timeout
            )
            
            self._connected = True
            logger.info(f"Connected to Milvus at {uri}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self._connected = False
            raise
    
    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        try:
            if self._connected:
                connections.disconnect("default")
                self._connected = False
                self._client = None
                logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")
    
    def ensure_connected(self) -> None:
        """Ensure connection is established."""
        if not self._connected:
            self.connect()
    
    def create_collection(self, drop_existing: bool = False) -> bool:
        """
        Create the RAG collection with proper schema and indexes.
        
        Args:
            drop_existing: If True, drops existing collection first.
            
        Returns:
            True if collection created successfully.
        """
        self.ensure_connected()
        
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                if drop_existing:
                    logger.warning(f"Dropping existing collection '{self.collection_name}'")
                    utility.drop_collection(self.collection_name)
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    self._collection = Collection(self.collection_name)
                    return True
            
            # Get schema
            schema = self.schema_manager.get_collection_schema()
            
            # Create collection
            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
                using="default"
            )
            
            logger.info(f"Created collection '{self.collection_name}'")
            
            # Create indexes
            self._create_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create indexes for BOTH dense and sparse vectors."""
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Dense vector index (HNSW)
        dense_config = self.schema_manager.get_dense_index_config()
        self._collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": dense_config.index_type,
                "metric_type": dense_config.metric_type,
                "params": dense_config.params
            }
        )
        logger.info("Created HNSW index on dense_vector")
        
        # RE-ENABLED: Sparse vector index (Sparse Inverted Index)
        sparse_config = self.schema_manager.get_sparse_index_config()
        self._collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": sparse_config.index_type,
                "metric_type": sparse_config.metric_type,
                "params": sparse_config.params
            }
        )
        logger.info("Created SPARSE_INVERTED_INDEX on sparse_vector")
        
        # Load collection into memory
        self._collection.load()
        logger.info("Collection loaded into memory with both indexes")
    
    def _compute_dense_embedding(self, text: str) -> np.ndarray:
        """
        Compute dense embedding for text.
        
        Args:
            text: Input text.
            
        Returns:
            Normalized embedding vector.
        """
        embedding = self.embedding_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding
    
    def _compute_sparse_embedding(self, text: str) -> Dict[int, float]:
        """
        Compute BM25-style sparse embedding for keyword matching.
        
        Uses deterministic hash-based vocabulary mapping to ensure:
        1. Same token always maps to same index (even across restarts)
        2. No dependency on corpus statistics (stateless)
        3. Efficient sparse representation
        
        Args:
            text: Input text to vectorize.
            
        Returns:
            Sparse vector as {index: weight} dict.
        """
        import re
        from collections import Counter
        
        # Tokenize: lowercase, remove punctuation, split on whitespace
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [t for t in text_clean.split() if len(t) >= 2]  # Filter very short tokens
        
        if not tokens:
            return {}
        
        # BM25 parameters
        k1 = 1.5  # Term frequency saturation
        b = 0.75  # Length normalization
        avgdl = 100  # Assumed average document length
        
        # Count token frequencies
        token_counts = Counter(tokens)
        doc_len = len(tokens)
        
        # Compute BM25-style weights
        sparse_vec = {}
        for token, tf in token_counts.items():
            # Deterministic hash-based index (vocab size 65536)
            idx = abs(hash(token)) % 65536
            
            # BM25 term weight (without IDF, since we're stateless)
            # BM25 formula: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            weight = numerator / denominator
            
            sparse_vec[idx] = weight
        
        # Normalize weights to sum to 1.0 (helps with scoring)
        total_weight = sum(sparse_vec.values())
        if total_weight > 0:
            sparse_vec = {idx: w / total_weight for idx, w in sparse_vec.items()}
        
        return sparse_vec
    
    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """
        Prepare metadata for insertion.
        
        Args:
            chunk: Chunk object with metadata.
            
        Returns:
            Cleaned metadata dictionary.
        """
        metadata = dict(chunk.metadata)
        
        # Add timestamp
        metadata["created_at"] = datetime.now().isoformat()
        
        # Ensure required fields
        metadata.setdefault("chunk_id", chunk.chunk_id)
        metadata.setdefault("chunk_index", chunk.chunk_index)
        
        # Convert any non-serializable values
        for key, value in list(metadata.items()):
            if isinstance(value, (np.integer, np.floating)):
                metadata[key] = float(value)
            elif isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
        
        return metadata
    
    def insert_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> int:
        """
        Insert chunks into the collection.
        
        Args:
            chunks: List of Chunk objects to insert.
            batch_size: Number of chunks per batch.
            show_progress: Whether to show progress.
            
        Returns:
            Number of chunks inserted.
        """
        self.ensure_connected()
        
        if not self._collection:
            self.create_collection()
        
        total_inserted = 0
        
        with Timer() as timer:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Batch encode all texts at once (10-50x faster than one at a time)
                texts_in_batch = [chunk.text for chunk in batch]
                dense_vecs = self.embedding_model.encode(
                    texts_in_batch,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Prepare data with BOTH dense and sparse vectors
                data = []
                for j, chunk in enumerate(batch):
                    # Compute sparse vector for this chunk
                    sparse_vec = self._compute_sparse_embedding(chunk.text)
                    
                    # Prepare record with BOTH vector types
                    record = {
                        "dense_vector": dense_vecs[j].tolist(),
                        "sparse_vector": sparse_vec,  # RE-ENABLED for Milvus 2.4+
                        "text_content": chunk.text[:65535],  # Truncate if needed
                        "metadata_json": self._prepare_metadata(chunk)
                    }
                    data.append(record)
                
                # Insert batch
                self._collection.insert(data)
                total_inserted += len(batch)
                
                if show_progress:
                    logger.info(f"Inserted {total_inserted}/{len(chunks)} chunks")
        
        # Flush to ensure persistence
        self._collection.flush()
        
        logger.info(
            f"Inserted {total_inserted} chunks in {timer.elapsed:.2f}s"
        )
        
        return total_inserted
    
    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search using dense vectors only.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional filter expression.
            min_score: Minimum similarity score.
            
        Returns:
            List of search results with scores.
        """
        self.ensure_connected()
        
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Compute query embedding
        query_vec = self._compute_dense_embedding(query)
        
        # Search parameters
        search_params = self.schema_manager.get_search_params("dense")
        
        with Timer() as timer:
            results = self._collection.search(
                data=[query_vec.tolist()],
                anns_field="dense_vector",
                param=search_params,
                limit=top_k,
                expr=filters,
                output_fields=["text_content", "metadata_json"]
            )
        
        # Process results
        processed_results = []
        for hit in results[0]:
            if hit.score >= min_score:
                processed_results.append({
                    "id": hit.id,
                    "score": float(hit.score),
                    "text": hit.entity.get("text_content"),
                    "metadata": hit.entity.get("metadata_json", {})
                })
        
        logger.info(
            f"Dense search returned {len(processed_results)} results in {timer.elapsed_ms():.2f}ms"
        )
        
        return processed_results
    
    def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using sparse vectors only (keyword matching).
        
        RE-ENABLED for Milvus 2.4+: Now performs actual sparse vector search.
        
        Args:
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional filter expression.
            
        Returns:
            List of search results with scores.
        """
        self.ensure_connected()
        
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Compute query sparse vector
        query_sparse = self._compute_sparse_embedding(query)
        
        if not query_sparse:
            logger.warning("Empty sparse vector for query, returning no results")
            return []
        
        # Search parameters for sparse index
        search_params = self.schema_manager.get_search_params("sparse")
        
        with Timer() as timer:
            results = self._collection.search(
                data=[query_sparse],
                anns_field="sparse_vector",
                param=search_params,
                limit=top_k,
                expr=filters,
                output_fields=["text_content", "metadata_json"]
            )
        
        # Process results
        processed_results = []
        for hit in results[0]:
            processed_results.append({
                "id": hit.id,
                "score": float(hit.score),
                "text": hit.entity.get("text_content"),
                "metadata": hit.entity.get("metadata_json", {})
            })
        
        logger.info(
            f"Sparse search returned {len(processed_results)} results in {timer.elapsed_ms():.2f}ms"
        )
        
        return processed_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        self.ensure_connected()
        
        if not self._collection:
            return {"exists": False, "count": 0}
        
        # Flush to get accurate count
        self._collection.flush()
        
        return {
            "exists": True,
            "name": self.collection_name,
            "count": self._collection.num_entities,
            "loaded": True  # We always load after creation
        }
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if deletion successful.
        """
        self.ensure_connected()
        
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
            
            self._collection = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
