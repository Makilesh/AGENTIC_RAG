"""
Milvus Schema Manager for Agentic RAG System.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

from ..utils import config, get_vector_db_logger

logger = get_vector_db_logger()


@dataclass
class IndexConfig:
    """Configuration for vector index."""
    index_type: str
    metric_type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    """Configuration for a Milvus collection."""
    name: str
    description: str
    dense_dim: int
    enable_sparse: bool = True
    max_text_length: int = 65535


class SchemaManager:
    """
    Manages Milvus collection schemas for the RAG system.
    
    Handles creation, validation, and management of collection schemas
    with support for dense vectors, sparse vectors, and JSON metadata.
    """
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the schema manager.
        
        Args:
            collection_name: Name of the collection. Defaults to config value.
        """
        self.collection_name = collection_name or config.milvus.collection_name
        self.dense_dim = config.embedding.dimension
        
        logger.info(
            f"SchemaManager initialized for collection '{self.collection_name}' "
            f"with dimension {self.dense_dim}"
        )
    
    def get_field_schemas(self) -> List[FieldSchema]:
        """
        Get the field schemas for the collection.
        
        Returns:
            List of FieldSchema objects defining the collection structure.
        """
        fields = [
            # Primary key
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Auto-generated primary key"
            ),
            
            # Dense vector for semantic search
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim,
                description="Dense embedding vector for semantic search"
            ),
            
            # Sparse vector for keyword search (RE-ENABLED for Milvus 2.4+)
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="Sparse BM25-style vector for keyword matching"
            ),
            
            # Text content
            FieldSchema(
                name="text_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Original text content of the chunk"
            ),
            
            # JSON metadata
            FieldSchema(
                name="metadata_json",
                dtype=DataType.JSON,
                description="Flexible metadata storage as JSON"
            ),
        ]
        
        return fields
    
    def get_collection_schema(self) -> CollectionSchema:
        """
        Get the complete collection schema.
        
        Returns:
            CollectionSchema object for the RAG collection.
        """
        fields = self.get_field_schemas()
        
        schema = CollectionSchema(
            fields=fields,
            description="Agentic RAG document embeddings with hybrid search support",
            enable_dynamic_field=False
        )
        
        return schema
    
    def get_dense_index_config(self) -> IndexConfig:
        """
        Get the HNSW index configuration for dense vectors.
        
        Returns:
            IndexConfig for the dense vector field.
        """
        return IndexConfig(
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": 16,  # Number of bi-directional links
                "efConstruction": 200,  # Size of dynamic candidate list during construction
            }
        )
    
    def get_sparse_index_config(self) -> IndexConfig:
        """
        Get the index configuration for sparse vectors.
        
        Returns:
            IndexConfig for the sparse vector field.
        """
        return IndexConfig(
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",  # Inner product for sparse vectors
            params={
                "drop_ratio_build": 0.2,  # Ratio of small values to drop during build
            }
        )
    
    def get_search_params(self, search_type: str = "dense") -> Dict[str, Any]:
        """
        Get search parameters for a specific search type.
        
        Args:
            search_type: Type of search ('dense', 'sparse', or 'hybrid').
            
        Returns:
            Dictionary of search parameters.
        """
        if search_type == "dense":
            return {
                "metric_type": "COSINE",
                "params": {
                    "ef": 100,  # Size of dynamic candidate list during search
                }
            }
        elif search_type == "sparse":
            return {
                "metric_type": "IP",
                "params": {
                    "drop_ratio_search": 0.2,
                }
            }
        else:  # hybrid
            return {
                "dense": self.get_search_params("dense"),
                "sparse": self.get_search_params("sparse"),
                "rerank": {
                    "type": "rrf",
                    "k": config.search.rrf_k,
                }
            }
    
    def validate_schema(self, existing_schema: CollectionSchema) -> bool:
        """
        Validate that an existing schema matches expected structure.
        
        Args:
            existing_schema: The schema to validate.
            
        Returns:
            True if schema is valid, False otherwise.
        """
        expected_fields = {
            "id": DataType.INT64,
            "dense_vector": DataType.FLOAT_VECTOR,
            "sparse_vector": DataType.SPARSE_FLOAT_VECTOR,
            "text_content": DataType.VARCHAR,
            "metadata_json": DataType.JSON,
        }
        
        for field in existing_schema.fields:
            if field.name in expected_fields:
                if field.dtype != expected_fields[field.name]:
                    logger.warning(
                        f"Field '{field.name}' has unexpected type: "
                        f"expected {expected_fields[field.name]}, got {field.dtype}"
                    )
                    return False
                del expected_fields[field.name]
        
        if expected_fields:
            logger.warning(f"Missing fields: {list(expected_fields.keys())}")
            return False
        
        return True


def get_metadata_structure() -> Dict[str, str]:
    """
    Get the expected metadata structure documentation.
    
    Returns:
        Dictionary describing metadata fields and their types.
    """
    return {
        "source_type": "str - Document type (pdf, docx, pptx, excel, txt)",
        "file_name": "str - Original file name",
        "file_path": "str - Full path to source file",
        "chunk_id": "str - Unique chunk identifier",
        "chunk_index": "int - Position index in document",
        "total_chunks": "int - Total chunks from this document",
        "parent_chunk_id": "str - For hierarchical retrieval (optional)",
        "semantic_section": "str - Section name/heading (optional)",
        "page_number": "int - Page number for PDF/DOCX (optional)",
        "slide_number": "int - Slide number for PPTX (optional)",
        "sheet_name": "str - Sheet name for Excel (optional)",
        "table_id": "str - Table identifier for Excel (optional)",
        "representation": "str - Chunk representation type (optional)",
        "created_at": "str - ISO timestamp of ingestion",
        "additional_context": "dict - Extra contextual data (optional)",
    }
