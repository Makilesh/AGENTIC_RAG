"""
Semantic Chunking module for Agentic RAG System.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils import config, get_data_processing_logger

logger = get_data_processing_logger()


@dataclass
class Chunk:
    """
    Represents a semantically coherent chunk of text.
    
    Attributes:
        text: The chunk text content.
        chunk_id: Unique identifier for the chunk.
        chunk_index: Position index in the document.
        start_char: Starting character position in original document.
        end_char: Ending character position in original document.
        metadata: Additional metadata for the chunk.
    """
    text: str
    chunk_id: str
    chunk_index: int
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        """Estimate token count (rough approximation)."""
        return len(self.text.split())


class SemanticChunker:
    """
    Semantic Chunking implementation using embedding similarity.
    
    Creates chunks based on semantic boundaries rather than fixed character counts.
    Uses sentence embeddings to detect semantic shifts between sentences.
    
    Attributes:
        embedding_model: SentenceTransformer model for embeddings.
        target_chunk_size: Target size for chunks in tokens.
        min_chunk_size: Minimum chunk size in tokens.
        max_chunk_size: Maximum chunk size in tokens.
        overlap_tokens: Number of overlap tokens between chunks.
        similarity_threshold: Threshold for detecting semantic boundaries.
    """
    
    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        target_chunk_size: int = 600,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        overlap_tokens: int = 100,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize the SemanticChunker.
        
        Args:
            embedding_model: Pre-loaded embedding model. If None, loads default.
            target_chunk_size: Target chunk size in tokens.
            min_chunk_size: Minimum chunk size in tokens.
            max_chunk_size: Maximum chunk size in tokens.
            overlap_tokens: Token overlap between consecutive chunks.
            similarity_threshold: Cosine similarity threshold for boundaries.
        """
        self.embedding_model = embedding_model or SentenceTransformer(
            config.embedding.model_name
        )
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        
        logger.info(
            f"SemanticChunker initialized with target_size={target_chunk_size}, "
            f"similarity_threshold={similarity_threshold}"
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Handles common abbreviations and edge cases.
        
        Args:
            text: Input text to split.
            
        Returns:
            List of sentences.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Replace common abbreviations with placeholders to avoid splitting
        abbrev_map = {
            'Mr.': 'Mr§', 'Mrs.': 'Mrs§', 'Ms.': 'Ms§', 'Dr.': 'Dr§',
            'Prof.': 'Prof§', 'Sr.': 'Sr§', 'Jr.': 'Jr§', 'vs.': 'vs§',
            'etc.': 'etc§', 'e.g.': 'eg§', 'i.e.': 'ie§', 'St.': 'St§', 'Mt.': 'Mt§'
        }
        for abbrev, placeholder in abbrev_map.items():
            text = text.replace(abbrev, placeholder)
        
        # Split on sentence boundaries (period, !, ?) followed by space and uppercase
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Split and filter empty sentences
        raw_sentences = re.split(sentence_pattern, text)
        
        # Restore abbreviations
        sentences = []
        for s in raw_sentences:
            for abbrev, placeholder in abbrev_map.items():
                s = s.replace(placeholder, abbrev)
            if s.strip():
                sentences.append(s.strip())
        
        # Handle very long sentences by splitting on semicolons or commas
        processed_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 100:
                # Split long sentences on semicolons or line breaks
                sub_sentences = re.split(r';\s*|\n+', sentence)
                processed_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
            else:
                processed_sentences.append(sentence)
        
        return processed_sentences
    
    def _compute_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of sentences.
        
        Args:
            sentences: List of sentences to embed.
            
        Returns:
            NumPy array of embeddings.
        """
        if not sentences:
            return np.array([])
        
        embeddings = self.embedding_model.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector.
            b: Second vector.
            
        Returns:
            Cosine similarity score.
        """
        return float(np.dot(a, b))
    
    def _find_semantic_boundaries(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Find indices where semantic boundaries occur.
        
        A semantic boundary is detected when cosine similarity between
        adjacent sentences drops below the threshold.
        
        Args:
            sentences: List of sentences.
            embeddings: Corresponding sentence embeddings.
            
        Returns:
            List of sentence indices that start new semantic blocks.
        """
        if len(sentences) <= 1:
            return [0]
        
        boundaries = [0]  # First sentence always starts a block
        
        for i in range(1, len(embeddings)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            if similarity < self.similarity_threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Uses word count as a rough approximation (actual tokens may vary).
        
        Args:
            text: Input text.
            
        Returns:
            Estimated token count.
        """
        return len(text.split())
    
    def _merge_small_chunks(
        self,
        chunks: List[List[str]]
    ) -> List[List[str]]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of chunks (each chunk is a list of sentences).
            
        Returns:
            Merged chunks.
        """
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            current_tokens = self._estimate_tokens(' '.join(current))
            chunk_tokens = self._estimate_tokens(' '.join(chunk))
            combined_tokens = current_tokens + chunk_tokens
            
            # Merge if current chunk is too small and combined won't be too large
            if current_tokens < self.min_chunk_size and combined_tokens <= self.max_chunk_size:
                current.extend(chunk)
            else:
                merged.append(current)
                current = chunk
        
        merged.append(current)
        return merged
    
    def _split_large_chunks(
        self,
        chunks: List[List[str]]
    ) -> List[List[str]]:
        """
        Split chunks that exceed the maximum size.
        
        Args:
            chunks: List of chunks (each chunk is a list of sentences).
            
        Returns:
            Split chunks.
        """
        split_chunks = []
        
        for chunk in chunks:
            chunk_text = ' '.join(chunk)
            tokens = self._estimate_tokens(chunk_text)
            
            if tokens <= self.max_chunk_size:
                split_chunks.append(chunk)
            else:
                # Split into roughly equal parts
                num_parts = (tokens // self.target_chunk_size) + 1
                sentences_per_part = max(1, len(chunk) // num_parts)
                
                for i in range(0, len(chunk), sentences_per_part):
                    part = chunk[i:i + sentences_per_part]
                    if part:
                        split_chunks.append(part)
        
        return split_chunks
    
    def _add_overlap(
        self,
        chunks: List[str],
        overlap_tokens: int
    ) -> List[str]:
        """
        Add overlap between consecutive chunks.
        
        Args:
            chunks: List of chunk texts.
            overlap_tokens: Number of tokens to overlap.
            
        Returns:
            Chunks with overlap added.
        """
        if len(chunks) <= 1 or overlap_tokens <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_words = chunks[i-1].split()
                overlap_text = ' '.join(prev_words[-overlap_tokens:])
                
                # Prepend overlap to current chunk
                overlapped_chunk = f"{overlap_text} {chunk}"
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def chunk(
        self,
        text: str,
        document_id: str = "doc",
        base_metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """
        Perform semantic chunking on the input text.
        
        Args:
            text: The document text to chunk.
            document_id: Identifier for the document.
            base_metadata: Metadata to include with all chunks.
            
        Returns:
            List of Chunk objects.
        """
        base_metadata = base_metadata or {}
        
        # Handle empty text
        if not text or not text.strip():
            logger.warning(f"Empty text provided for document {document_id}")
            return []
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            logger.warning(f"No sentences extracted from document {document_id}")
            return []
        
        logger.debug(f"Split document into {len(sentences)} sentences")
        
        # Step 2: Compute embeddings
        embeddings = self._compute_embeddings(sentences)
        
        # Step 3: Find semantic boundaries
        boundaries = self._find_semantic_boundaries(sentences, embeddings)
        logger.debug(f"Found {len(boundaries)} semantic boundaries")
        
        # Step 4: Create initial chunks from boundaries
        initial_chunks = []
        for i in range(len(boundaries)):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            chunk_sentences = sentences[start_idx:end_idx]
            initial_chunks.append(chunk_sentences)
        
        # Step 5: Merge small chunks
        merged_chunks = self._merge_small_chunks(initial_chunks)
        logger.debug(f"Merged to {len(merged_chunks)} chunks")
        
        # Step 6: Split large chunks
        sized_chunks = self._split_large_chunks(merged_chunks)
        logger.debug(f"After size adjustment: {len(sized_chunks)} chunks")
        
        # Step 7: Convert to text
        chunk_texts = [' '.join(chunk) for chunk in sized_chunks]
        
        # Step 8: Add overlap
        final_texts = self._add_overlap(chunk_texts, self.overlap_tokens)
        
        # Step 9: Create Chunk objects
        chunks = []
        char_offset = 0
        
        for i, chunk_text in enumerate(final_texts):
            chunk_id = f"{document_id}_chunk_{i}"
            
            # Estimate character positions (approximate)
            start_char = char_offset
            end_char = start_char + len(chunk_text)
            
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(final_texts),
                    "token_estimate": self._estimate_tokens(chunk_text)
                }
            )
            chunks.append(chunk)
            char_offset = end_char
        
        logger.info(
            f"Created {len(chunks)} chunks for document {document_id}",
        )
        
        return chunks
    
    def chunk_with_structure(
        self,
        text: str,
        structure_markers: List[Tuple[str, int]],
        document_id: str = "doc",
        base_metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """
        Perform semantic chunking while respecting document structure.
        
        Structure markers indicate section boundaries that should not be crossed.
        
        Args:
            text: The document text to chunk.
            structure_markers: List of (section_name, char_position) tuples.
            document_id: Identifier for the document.
            base_metadata: Metadata to include with all chunks.
            
        Returns:
            List of Chunk objects.
        """
        base_metadata = base_metadata or {}
        
        if not structure_markers:
            return self.chunk(text, document_id, base_metadata)
        
        # Sort markers by position
        sorted_markers = sorted(structure_markers, key=lambda x: x[1])
        
        # Split text by structure markers
        all_chunks = []
        
        for i, (section_name, start_pos) in enumerate(sorted_markers):
            end_pos = sorted_markers[i + 1][1] if i + 1 < len(sorted_markers) else len(text)
            section_text = text[start_pos:end_pos]
            
            # Chunk each section separately
            section_metadata = {
                **base_metadata,
                "section": section_name
            }
            
            section_chunks = self.chunk(
                section_text,
                f"{document_id}_{section_name}",
                section_metadata
            )
            
            all_chunks.extend(section_chunks)
        
        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.chunk_id = f"{document_id}_chunk_{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)
        
        return all_chunks
