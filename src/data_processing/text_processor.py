"""
Text Processor for Agentic RAG System.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..utils import get_data_processing_logger
from .semantic_chunker import Chunk, SemanticChunker

logger = get_data_processing_logger()


@dataclass
class TextDocument:
    """
    Represents a processed text document.
    
    Attributes:
        file_name: Name of the text file.
        file_path: Full path to the file.
        content: Full text content.
        encoding: Detected encoding.
        metadata: Document-level metadata.
    """
    file_name: str
    file_path: str
    content: str
    encoding: str
    metadata: dict = field(default_factory=dict)


class TextProcessor:
    """
    Process plain text files for RAG ingestion.
    
    Handles encoding detection, paragraph detection,
    and semantic chunking of text content.
    """
    
    # Common encodings to try
    ENCODINGS = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
    
    def __init__(self, chunker: Optional[SemanticChunker] = None):
        """
        Initialize the Text processor.
        
        Args:
            chunker: SemanticChunker instance. If None, creates a new one.
        """
        self.chunker = chunker or SemanticChunker()
        logger.info("TextProcessor initialized")
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect the encoding of a text file.
        
        Tries common encodings and returns the first that works.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Detected encoding name.
        """
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Read first 1KB to test
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Default to utf-8 with error handling
        return 'utf-8'
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content.
            
        Returns:
            Cleaned text.
        """
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text.strip()
    
    def _detect_structure(self, text: str) -> List[tuple]:
        """
        Detect structural elements in text.
        
        Looks for markdown-style headings, numbered sections, etc.
        
        Args:
            text: Text content.
            
        Returns:
            List of (heading_text, char_position) tuples.
        """
        structure = []
        
        # Markdown headings
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            level = len(match.group(1))
            heading = match.group(2).strip()
            structure.append((heading, match.start()))
        
        # Numbered sections (e.g., "1. Introduction")
        for match in re.finditer(r'^(\d+\.)+\s+([A-Z][^\n]+)$', text, re.MULTILINE):
            heading = match.group(2).strip()
            structure.append((heading, match.start()))
        
        # All-caps headings (common in plain text)
        for match in re.finditer(r'^([A-Z][A-Z\s]{5,50})$', text, re.MULTILINE):
            heading = match.group(1).strip()
            if not heading.isspace():
                structure.append((heading, match.start()))
        
        # Sort by position
        structure.sort(key=lambda x: x[1])
        
        return structure
    
    def extract(self, file_path: str) -> TextDocument:
        """
        Extract content from a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            TextDocument containing extracted content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        if not path.suffix.lower() in ['.txt', '.text', '.md', '.markdown', '']:
            raise ValueError(f"Not a text file: {file_path}")
        
        logger.info(f"Processing text file: {path.name}")
        
        # Detect encoding
        encoding = self._detect_encoding(file_path)
        
        # Read content
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
        
        # Clean content
        content = self._clean_text(content)
        
        # Build metadata
        metadata = {
            "source_type": "txt",
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "encoding": encoding,
            "char_count": len(content),
            "word_count": len(content.split())
        }
        
        logger.info(
            f"Extracted {metadata['word_count']} words from {path.name}",
        )
        
        return TextDocument(
            file_name=path.name,
            file_path=str(path.absolute()),
            content=content,
            encoding=encoding,
            metadata=metadata
        )
    
    def process(self, file_path: str) -> List[Chunk]:
        """
        Extract and chunk a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            List of Chunk objects ready for embedding.
        """
        # Extract content
        text_doc = self.extract(file_path)
        
        # Prepare base metadata
        base_metadata = {
            "source_type": "txt",
            "file_name": text_doc.file_name,
            "file_path": text_doc.file_path,
            "encoding": text_doc.encoding
        }
        
        # Detect structure
        structure = self._detect_structure(text_doc.content)
        
        # Chunk with structure awareness if structure detected
        if structure:
            chunks = self.chunker.chunk_with_structure(
                text_doc.content,
                structure,
                text_doc.file_name,
                base_metadata
            )
        else:
            chunks = self.chunker.chunk(
                text_doc.content,
                text_doc.file_name,
                base_metadata
            )
        
        # Re-index all chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.chunk_id = f"{text_doc.file_name}_chunk_{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.info(
            f"Created {len(chunks)} chunks from text file {text_doc.file_name}",
        )
        
        return chunks
