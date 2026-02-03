"""
PDF Processor for Agentic RAG System.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz

from ..utils import get_data_processing_logger
from .semantic_chunker import Chunk, SemanticChunker

logger = get_data_processing_logger()


@dataclass
class PDFPage:
    """
    Represents a single page from a PDF document.
    
    Attributes:
        page_number: 1-indexed page number.
        text: Extracted text content.
        tables: List of extracted tables as markdown.
        has_images: Whether the page contains images.
        structure_markers: Section headings found on this page.
    """
    page_number: int
    text: str
    tables: List[str] = field(default_factory=list)
    has_images: bool = False
    structure_markers: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class PDFDocument:
    """
    Represents a processed PDF document.
    
    Attributes:
        file_name: Name of the PDF file.
        file_path: Full path to the file.
        total_pages: Number of pages in the document.
        pages: List of PDFPage objects.
        full_text: Combined text from all pages.
        metadata: Document-level metadata.
    """
    file_name: str
    file_path: str
    total_pages: int
    pages: List[PDFPage]
    full_text: str
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """
    Process PDF documents for RAG ingestion.
    
    Extracts text with layout preservation, identifies tables,
    and detects document structure (headings, sections).
    """
    
    def __init__(self, chunker: Optional[SemanticChunker] = None):
        """
        Initialize the PDF processor.
        
        Args:
            chunker: SemanticChunker instance. If None, creates a new one.
        """
        self.chunker = chunker or SemanticChunker()
        logger.info("PDFProcessor initialized")
    
    def _detect_headings(self, page: fitz.Page) -> List[Tuple[str, int, float]]:
        """
        Detect headings based on font size and style.
        
        Args:
            page: PyMuPDF page object.
            
        Returns:
            List of (heading_text, char_position, font_size) tuples.
        """
        headings = []
        blocks = page.get_text("dict")["blocks"]
        char_offset = 0
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    font_size = span["size"]
                    flags = span["flags"]
                    
                    # Detect headings by font size (larger than 12) and/or bold
                    is_bold = flags & 2 ** 4  # Bold flag
                    is_large = font_size > 12
                    
                    if text and (is_large or is_bold) and len(text) < 200:
                        # Check if it looks like a heading (not all caps long text)
                        if not (text.isupper() and len(text) > 50):
                            headings.append((text, char_offset, font_size))
                    
                    char_offset += len(text) + 1
        
        return headings
    
    def _extract_tables(self, page: fitz.Page) -> List[str]:
        """
        Extract tables from a page and convert to markdown.
        
        Args:
            page: PyMuPDF page object.
            
        Returns:
            List of tables in markdown format.
        """
        tables = []
        
        try:
            # Use PyMuPDF's table detection
            table_finder = page.find_tables()
            
            for table in table_finder.tables:
                # Extract table data
                data = table.extract()
                
                if not data or len(data) < 2:
                    continue
                
                # Convert to markdown
                md_table = self._table_to_markdown(data)
                if md_table:
                    tables.append(md_table)
                    
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
        
        return tables
    
    def _table_to_markdown(self, data: List[List[str]]) -> str:
        """
        Convert table data to markdown format.
        
        Args:
            data: 2D list of table cell values.
            
        Returns:
            Markdown formatted table string.
        """
        if not data or not data[0]:
            return ""
        
        # Clean cell values
        cleaned_data = []
        for row in data:
            cleaned_row = []
            for cell in row:
                cell_text = str(cell) if cell else ""
                cell_text = cell_text.replace("|", "\\|").strip()
                cleaned_row.append(cell_text)
            cleaned_data.append(cleaned_row)
        
        # Build markdown table
        lines = []
        
        # Header row
        header = "| " + " | ".join(cleaned_data[0]) + " |"
        lines.append(header)
        
        # Separator row
        separator = "|" + "|".join(["---"] * len(cleaned_data[0])) + "|"
        lines.append(separator)
        
        # Data rows
        for row in cleaned_data[1:]:
            row_str = "| " + " | ".join(row) + " |"
            lines.append(row_str)
        
        return "\n".join(lines)
    
    def _extract_page_text(self, page: fitz.Page) -> str:
        """
        Extract text from a page with layout preservation.
        
        Args:
            page: PyMuPDF page object.
            
        Returns:
            Extracted text content.
        """
        # Extract text with layout preservation
        text = page.get_text("text", sort=True)
        
        # Clean up the text
        text = self._clean_text(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers (common patterns)
        text = re.sub(r'\n[Pp]age \d+ of \d+\n', '\n', text)
        
        return text.strip()
    
    def extract(self, file_path: str) -> PDFDocument:
        """
        Extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            PDFDocument containing extracted content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {file_path}")
        
        logger.info(f"Processing PDF: {path.name}")
        
        doc = fitz.open(file_path)
        pages = []
        full_text_parts = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text
                text = self._extract_page_text(page)
                full_text_parts.append(text)
                
                # Detect headings
                headings = self._detect_headings(page)
                structure_markers = [(h[0], h[1]) for h in headings]
                
                # Extract tables
                tables = self._extract_tables(page)
                
                # Check for images
                has_images = len(page.get_images()) > 0
                
                pdf_page = PDFPage(
                    page_number=page_num + 1,
                    text=text,
                    tables=tables,
                    has_images=has_images,
                    structure_markers=structure_markers
                )
                pages.append(pdf_page)
                
        finally:
            doc.close()
        
        # Combine full text
        full_text = "\n\n".join(full_text_parts)
        
        # Build metadata
        metadata = {
            "source_type": "pdf",
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "total_pages": len(pages),
            "has_tables": any(p.tables for p in pages),
            "has_images": any(p.has_images for p in pages)
        }
        
        logger.info(
            f"Extracted {len(pages)} pages from {path.name}",
        )
        
        return PDFDocument(
            file_name=path.name,
            file_path=str(path.absolute()),
            total_pages=len(pages),
            pages=pages,
            full_text=full_text,
            metadata=metadata
        )
    
    def process(self, file_path: str) -> List[Chunk]:
        """
        Extract and chunk a PDF document.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of Chunk objects ready for embedding.
        """
        # Extract content
        pdf_doc = self.extract(file_path)
        
        # Prepare base metadata
        base_metadata = {
            "source_type": "pdf",
            "file_name": pdf_doc.file_name,
            "file_path": pdf_doc.file_path,
            "total_pages": pdf_doc.total_pages,
            "has_tables": pdf_doc.metadata.get("has_tables", False)
        }
        
        all_chunks = []
        
        # Process page by page to maintain page number context
        for page in pdf_doc.pages:
            page_metadata = {
                **base_metadata,
                "page_number": page.page_number
            }
            
            # Chunk the page text
            page_chunks = self.chunker.chunk(
                page.text,
                f"{pdf_doc.file_name}_page{page.page_number}",
                page_metadata
            )
            all_chunks.extend(page_chunks)
            
            # Add table chunks if present
            for i, table in enumerate(page.tables):
                table_metadata = {
                    **page_metadata,
                    "is_table": True,
                    "table_index": i
                }
                
                table_chunk = Chunk(
                    text=f"Table from page {page.page_number}:\n{table}",
                    chunk_id=f"{pdf_doc.file_name}_page{page.page_number}_table{i}",
                    chunk_index=len(all_chunks),
                    metadata=table_metadata
                )
                all_chunks.append(table_chunk)
        
        # Re-index all chunks
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)
        
        logger.info(
            f"Created {len(all_chunks)} chunks from PDF {pdf_doc.file_name}",
        )
        
        return all_chunks
