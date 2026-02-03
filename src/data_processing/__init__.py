"""
Data Processing modules for Agentic RAG System.

This package provides document loading, semantic chunking, and
format-specific processing for PDF, DOCX, PPTX, Excel, and text files.
"""

from .semantic_chunker import Chunk, SemanticChunker
from .document_loader import DocumentLoader, get_document_loader, SUPPORTED_FORMATS
from .pdf_processor import PDFProcessor, PDFDocument, PDFPage
from .docx_processor import DOCXProcessor, DOCXDocument, DOCXSection
from .pptx_processor import PPTXProcessor, PPTXDocument, PPTXSlide
from .excel_processor import ExcelProcessor, ExcelDocument, ExcelTable, ExcelSheet
from .text_processor import TextProcessor, TextDocument


__all__ = [
    # Main interfaces
    "DocumentLoader",
    "get_document_loader",
    "SUPPORTED_FORMATS",
    # Chunking
    "Chunk",
    "SemanticChunker",
    # PDF
    "PDFProcessor",
    "PDFDocument",
    "PDFPage",
    # DOCX
    "DOCXProcessor",
    "DOCXDocument",
    "DOCXSection",
    # PPTX
    "PPTXProcessor",
    "PPTXDocument",
    "PPTXSlide",
    # Excel
    "ExcelProcessor",
    "ExcelDocument",
    "ExcelTable",
    "ExcelSheet",
    # Text
    "TextProcessor",
    "TextDocument",
]
