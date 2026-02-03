"""
Document Loader for Agentic RAG System.
"""

from pathlib import Path
from typing import List, Optional, Union

from ..utils import get_data_processing_logger, metrics_collector
from .semantic_chunker import Chunk, SemanticChunker
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .pptx_processor import PPTXProcessor
from .excel_processor import ExcelProcessor
from .text_processor import TextProcessor

logger = get_data_processing_logger()


SUPPORTED_FORMATS = {
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.doc': 'docx',  # Note: .doc may have limited support
    '.pptx': 'pptx',
    '.ppt': 'pptx',  # Note: .ppt may have limited support
    '.xlsx': 'excel',
    '.xls': 'excel',
    '.xlsm': 'excel',
    '.txt': 'text',
    '.text': 'text',
    '.md': 'text',
    '.markdown': 'text',
}


class DocumentLoader:
    """
    Unified document loader for multiple file formats.
    
    Automatically detects file type and uses the appropriate processor
    to extract and chunk document content.
    
    Attributes:
        chunker: Shared SemanticChunker instance.
        processors: Dictionary of initialized processors.
    """
    
    def __init__(self, chunker: Optional[SemanticChunker] = None):
        """
        Initialize the document loader.
        
        Args:
            chunker: Optional SemanticChunker instance. If None, creates a new one.
        """
        self.chunker = chunker or SemanticChunker()
        
        # Initialize all processors with shared chunker
        self.processors = {
            'pdf': PDFProcessor(self.chunker),
            'docx': DOCXProcessor(self.chunker),
            'pptx': PPTXProcessor(self.chunker),
            'excel': ExcelProcessor(self.chunker),
            'text': TextProcessor(self.chunker),
        }
        
        logger.info("DocumentLoader initialized with all processors")
    
    def _get_processor_type(self, file_path: str) -> Optional[str]:
        """
        Determine the processor type based on file extension.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Processor type string or None if unsupported.
        """
        ext = Path(file_path).suffix.lower()
        return SUPPORTED_FORMATS.get(ext)
    
    def is_supported(self, file_path: str) -> bool:
        """
        Check if a file type is supported.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            True if the file type is supported.
        """
        return self._get_processor_type(file_path) is not None
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions (e.g., ['.pdf', '.docx', ...]).
        """
        return list(SUPPORTED_FORMATS.keys())
    
    def load(self, file_path: str) -> List[Chunk]:
        """
        Load and process a document file.
        
        Automatically detects file type and processes accordingly.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            List of Chunk objects ready for embedding.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file type is not supported.
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get processor type
        processor_type = self._get_processor_type(file_path)
        
        if not processor_type:
            supported = ", ".join(self.get_supported_formats())
            raise ValueError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported formats: {supported}"
            )
        
        # Start metrics tracking
        doc_metrics = metrics_collector.start_document_processing(
            file_name=path.name,
            file_type=processor_type,
            file_size_bytes=path.stat().st_size
        )
        
        try:
            # Get the appropriate processor
            processor = self.processors[processor_type]
            
            # Process the document
            logger.info(f"Loading document: {path.name} (type: {processor_type})")
            chunks = processor.process(str(path.absolute()))
            
            # Complete metrics tracking
            metrics_collector.complete_document_processing(
                doc_metrics,
                num_chunks=len(chunks),
                success=True
            )
            
            logger.info(
                f"Successfully processed {path.name}: {len(chunks)} chunks created"
            )
            
            return chunks
            
        except Exception as e:
            # Log error and complete metrics
            metrics_collector.complete_document_processing(
                doc_metrics,
                num_chunks=0,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Error processing {path.name}: {e}")
            raise
    
    def load_multiple(
        self, 
        file_paths: List[str],
        continue_on_error: bool = True
    ) -> List[Chunk]:
        """
        Load and process multiple documents.
        
        Args:
            file_paths: List of paths to document files.
            continue_on_error: If True, continues processing on error.
            
        Returns:
            Combined list of Chunk objects from all documents.
        """
        all_chunks = []
        errors = []
        
        for file_path in file_paths:
            try:
                chunks = self.load(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                if not continue_on_error:
                    raise
        
        if errors:
            logger.warning(
                f"Completed with {len(errors)} errors out of {len(file_paths)} files"
            )
        
        logger.info(
            f"Loaded {len(file_paths)} documents: {len(all_chunks)} total chunks"
        )
        
        return all_chunks
    
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        continue_on_error: bool = True
    ) -> List[Chunk]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory.
            recursive: If True, searches subdirectories.
            continue_on_error: If True, continues processing on error.
            
        Returns:
            Combined list of Chunk objects from all documents.
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        # Find all supported files
        file_paths = []
        
        pattern_func = dir_path.rglob if recursive else dir_path.glob
        
        for ext in SUPPORTED_FORMATS.keys():
            for file_path in pattern_func(f"*{ext}"):
                file_paths.append(str(file_path))
        
        if not file_paths:
            logger.warning(f"No supported documents found in {directory_path}")
            return []
        
        logger.info(f"Found {len(file_paths)} documents in {directory_path}")
        
        return self.load_multiple(file_paths, continue_on_error)


def get_document_loader() -> DocumentLoader:
    """
    Get a configured DocumentLoader instance.
    
    Returns:
        Configured DocumentLoader.
    """
    return DocumentLoader()
