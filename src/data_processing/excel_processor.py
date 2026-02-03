"""
Excel Processor for Agentic RAG System.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..utils import get_data_processing_logger
from .semantic_chunker import Chunk, SemanticChunker

logger = get_data_processing_logger()


@dataclass
class ExcelTable:
    """
    Represents a detected table within an Excel sheet.
    
    Attributes:
        table_id: Unique identifier for the table.
        sheet_name: Name of the sheet containing this table.
        headers: List of column headers.
        data: List of dictionaries representing rows.
        row_count: Number of data rows (excluding header).
        start_row: Starting row in the sheet.
        start_col: Starting column in the sheet.
        natural_language_desc: Natural language description of the table.
        summary_stats: Summary statistics for numeric columns.
    """
    table_id: str
    sheet_name: str
    headers: List[str]
    data: List[Dict[str, Any]]
    row_count: int
    start_row: int = 0
    start_col: int = 0
    natural_language_desc: str = ""
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ExcelSheet:
    """
    Represents a single sheet from an Excel workbook.
    
    Attributes:
        name: Sheet name.
        tables: List of tables detected in the sheet.
        raw_data: Raw DataFrame of the sheet.
    """
    name: str
    tables: List[ExcelTable]
    raw_data: Optional[pd.DataFrame] = None


@dataclass
class ExcelDocument:
    """
    Represents a processed Excel workbook.
    
    Attributes:
        file_name: Name of the Excel file.
        file_path: Full path to the file.
        sheets: List of ExcelSheet objects.
        metadata: Workbook-level metadata.
    """
    file_name: str
    file_path: str
    sheets: List[ExcelSheet]
    metadata: dict = field(default_factory=dict)


class ExcelProcessor:
    """
    Process Excel spreadsheets for RAG ingestion.
    
    Implements sophisticated table-to-text conversion that creates
    natural language descriptions of tabular data for semantic search.
    This is a KEY DIFFERENTIATOR from naive approaches.
    """
    
    def __init__(self, chunker: Optional[SemanticChunker] = None):
        """
        Initialize the Excel processor.
        
        Args:
            chunker: SemanticChunker instance. If None, creates a new one.
        """
        self.chunker = chunker or SemanticChunker()
        logger.info("ExcelProcessor initialized")
    
    def _detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Detect the header row in a DataFrame.
        
        Looks for rows where most cells are strings and non-empty.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Index of the header row.
        """
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            # Check if most cells are strings and non-empty
            string_count = sum(
                1 for val in row 
                if isinstance(val, str) and val.strip()
            )
            
            if string_count >= len(row) * 0.5:
                return idx
        
        return 0
    
    def _clean_header(self, header: Any) -> str:
        """
        Clean and normalize a header value.
        
        Args:
            header: Raw header value.
            
        Returns:
            Cleaned header string.
        """
        if pd.isna(header):
            return "Column"
        
        header_str = str(header).strip()
        header_str = re.sub(r'\s+', ' ', header_str)
        header_str = re.sub(r'[^\w\s\-\_]', '', header_str)
        
        return header_str[:50] if header_str else "Column"
    
    def _format_value(self, value: Any, dtype: str = None) -> str:
        """
        Format a cell value for natural language output.
        
        Args:
            value: Cell value.
            dtype: Optional data type hint.
            
        Returns:
            Formatted string representation.
        """
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, (int, np.integer)):
            # Format large numbers with commas
            if abs(value) >= 1000000:
                return f"{value/1000000:.1f} million"
            elif abs(value) >= 1000:
                return f"{value:,}"
            return str(value)
        
        if isinstance(value, (float, np.floating)):
            if abs(value) >= 1000000:
                return f"${value/1000000:.1f} million" if dtype == "currency" else f"{value/1000000:.1f} million"
            elif abs(value) >= 1000:
                return f"${value:,.2f}" if dtype == "currency" else f"{value:,.2f}"
            return f"{value:.2f}"
        
        return str(value).strip()
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer the semantic type of a column.
        
        Args:
            series: Pandas Series to analyze.
            
        Returns:
            Inferred type string.
        """
        col_name = str(series.name).lower()
        
        # Check column name for hints
        if any(word in col_name for word in ['revenue', 'sales', 'price', 'cost', 'amount', 'total', '$']):
            return "currency"
        if any(word in col_name for word in ['date', 'time', 'year', 'month', 'day']):
            return "date"
        if any(word in col_name for word in ['percent', 'rate', '%', 'ratio']):
            return "percentage"
        if any(word in col_name for word in ['count', 'number', 'quantity', 'qty']):
            return "count"
        
        # Check data type
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        
        return "text"
    
    def _generate_table_description(self, table: ExcelTable) -> str:
        """
        Generate a natural language description of a table.
        
        This is the CORE of table-to-text conversion for semantic search.
        
        Args:
            table: ExcelTable object.
            
        Returns:
            Natural language description of the table.
        """
        parts = []
        
        # Introduction
        parts.append(
            f"This table from sheet '{table.sheet_name}' contains {table.row_count} rows "
            f"with {len(table.headers)} columns: {', '.join(table.headers)}."
        )
        
        # Convert data to DataFrame for analysis
        df = pd.DataFrame(table.data)
        
        if df.empty:
            return parts[0]
        
        # Describe each row with context
        row_descriptions = []
        for i, row in enumerate(table.data[:10]):  # Limit to first 10 rows
            row_parts = []
            for col in table.headers:
                value = row.get(col, "N/A")
                col_type = self._infer_column_type(df[col]) if col in df.columns else "text"
                formatted = self._format_value(value, col_type)
                row_parts.append(f"{col}: {formatted}")
            
            row_desc = ", ".join(row_parts)
            row_descriptions.append(row_desc)
        
        # Add row descriptions
        parts.append("\nDetailed data:")
        for i, desc in enumerate(row_descriptions, 1):
            parts.append(f"Row {i}: {desc}")
        
        if len(table.data) > 10:
            parts.append(f"... and {len(table.data) - 10} more rows.")
        
        # Add summary statistics for numeric columns
        if table.summary_stats:
            parts.append("\nSummary statistics:")
            for col, stats in table.summary_stats.items():
                stat_parts = []
                if 'sum' in stats:
                    stat_parts.append(f"total: {self._format_value(stats['sum'])}")
                if 'mean' in stats:
                    stat_parts.append(f"average: {self._format_value(stats['mean'])}")
                if 'min' in stats:
                    stat_parts.append(f"min: {self._format_value(stats['min'])}")
                if 'max' in stats:
                    stat_parts.append(f"max: {self._format_value(stats['max'])}")
                
                if stat_parts:
                    parts.append(f"- {col}: {', '.join(stat_parts)}")
        
        return "\n".join(parts)
    
    def _generate_column_summary(self, table: ExcelTable) -> str:
        """
        Generate a column-focused summary for structured queries.
        
        Args:
            table: ExcelTable object.
            
        Returns:
            Column summary text.
        """
        df = pd.DataFrame(table.data)
        
        if df.empty:
            return f"Table with columns: {', '.join(table.headers)}"
        
        parts = [f"Table structure from '{table.sheet_name}':"]
        
        for col in table.headers:
            if col not in df.columns:
                continue
            
            col_type = self._infer_column_type(df[col])
            unique_count = df[col].nunique()
            
            col_desc = f"- {col} ({col_type}): {unique_count} unique values"
            
            # Add sample values for text columns
            if col_type == "text" and unique_count <= 10:
                samples = df[col].dropna().unique()[:5]
                col_desc += f" (e.g., {', '.join(str(s) for s in samples)})"
            
            parts.append(col_desc)
        
        return "\n".join(parts)
    
    def _generate_sample_rows(self, table: ExcelTable) -> str:
        """
        Generate sample row representation for example-based retrieval.
        
        Args:
            table: ExcelTable object.
            
        Returns:
            Sample rows as text.
        """
        if not table.data:
            return ""
        
        parts = [f"Sample data from '{table.sheet_name}':"]
        
        # Header row
        parts.append("| " + " | ".join(table.headers) + " |")
        parts.append("|" + "|".join(["---"] * len(table.headers)) + "|")
        
        # Sample rows
        for row in table.data[:5]:
            row_values = [str(row.get(col, ""))[:30] for col in table.headers]
            parts.append("| " + " | ".join(row_values) + " |")
        
        return "\n".join(parts)
    
    def _compute_summary_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for numeric columns.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary of statistics per column.
        """
        stats = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    col_stats = {
                        'sum': float(df[col].sum()),
                        'mean': float(df[col].mean()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    }
                    stats[col] = col_stats
                except (ValueError, TypeError):
                    continue
        
        return stats
    
    def _extract_tables_from_sheet(
        self, 
        df: pd.DataFrame, 
        sheet_name: str
    ) -> List[ExcelTable]:
        """
        Extract tables from a sheet DataFrame.
        
        Handles header detection and table boundary detection.
        
        Args:
            df: DataFrame of the sheet.
            sheet_name: Name of the sheet.
            
        Returns:
            List of ExcelTable objects.
        """
        if df.empty:
            return []
        
        # Detect header row
        header_row = self._detect_header_row(df)
        
        # Set headers
        if header_row > 0:
            df = df.iloc[header_row:].reset_index(drop=True)
        
        # Clean headers
        headers = [self._clean_header(col) for col in df.columns]
        df.columns = headers
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        if df.empty:
            return []
        
        # Convert to records
        data = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in headers:
                value = row[col]
                if pd.notna(value):
                    row_dict[col] = value
            if row_dict:
                data.append(row_dict)
        
        if not data:
            return []
        
        # Compute summary stats
        summary_stats = self._compute_summary_stats(df)
        
        # Create table object
        table = ExcelTable(
            table_id=f"{sheet_name}_table_0",
            sheet_name=sheet_name,
            headers=headers,
            data=data,
            row_count=len(data),
            summary_stats=summary_stats
        )
        
        # Generate natural language description
        table.natural_language_desc = self._generate_table_description(table)
        
        return [table]
    
    def extract(self, file_path: str) -> ExcelDocument:
        """
        Extract content from an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            
        Returns:
            ExcelDocument containing extracted content.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        if not path.suffix.lower() in ['.xlsx', '.xls', '.xlsm']:
            raise ValueError(f"Not an Excel file: {file_path}")
        
        logger.info(f"Processing Excel: {path.name}")
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheets = []
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                
                if df.empty:
                    continue
                
                tables = self._extract_tables_from_sheet(df, sheet_name)
                
                sheet = ExcelSheet(
                    name=sheet_name,
                    tables=tables,
                    raw_data=df
                )
                sheets.append(sheet)
                
            except Exception as e:
                logger.warning(f"Error processing sheet '{sheet_name}': {e}")
                continue
        
        # Build metadata
        metadata = {
            "source_type": "excel",
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "num_sheets": len(sheets),
            "total_tables": sum(len(s.tables) for s in sheets)
        }
        
        logger.info(
            f"Extracted {len(sheets)} sheets with "
            f"{metadata['total_tables']} tables from {path.name}",
        )
        
        return ExcelDocument(
            file_name=path.name,
            file_path=str(path.absolute()),
            sheets=sheets,
            metadata=metadata
        )
    
    def process(self, file_path: str) -> List[Chunk]:
        """
        Extract and create multi-representation chunks from an Excel file.
        
        Creates THREE types of chunks per table for optimal retrieval:
        1. Natural language description (for semantic search)
        2. Column structure summary (for structured queries)
        3. Sample rows (for example-based retrieval)
        
        Args:
            file_path: Path to the Excel file.
            
        Returns:
            List of Chunk objects ready for embedding.
        """
        # Extract content
        excel_doc = self.extract(file_path)
        
        all_chunks = []
        
        for sheet in excel_doc.sheets:
            for table in sheet.tables:
                base_metadata = {
                    "source_type": "excel",
                    "file_name": excel_doc.file_name,
                    "file_path": excel_doc.file_path,
                    "sheet_name": table.sheet_name,
                    "table_id": table.table_id,
                    "columns": table.headers,
                    "row_count": table.row_count
                }
                
                # Representation 1: Natural language description
                nl_chunk = Chunk(
                    text=table.natural_language_desc,
                    chunk_id=f"{table.table_id}_nl",
                    chunk_index=len(all_chunks),
                    metadata={
                        **base_metadata,
                        "representation": "natural_language",
                        "chunk_index": len(all_chunks)
                    }
                )
                all_chunks.append(nl_chunk)
                
                # Representation 2: Column structure summary
                col_summary = self._generate_column_summary(table)
                struct_chunk = Chunk(
                    text=col_summary,
                    chunk_id=f"{table.table_id}_struct",
                    chunk_index=len(all_chunks),
                    metadata={
                        **base_metadata,
                        "representation": "structure",
                        "chunk_index": len(all_chunks)
                    }
                )
                all_chunks.append(struct_chunk)
                
                # Representation 3: Sample rows
                sample_rows = self._generate_sample_rows(table)
                if sample_rows:
                    sample_chunk = Chunk(
                        text=sample_rows,
                        chunk_id=f"{table.table_id}_samples",
                        chunk_index=len(all_chunks),
                        metadata={
                            **base_metadata,
                            "representation": "samples",
                            "chunk_index": len(all_chunks)
                        }
                    )
                    all_chunks.append(sample_chunk)
        
        # Update total chunks count
        for chunk in all_chunks:
            chunk.metadata["total_chunks"] = len(all_chunks)
        
        logger.info(
            f"Created {len(all_chunks)} chunks (multi-representation) "
            f"from Excel {excel_doc.file_name}",
        )
        
        return all_chunks
