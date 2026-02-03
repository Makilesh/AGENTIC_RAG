"""File uploader component for Streamlit UI."""

import streamlit as st
import tempfile
from pathlib import Path
from typing import List, Tuple
import os


def render_file_uploader() -> List:
    """Render file upload widget in sidebar."""
    st.sidebar.header("📁 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "xls", "txt", "md"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, PPTX, Excel, TXT"
    )
    
    return uploaded_files or []


def process_uploaded_files(
    uploaded_files: List,
    milvus_client,
    document_loader
) -> Tuple[int, int]:
    """
    Process uploaded files and index them.
    
    Returns:
        Tuple of (documents_processed, chunks_created)
    """
    if not uploaded_files:
        return 0, 0
    
    total_docs = 0
    total_chunks = 0
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status.text(f"Processing: {uploaded_file.name}")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=Path(uploaded_file.name).suffix
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        try:
            # Process document
            chunks = document_loader.load(tmp_path)
            
            if chunks:
                # Index in Milvus
                milvus_client.insert_chunks(chunks, show_progress=False)
                total_chunks += len(chunks)
                total_docs += 1
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            # Cleanup temp file - handle Windows file locking
            try:
                import time
                time.sleep(0.1)  # Brief delay to allow file handles to close
                os.unlink(tmp_path)
            except PermissionError:
                # File still locked, try again after a longer delay
                try:
                    time.sleep(0.5)
                    os.unlink(tmp_path)
                except:
                    pass  # File will be cleaned up by OS eventually
        
        progress.progress((i + 1) / len(uploaded_files))
    
    progress.empty()
    status.empty()
    
    if total_docs > 0:
        st.success(f"✅ Indexed {total_docs} documents ({total_chunks} chunks)")
    
    return total_docs, total_chunks


def render_processing_options():
    """Render processing configuration options."""
    st.sidebar.header("⚙️ Settings")
    
    chunk_size = st.sidebar.slider(
        "Chunk Size (tokens)",
        min_value=400,
        max_value=1000,
        value=600,
        step=50
    )
    
    search_strategy = st.sidebar.selectbox(
        "Search Strategy",
        options=["hybrid", "dense", "sparse"],
        index=0
    )
    
    top_k = st.sidebar.slider(
        "Top K Results",
        min_value=5,
        max_value=20,
        value=10
    )
    
    return {
        "chunk_size": chunk_size,
        "search_strategy": search_strategy,
        "top_k": top_k
    }


def render_system_status(docs_indexed: int, chunks_indexed: int):
    """Render system status in sidebar."""
    st.sidebar.header("📊 System Status")
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Documents", docs_indexed)
    col2.metric("Chunks", chunks_indexed)
