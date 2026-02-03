"""
Agentic RAG System - Streamlit Application

Main entry point for the intelligent document Q&A system with
multi-agent orchestration and self-corrective mechanisms.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from components.file_uploader import render_file_uploader, process_uploaded_files
from components.agent_visualizer import render_agent_workflow, render_agent_decisions, render_self_correction_indicator, render_retrieval_metrics
from components.answer_display import render_answer, render_sources


def init_session_state():
    """Initialize session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "milvus_client" not in st.session_state:
        st.session_state.milvus_client = None
    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = 0
    if "chunks_indexed" not in st.session_state:
        st.session_state.chunks_indexed = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = None
    if "document_loader" not in st.session_state:
        st.session_state.document_loader = None


def init_system():
    """Initialize the RAG system components."""
    if st.session_state.orchestrator is None:
        with st.spinner("Initializing system..."):
            try:
                from src.vector_db import MilvusRAGClient
                from src.workflow import create_orchestrator
                
                client = MilvusRAGClient()
                client.connect()
                client.create_collection()
                
                st.session_state.milvus_client = client
                st.session_state.orchestrator = create_orchestrator(client)
                
                # Get stats
                stats = client.get_collection_stats()
                st.session_state.chunks_indexed = stats.get("count", 0)
                
            except Exception as e:
                st.error(f"Failed to initialize system: {e}")
                st.info("Make sure Milvus is running. Use: docker-compose up -d")


def render_sidebar():
    """Render the sidebar with configuration and file upload."""
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "pptx", "xlsx", "txt"],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, PPTX, Excel, TXT"
        )
        
        if uploaded_files and st.button("📥 Process Documents", type="primary"):
            if st.session_state.milvus_client is None:
                st.error("System not initialized. Please wait...")
            else:
                # Initialize document loader once and cache it
                if st.session_state.document_loader is None:
                    from src.data_processing import DocumentLoader
                    st.session_state.document_loader = DocumentLoader()
                
                docs, chunks = process_uploaded_files(
                    uploaded_files,
                    st.session_state.milvus_client,
                    st.session_state.document_loader
                )
                st.session_state.documents_indexed += docs
                st.session_state.chunks_indexed += chunks
        
        st.divider()
        
        # System status
        st.header("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.documents_indexed)
        with col2:
            st.metric("Chunks", st.session_state.chunks_indexed)
        
        # LLM status
        st.subheader("🤖 LLM Status")
        try:
            from src.llm import get_llm
            llm = get_llm()
            status = llm.check_availability()
            
            if status["primary"]:
                st.success("✅ Gemini: Online")
            else:
                st.warning("⚠️ Gemini: Offline")
            
            if status["fallback"]:
                st.success("✅ Ollama: Online")
            else:
                st.info("ℹ️ Ollama: Offline")
        except:
            st.warning("LLM status unavailable")
        
        st.divider()
        
        # Search settings
        st.header("⚙️ Settings")
        st.selectbox(
            "Search Strategy", 
            ["Hybrid (Dense + Sparse)", "Dense Only (Semantic)", "Sparse Only (Keyword)"], 
            key="search_strategy"
        )
        st.slider("Results to Retrieve", 5, 20, 10, key="top_k")


def render_main():
    """Render the main query interface."""
    st.title("🤖 Agentic RAG System")
    st.caption("Intelligent Document Q&A with Self-Corrective AI Agents")
    
    # Handle example query selection via session state (BUG 4 fix)
    default_query = ""
    if st.session_state.selected_example:
        default_query = st.session_state.selected_example
        st.session_state.selected_example = None  # Clear after use
    
    # Query input
    query = st.text_area(
        "Ask a question about your documents:",
        value=default_query,
        placeholder="e.g., What was the Q4 revenue for North America?",
        height=100
    )
    
    # Example queries
    with st.expander("💡 Example Queries"):
        examples = [
            "What are the main findings in the report?",
            "Summarize the key recommendations",
            "What is the total revenue across all regions?",
            "Compare the performance metrics year over year"
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state.selected_example = ex
                st.rerun()
    
    # Process query
    if st.button("🔍 Search & Answer", type="primary", disabled=not query):
        if st.session_state.orchestrator is None:
            st.error("System not initialized. Please wait or refresh.")
            return
        
        with st.spinner("Processing query through agent workflow..."):
            try:
                result = st.session_state.orchestrator.process_query(query)
                st.session_state.last_result = result
            except Exception as e:
                st.error(f"Error processing query: {e}")
                return
    
    # Display results
    if st.session_state.last_result:
        result = st.session_state.last_result
        
        # Show agent decisions
        st.subheader("🔄 Agent Workflow")
        render_agent_decisions(result.get("agent_decisions", []))
        
        # Show self-correction indicator (BUG 5 fix - call the component)
        render_self_correction_indicator(result)
        
        # Show answer
        st.subheader("📝 Answer")
        render_answer(result)
        
        # Show retrieval metrics (BUG 5 fix - call the component)
        render_retrieval_metrics(result)
        
        # Show sources
        if result.get("retrieved_documents"):
            st.subheader("📚 Sources")
            render_sources(result["retrieved_documents"])


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Agentic RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stAlert {border-radius: 10px;}
        .agent-decision {padding: 10px; border-left: 3px solid #1f77b4; margin: 5px 0;}
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    init_system()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
