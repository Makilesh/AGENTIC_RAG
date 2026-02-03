"""Answer display component for Streamlit UI."""

import streamlit as st
from typing import Dict, List, Any


def render_answer(state: Dict[str, Any]):
    """Render the generated answer with formatting."""
    
    answer = state.get("generated_answer", "")
    confidence = state.get("confidence_score", 0)
    
    if not answer:
        st.warning("No answer generated.")
        return
    
    st.subheader("💬 Answer")
    
    # Confidence indicator
    if confidence >= 0.8:
        st.success(f"High Confidence: {confidence:.0%}")
    elif confidence >= 0.6:
        st.warning(f"Medium Confidence: {confidence:.0%}")
    else:
        st.error(f"Low Confidence: {confidence:.0%}")
    
    # Answer content
    st.markdown(answer)
    
    # Validation status
    validation = state.get("validation_status", "")
    if validation:
        with st.expander("🛡️ Validation Details"):
            st.text(validation)


def render_sources(docs: List[Dict[str, Any]]):
    """Render source citations with expandable details."""
    
    if not docs:
        st.info("No sources available.")
        return
    
    st.subheader("📚 Sources")
    
    for i, doc in enumerate(docs[:5], 1):
        score = doc.get("score", 0)
        text = doc.get("text", "")[:500]
        metadata = doc.get("metadata", {})
        
        source_name = metadata.get("file_name", f"Source {i}")
        source_type = metadata.get("source_type", "unknown")
        
        # Type icons
        icons = {
            "pdf": "📄",
            "docx": "📝",
            "pptx": "📊",
            "excel": "📈",
            "txt": "📃"
        }
        icon = icons.get(source_type, "📎")
        
        with st.expander(f"{icon} {source_name} (Score: {score:.2f})"):
            st.markdown(f"**Type:** {source_type.upper()}")
            
            if metadata.get("page_number"):
                st.markdown(f"**Page:** {metadata['page_number']}")
            if metadata.get("slide_number"):
                st.markdown(f"**Slide:** {metadata['slide_number']}")
            if metadata.get("sheet_name"):
                st.markdown(f"**Sheet:** {metadata['sheet_name']}")
            
            st.markdown("---")
            st.markdown(text + "..." if len(doc.get("text", "")) > 500 else text)


def render_query_input() -> str:
    """Render query input area."""
    
    st.subheader("❓ Ask a Question")
    
    # Example queries
    examples = [
        "What is the Q4 revenue for North America?",
        "Summarize the main findings from the report",
        "What are the key recommendations in the presentation?",
        "Compare revenue across all regions"
    ]
    
    with st.expander("💡 Example Queries"):
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                return ex
    
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask anything about your documents..."
    )
    
    return query


def render_result_summary(state: Dict[str, Any]):
    """Render a summary card of the query result."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Quality Score",
            f"{state.get('context_quality_score', 0):.2f}"
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{state.get('confidence_score', 0):.2f}"
        )
    
    with col3:
        st.metric(
            "Rewrites",
            state.get("rewrite_iteration", 0)
        )
    
    with col4:
        st.metric(
            "Sources",
            len(state.get("retrieved_documents", []))
        )
