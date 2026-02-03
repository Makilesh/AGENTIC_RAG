"""Agent workflow visualizer for Streamlit UI."""

import streamlit as st
from typing import Dict, List, Any


def render_agent_workflow(state: Dict[str, Any]):
    """Render the agent workflow execution visualization."""
    
    st.subheader("🔄 Agent Workflow")
    
    agents = [
        ("Query Analyzer", "query_analyzer", "🔍"),
        ("Retrieval Router", "retrieval_router", "🗂️"),
        ("Retrieval Executor", "retrieval_executor", "📥"),
        ("Quality Assessor", "quality_assessor", "✅"),
        ("Query Rewriter", "query_rewriter", "✏️"),
        ("Answer Synthesizer", "answer_synthesizer", "💡"),
        ("Validator", "validator", "🛡️")
    ]
    
    # Create workflow columns
    cols = st.columns(7)
    
    decisions = state.get("agent_decisions", [])
    executed_agents = {d.get("agent") for d in decisions}
    
    for i, (name, key, icon) in enumerate(agents):
        with cols[i]:
            executed = key in executed_agents or any(key in d.get("agent", "").lower() for d in decisions)
            
            if executed:
                st.markdown(f"### {icon}")
                st.markdown(f"**{name.split()[0]}**")
                st.success("Done")
            else:
                st.markdown(f"### {icon}")
                st.markdown(f"**{name.split()[0]}**")
                st.text("Pending")


def render_agent_decisions(decisions: List[Dict[str, Any]]):
    """Render detailed agent decisions in expandable sections."""
    
    if not decisions:
        st.info("No agent decisions recorded yet.")
        return
    
    for decision in decisions:
        agent = decision.get("agent", "Unknown Agent")
        reasoning = decision.get("reasoning", "No reasoning provided")
        output = decision.get("output", {})
        
        # Get appropriate icon
        icons = {
            "query_analyzer": "🔍",
            "retrieval_router": "🗂️", 
            "retrieval_executor": "📥",
            "quality_assessor": "✅",
            "query_rewriter": "✏️",
            "answer_synthesizer": "💡",
            "validator": "🛡️"
        }
        icon = icons.get(agent.lower().replace(" ", "_"), "🤖")
        
        with st.expander(f"{icon} {agent}", expanded=False):
            st.markdown(f"**Decision:** {reasoning}")
            
            if output:
                st.json(output)


def render_self_correction_indicator(state: Dict[str, Any]):
    """Render self-correction loop indicator if query was rewritten."""
    
    rewrite_history = state.get("query_rewrite_history", [])
    rewrite_iteration = state.get("rewrite_iteration", 0)
    
    if rewrite_iteration > 0 and rewrite_history:
        st.subheader("🔁 Self-Correction Applied")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Query:**")
            st.info(state.get("original_query", ""))
        
        with col2:
            st.markdown("**Rewritten Query:**")
            if rewrite_history:
                latest = rewrite_history[-1]
                st.success(latest.get("rewritten_query", ""))
        
        # Show quality improvement
        if len(rewrite_history) >= 1:
            initial_score = rewrite_history[0].get("quality_before", 0)
            final_score = state.get("context_quality_score", 0)
            
            if initial_score > 0:
                improvement = final_score - initial_score
                st.metric(
                    "Quality Improvement",
                    f"{final_score:.2f}",
                    f"{improvement:+.2f}"
                )


def render_retrieval_metrics(state: Dict[str, Any]):
    """Render retrieval quality metrics."""
    
    quality_score = state.get("context_quality_score", 0)
    confidence = state.get("confidence_score", 0)
    docs = state.get("retrieved_documents", [])
    
    st.subheader("📈 Retrieval Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "normal" if quality_score >= 0.7 else "inverse"
        st.metric("Context Quality", f"{quality_score:.2f}")
    
    with col2:
        st.metric("Documents Retrieved", len(docs))
    
    with col3:
        color = "normal" if confidence >= 0.8 else "inverse"
        st.metric("Confidence", f"{confidence:.2f}")
