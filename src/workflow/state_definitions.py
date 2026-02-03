"""State definitions for LangGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """
    Complete state for the Agentic RAG workflow.
    
    This state is passed between all agents and tracks the entire
    query processing lifecycle including self-corrective loops.
    """
    # Query information
    original_query: str
    current_query: str  # May be rewritten
    
    # Query analysis
    parsed_intent: Dict[str, Any]
    
    # Retrieval planning
    retrieval_plan: Dict[str, Any]
    
    # Retrieved documents
    retrieved_documents: List[Dict[str, Any]]
    
    # Quality assessment
    context_quality_score: float
    quality_assessment: Dict[str, Any]
    
    # Query rewriting (self-corrective loop)
    rewrite_iteration: int
    query_rewrite_history: List[Dict[str, Any]]
    
    # Answer generation
    final_context: str
    generated_answer: str
    sources_used: List[str]
    
    # Validation
    confidence_score: float
    validation_result: Dict[str, Any]
    
    # Agent decision log (for UI visibility)
    agent_decisions: List[Dict[str, Any]]
    
    # Status
    status: str  # "processing", "completed", "error"
    error_message: Optional[str]
    low_quality_warning: bool


def create_initial_state(query: str) -> AgentState:
    """Create initial state for a new query."""
    return AgentState(
        original_query=query,
        current_query=query,
        parsed_intent={},
        retrieval_plan={},
        retrieved_documents=[],
        context_quality_score=0.0,
        quality_assessment={},
        rewrite_iteration=0,
        query_rewrite_history=[],
        final_context="",
        generated_answer="",
        sources_used=[],
        confidence_score=0.0,
        validation_result={},
        agent_decisions=[],
        status="processing",
        error_message=None,
        low_quality_warning=False
    )


def add_agent_decision(
    state: AgentState,
    agent_name: str,
    decision: str,
    details: Dict[str, Any]
) -> None:
    """Add an agent decision to the state for UI visibility."""
    state["agent_decisions"].append({
        "agent": agent_name,
        "decision": decision,
        "details": details
    })
