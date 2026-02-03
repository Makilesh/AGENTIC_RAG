"""LangGraph Orchestrator."""

from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, END

from ..agents import (
    QueryAnalyzerAgent, RetrievalRouterAgent, RetrievalExecutorAgent,
    QualityAssessorAgent, QueryRewriterAgent, AnswerSynthesizerAgent, ValidatorAgent
)
from ..vector_db import HybridSearcher, MilvusRAGClient
from ..llm import format_context_for_prompt
from ..utils import get_workflow_logger, config
from .state_definitions import AgentState, create_initial_state

logger = get_workflow_logger()


class AgenticRAGOrchestrator:
    
    def __init__(self, milvus_client: MilvusRAGClient = None):
        # Initialize agents
        self.query_analyzer = QueryAnalyzerAgent()
        self.retrieval_router = RetrievalRouterAgent()
        self.quality_assessor = QualityAssessorAgent()
        self.query_rewriter = QueryRewriterAgent()
        self.answer_synthesizer = AnswerSynthesizerAgent()
        self.validator = ValidatorAgent()
        
        # Initialize vector DB
        if milvus_client:
            self.milvus_client = milvus_client
        else:
            self.milvus_client = MilvusRAGClient()
            self.milvus_client.connect()
            self.milvus_client.create_collection()
        
        self.searcher = HybridSearcher(self.milvus_client)
        self.retrieval_executor = RetrievalExecutorAgent(self.searcher)
        
        # Build workflow graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
        
        logger.info("AgenticRAGOrchestrator initialized with LangGraph workflow")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("query_analyzer", self._analyze_query)
        graph.add_node("retrieval_router", self._route_retrieval)
        graph.add_node("retrieval_executor", self._execute_retrieval)
        graph.add_node("quality_assessor", self._assess_quality)
        graph.add_node("query_rewriter", self._rewrite_query)
        graph.add_node("answer_synthesizer", self._synthesize_answer)
        graph.add_node("validator", self._validate_answer)
        
        # Set entry point
        graph.set_entry_point("query_analyzer")
        
        # Add edges
        graph.add_edge("query_analyzer", "retrieval_router")
        graph.add_edge("retrieval_router", "retrieval_executor")
        graph.add_edge("retrieval_executor", "quality_assessor")
        
        # Conditional edge after quality assessment (self-corrective loop)
        graph.add_conditional_edges(
            "quality_assessor",
            self._should_rewrite,
            {
                "rewrite": "query_rewriter",
                "proceed": "answer_synthesizer"
            }
        )
        
        # After rewrite, go back to retrieval
        graph.add_edge("query_rewriter", "retrieval_router")
        
        # Final edges
        graph.add_edge("answer_synthesizer", "validator")
        graph.add_edge("validator", END)
        
        return graph
    
    def _analyze_query(self, state: AgentState) -> Dict[str, Any]:
        """Query Analyzer node."""
        result = self.query_analyzer.analyze(state["original_query"])
        decision = {
            "agent": "Query Analyzer",
            "reasoning": f"Detected {result['intent_type']} query with complexity {result.get('complexity_score', 0):.2f}",
            "output": result
        }
        return {
            "parsed_intent": result,
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def _route_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """Retrieval Router node."""
        result = self.retrieval_router.route(
            state["current_query"], state["parsed_intent"]
        )
        decision = {
            "agent": "Retrieval Router",
            "reasoning": f"Routing to {result['search_strategy']} search",
            "output": result
        }
        return {
            "retrieval_plan": result,
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def _execute_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """Retrieval Executor node."""
        results = self.retrieval_executor.execute(
            state["current_query"], state["retrieval_plan"]
        )
        decision = {
            "agent": "Retrieval Executor",
            "reasoning": f"Retrieved {len(results)} documents",
            "output": {"count": len(results)}
        }
        return {
            "retrieved_documents": results,
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def _assess_quality(self, state: AgentState) -> Dict[str, Any]:
        """Quality Assessor node - CRITICAL for self-corrective loop."""
        result = self.quality_assessor.assess(
            state["current_query"], state["retrieved_documents"]
        )
        status = "PASS" if result["passes_threshold"] else "NEEDS IMPROVEMENT"
        decision = {
            "agent": "Quality Assessor",
            "reasoning": f"Quality score: {result['overall_score']:.2f} - {status}",
            "output": result
        }
        return {
            "context_quality_score": result["overall_score"],
            "quality_assessment": result,
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def _should_rewrite(self, state: AgentState) -> Literal["rewrite", "proceed"]:
        """Conditional edge: determine if query needs rewriting."""
        quality = state["context_quality_score"]
        iteration = state["rewrite_iteration"]
        max_iterations = config.quality.max_rewrite_iterations
        
        if quality < config.quality.quality_threshold and iteration < max_iterations:
            logger.info(f"Quality {quality:.2f} below threshold, triggering rewrite (iteration {iteration + 1})")
            return "rewrite"
        
        if quality < config.quality.quality_threshold:
            # Max iterations reached, proceed with warning
            logger.warning(f"Quality still low after {iteration} rewrites, proceeding with warning")
        
        return "proceed"
    
    def _rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """Query Rewriter node - KEY for self-corrective loop."""
        # Build context summary
        context_texts = [d.get("text", "")[:200] for d in state["retrieved_documents"][:3]]
        context_summary = " ".join(context_texts)
        
        result = self.query_rewriter.rewrite(
            original_query=state["original_query"],
            quality_score=state["context_quality_score"],
            missing_aspects=state["quality_assessment"].get("missing_aspects", []),
            context_summary=context_summary,
            iteration=state["rewrite_iteration"]
        )
        
        decision = {
            "agent": "Query Rewriter",
            "reasoning": f"Rewrote query: '{result['rewritten_query'][:50]}...'",
            "output": result
        }
        
        # Update rewrite history
        history = list(state.get("query_rewrite_history", []))
        history.append({
            "iteration": state["rewrite_iteration"] + 1,
            "original": state["current_query"],
            "rewritten": result["rewritten_query"],
            "rationale": result["rationale"],
            "quality_before": state["context_quality_score"]
        })
        
        return {
            "current_query": result["rewritten_query"],
            "rewrite_iteration": state["rewrite_iteration"] + 1,
            "query_rewrite_history": history,
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def _synthesize_answer(self, state: AgentState) -> Dict[str, Any]:
        """Answer Synthesizer node."""
        low_quality = state["context_quality_score"] < config.quality.quality_threshold
        
        result = self.answer_synthesizer.synthesize(
            query=state["original_query"],
            context_documents=state["retrieved_documents"],
            low_quality_warning=low_quality
        )
        
        decision = {
            "agent": "Answer Synthesizer",
            "reasoning": f"Generated answer with {len(result.get('sources_used', []))} sources",
            "output": {"sources": result.get("sources_used", [])}
        }
        
        # Build final context string for validation
        context, _ = format_context_for_prompt(state["retrieved_documents"])
        
        return {
            "generated_answer": result["generated_answer"],
            "sources_used": result.get("sources_used", []),
            "final_context": context,
            "low_quality_warning": low_quality,
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def _validate_answer(self, state: AgentState) -> Dict[str, Any]:
        """Validator node."""
        result = self.validator.validate(
            query=state["original_query"],
            answer=state["generated_answer"],
            context=state["final_context"]
        )
        
        decision = {
            "agent": "Validator",
            "reasoning": f"Confidence: {result['confidence_score']:.2f} - {result['validation_status'].upper()}",
            "output": result
        }
        
        return {
            "confidence_score": result["confidence_score"],
            "validation_result": result,
            "status": "completed",
            "agent_decisions": state["agent_decisions"] + [decision]
        }
    
    def process_query(self, query: str) -> AgentState:
        """
        Process a query through the complete agentic workflow.
        
        Args:
            query: User's question.
            
        Returns:
            Final AgentState with answer and all decisions.
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        initial_state = create_initial_state(query)
        
        try:
            final_state = self.app.invoke(initial_state)
            logger.info(f"Query processed. Confidence: {final_state['confidence_score']:.2f}")
            return final_state
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            initial_state["status"] = "error"
            initial_state["error_message"] = str(e)
            return initial_state


def create_orchestrator(milvus_client: MilvusRAGClient = None) -> AgenticRAGOrchestrator:
    """Create and return an orchestrator instance."""
    return AgenticRAGOrchestrator(milvus_client)
