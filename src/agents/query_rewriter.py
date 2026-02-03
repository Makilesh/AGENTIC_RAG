"""Query Rewriter Agent."""

from typing import Any, Dict, List
from ..llm import get_llm, get_prompt, format_missing_aspects
from ..utils import get_agent_logger, log_agent_decision, config

logger = get_agent_logger("query_rewriter")


class QueryRewriterAgent:
    
    def __init__(self):
        self.llm = get_llm()
        self.name = "Query Rewriter"
        self.max_iterations = config.quality.max_rewrite_iterations
    
    def rewrite(
        self,
        original_query: str,
        quality_score: float,
        missing_aspects: List[str],
        context_summary: str,
        iteration: int
    ) -> Dict[str, Any]:
        """
        Rewrite query to improve retrieval quality.
        
        Returns:
            Dict with rewritten_query, rationale, and strategy used.
        """
        if iteration >= self.max_iterations:
            return {
                "rewritten_query": original_query,
                "rationale": "Maximum rewrite iterations reached",
                "rewrite_strategy": "none",
                "should_continue": False
            }
        
        system_prompt = get_prompt("query_rewriter", "system")
        user_prompt = get_prompt("query_rewriter", "user").format(
            original_query=original_query,
            quality_score=quality_score,
            context_summary=context_summary[:1000],
            missing_aspects=format_missing_aspects(missing_aspects),
            iteration=iteration + 1,
            max_iterations=self.max_iterations
        )
        
        try:
            result = self.llm.complete_json(user_prompt, system_prompt)
            
            rewrite_result = {
                "rewritten_query": result.get("rewritten_query", original_query),
                "rationale": result.get("rationale", "Query reformulated"),
                "rewrite_strategy": result.get("rewrite_strategy", "expansion"),
                "focus_areas": result.get("focus_areas", []),
                "should_continue": True,
                "iteration": iteration + 1
            }
            
            log_agent_decision(
                logger, self.name,
                {"original": original_query[:50], "score": quality_score},
                {"rewritten": rewrite_result["rewritten_query"][:50]},
                f"Rewrote query (iteration {iteration + 1}): {rewrite_result['rewrite_strategy']}"
            )
            
            return rewrite_result
            
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return {
                "rewritten_query": original_query,
                "rationale": f"Rewrite failed: {e}",
                "rewrite_strategy": "none",
                "should_continue": False,
                "error": str(e)
            }
