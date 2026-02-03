"""Quality Assessor Agent."""

from typing import Any, Dict, List
from ..llm import get_llm, get_prompt, format_context_for_prompt
from ..utils import get_agent_logger, log_agent_decision, config

logger = get_agent_logger("quality_assessor")


class QualityAssessorAgent:
    
    def __init__(self):
        self.llm = get_llm()
        self.name = "Quality Assessor"
        self.quality_threshold = config.quality.quality_threshold
    
    def assess(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess the quality of retrieved context for answering the query.
        
        Returns:
            Dict with overall_score, missing_aspects, and assessment details.
        """
        if not retrieved_documents:
            return {
                "overall_score": 0.0,
                "relevance_score": 0.0,
                "completeness_score": 0.0,
                "missing_aspects": ["No documents retrieved"],
                "passes_threshold": False,
                "assessment_reasoning": "No documents were retrieved for this query."
            }
        
        # Format context
        context, _ = format_context_for_prompt(retrieved_documents)
        
        system_prompt = get_prompt("quality_assessor", "system")
        user_prompt = get_prompt("quality_assessor", "user").format(
            query=query,
            context=context[:6000]  # Limit context size
        )
        
        try:
            result = self.llm.complete_json(user_prompt, system_prompt)
            
            overall_score = float(result.get("overall_score", 0.5))
            
            assessment = {
                "overall_score": overall_score,
                "relevance_score": float(result.get("relevance_score", overall_score)),
                "completeness_score": float(result.get("completeness_score", overall_score)),
                "specificity_score": float(result.get("specificity_score", overall_score)),
                "missing_aspects": result.get("missing_aspects", []),
                "relevant_excerpts": result.get("relevant_excerpts", []),
                "passes_threshold": overall_score >= self.quality_threshold,
                "assessment_reasoning": result.get("assessment_reasoning", "")
            }
            
            status = "PASS" if assessment["passes_threshold"] else "NEEDS REWRITE"
            log_agent_decision(
                logger, self.name,
                {"query": query[:50], "num_docs": len(retrieved_documents)},
                {"score": overall_score, "status": status},
                f"Quality score {overall_score:.2f} - {status}"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                "overall_score": 0.5,
                "relevance_score": 0.5,
                "completeness_score": 0.5,
                "missing_aspects": [],
                "passes_threshold": False,
                "assessment_reasoning": f"Assessment failed: {e}",
                "error": str(e)
            }
