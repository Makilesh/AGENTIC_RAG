"""Validator Agent."""

from typing import Any, Dict
from ..llm import get_llm, get_prompt
from ..utils import get_agent_logger, log_agent_decision, config

logger = get_agent_logger("validator")


class ValidatorAgent:
    
    def __init__(self):
        self.llm = get_llm()
        self.name = "Validator"
        self.confidence_threshold = config.quality.confidence_threshold
    
    def validate(
        self,
        query: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Validate the generated answer against context.
        
        Returns:
            Dict with confidence_score, validation_status, and issues found.
        """
        system_prompt = get_prompt("validator", "system")
        user_prompt = get_prompt("validator", "user").format(
            query=query,
            answer=answer,
            context=context[:6000]
        )
        
        try:
            result = self.llm.complete_json(user_prompt, system_prompt)
            
            confidence = float(result.get("confidence_score", 0.5))
            
            validation = {
                "confidence_score": confidence,
                "validation_status": result.get("validation_status", "warning"),
                "has_citations": result.get("has_citations", False),
                "citation_accuracy": float(result.get("citation_accuracy", 0.5)),
                "hallucination_detected": result.get("hallucination_detected", False),
                "completeness": float(result.get("completeness", 0.5)),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "passes_threshold": confidence >= self.confidence_threshold,
                "final_verdict": result.get("final_verdict", "")
            }
            
            status = validation["validation_status"].upper()
            log_agent_decision(
                logger, self.name,
                {"query": query[:50]},
                {"confidence": confidence, "status": status},
                f"Validation {status} with confidence {confidence:.2f}"
            )
            
            return validation
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "confidence_score": 0.5,
                "validation_status": "warning",
                "has_citations": False,
                "hallucination_detected": False,
                "issues": [f"Validation error: {e}"],
                "passes_threshold": False,
                "error": str(e)
            }
