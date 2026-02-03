"""Query Analyzer Agent."""

from typing import Any, Dict
from ..llm import get_llm, get_prompt
from ..utils import get_agent_logger, log_agent_decision

logger = get_agent_logger("query_analyzer")


class QueryAnalyzerAgent:
    
    def __init__(self):
        self.llm = get_llm()
        self.name = "Query Analyzer"
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query and extract structured information.
        
        Returns:
            Dict with intent_type, entities, complexity_score, etc.
        """
        system_prompt = get_prompt("query_analyzer", "system")
        user_prompt = get_prompt("query_analyzer", "user").format(query=query)
        
        try:
            result = self.llm.complete_json(user_prompt, system_prompt)
            
            # Ensure required fields with defaults
            parsed = {
                "intent_type": result.get("intent_type", "factual_lookup"),
                "entities": result.get("entities", []),
                "concepts": result.get("concepts", []),
                "complexity_score": float(result.get("complexity_score", 0.5)),
                "implicit_filters": result.get("implicit_filters", {}),
                "requires_multi_doc": result.get("requires_multi_doc", False),
                "reformulated_query": result.get("reformulated_query", query)
            }
            
            log_agent_decision(
                logger, self.name, {"query": query}, parsed,
                f"Detected {parsed['intent_type']} query with complexity {parsed['complexity_score']:.2f}"
            )
            
            return parsed
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "intent_type": "factual_lookup",
                "entities": [],
                "concepts": [],
                "complexity_score": 0.5,
                "implicit_filters": {},
                "requires_multi_doc": False,
                "reformulated_query": query,
                "error": str(e)
            }
