"""Retrieval Router Agent."""

from typing import Any, Dict, List
from ..llm import get_llm, get_prompt
from ..utils import get_agent_logger, log_agent_decision

logger = get_agent_logger("retrieval_router")


class RetrievalRouterAgent:
    
    def __init__(self, available_doc_types: List[str] = None):
        self.llm = get_llm()
        self.name = "Retrieval Router"
        self.available_types = available_doc_types or ["pdf", "docx", "pptx", "excel", "txt"]
    
    def route(self, query: str, parsed_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine retrieval plan based on query analysis.
        
        Returns:
            Dict with target_indices, search_strategy, top_k, metadata_filters.
        """
        system_prompt = get_prompt("retrieval_router", "system")
        user_prompt = get_prompt("retrieval_router", "user").format(
            query=query,
            intent_type=parsed_intent.get("intent_type", "factual_lookup"),
            entities=parsed_intent.get("entities", []),
            complexity=parsed_intent.get("complexity_score", 0.5),
            available_types=self.available_types
        )
        
        try:
            result = self.llm.complete_json(user_prompt, system_prompt)
            
            plan = {
                "target_indices": result.get("target_indices", self.available_types),
                "search_strategy": result.get("search_strategy", "hybrid"),
                "top_k": min(max(result.get("top_k", 10), 5), 20),
                "metadata_filters": result.get("metadata_filters", {}),
                "reasoning": result.get("reasoning", "Default routing")
            }
            
            log_agent_decision(
                logger, self.name, {"query": query[:50]}, plan,
                f"Routing to {plan['target_indices']} with {plan['search_strategy']} strategy"
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return {
                "target_indices": self.available_types,
                "search_strategy": "hybrid",
                "top_k": 10,
                "metadata_filters": {},
                "reasoning": f"Default routing due to error: {e}"
            }
