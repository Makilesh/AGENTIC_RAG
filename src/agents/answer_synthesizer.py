"""Answer Synthesizer Agent."""

from typing import Any, Dict, List
from ..llm import get_llm, get_prompt, format_context_for_prompt
from ..utils import get_agent_logger, log_agent_decision

logger = get_agent_logger("answer_synthesizer")


class AnswerSynthesizerAgent:
    
    def __init__(self):
        self.llm = get_llm()
        self.name = "Answer Synthesizer"
    
    def synthesize(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        low_quality_warning: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer from context with source citations.
        
        Returns:
            Dict with generated_answer, sources_used, and confidence indicators.
        """
        if not context_documents:
            return {
                "generated_answer": "I could not find any relevant information to answer your question.",
                "sources_used": [],
                "has_citations": False,
                "low_quality_warning": True
            }
        
        context, sources = format_context_for_prompt(context_documents)
        
        system_prompt = get_prompt("answer_synthesizer", "system")
        user_prompt = get_prompt("answer_synthesizer", "user").format(
            query=query,
            context=context,
            sources=sources
        )
        
        try:
            answer = self.llm.complete(user_prompt, system_prompt, max_tokens=2000)
            
            # Extract sources used
            sources_used = []
            for doc in context_documents[:5]:
                source_name = doc.get("metadata", {}).get("file_name", "Unknown")
                if source_name not in sources_used:
                    sources_used.append(source_name)
            
            result = {
                "generated_answer": answer,
                "sources_used": sources_used,
                "has_citations": "[Source:" in answer,
                "low_quality_warning": low_quality_warning,
                "context_chunks_used": len(context_documents)
            }
            
            log_agent_decision(
                logger, self.name,
                {"query": query[:50], "num_sources": len(context_documents)},
                {"answer_length": len(answer), "sources": sources_used},
                f"Generated {len(answer)} char answer from {len(sources_used)} sources"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return {
                "generated_answer": f"I encountered an error while generating the answer: {e}",
                "sources_used": [],
                "has_citations": False,
                "error": str(e)
            }
