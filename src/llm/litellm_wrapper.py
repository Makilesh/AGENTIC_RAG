"""
LiteLLM Wrapper for Agentic RAG System.
"""

import json
import time
from typing import Any, Dict, List, Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils import config, get_llm_logger, get_gemini_api_key, log_llm_call, Timer

logger = get_llm_logger()

litellm.set_verbose = False


class LLMWrapper:
    
    def __init__(self):
        self.primary_model = config.llm.primary_model
        self.fallback_model = config.llm.fallback_model
        self.ollama_base_url = config.llm.ollama_base_url
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.timeout = config.llm.timeout
        self._last_model_used = None
        
        # Set API key
        api_key = get_gemini_api_key()
        if api_key:
            litellm.api_key = api_key
        
        logger.info(f"LLMWrapper initialized: primary={self.primary_model}, fallback={self.fallback_model}")
    
    @property
    def last_model_used(self) -> Optional[str]:
        return self._last_model_used
    
    def _call_primary(self, messages: List[Dict], **kwargs) -> str:
        """Call primary LLM (Gemini)."""
        response = litellm.completion(
            model=self.primary_model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=self.timeout
        )
        return response.choices[0].message.content
    
    def _call_fallback(self, messages: List[Dict], **kwargs) -> str:
        """Call fallback LLM (Ollama)."""
        # Add English instruction for Qwen
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += " IMPORTANT: Think and respond in English only."
        
        response = litellm.completion(
            model=self.fallback_model,
            api_base=self.ollama_base_url,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=self.timeout * 2
        )
        return response.choices[0].message.content
    
    def complete(
        self,
        prompt: str,
        system_message: str = "You are a helpful AI assistant.",
        **kwargs
    ) -> str:
        """
        Get LLM completion with automatic fallback.
        
        Args:
            prompt: User prompt.
            system_message: System message.
            **kwargs: Additional parameters (temperature, max_tokens).
            
        Returns:
            LLM response text.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        with Timer() as timer:
            try:
                response = self._call_primary(messages, **kwargs)
                self._last_model_used = self.primary_model
                is_fallback = False
            except Exception as e:
                logger.warning(f"Primary LLM failed: {e}. Falling back to Ollama...")
                try:
                    response = self._call_fallback(messages, **kwargs)
                    self._last_model_used = self.fallback_model
                    is_fallback = True
                except Exception as fallback_error:
                    logger.error(f"Both LLMs failed. Primary: {e}, Fallback: {fallback_error}")
                    raise RuntimeError(f"LLM unavailable: {e}")
        
        log_llm_call(
            logger, self._last_model_used, len(prompt), len(response),
            timer.elapsed_ms(), is_fallback
        )
        
        return response
    
    def complete_json(
        self,
        prompt: str,
        system_message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get LLM completion and parse as JSON."""
        response = self.complete(prompt, system_message, **kwargs)
        
        # Extract JSON from response
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
            if json_match:
                return json.loads(json_match.group(1))
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")
    
    def check_availability(self) -> Dict[str, bool]:
        """Check availability of LLMs."""
        status = {"primary": False, "fallback": False}
        
        try:
            self._call_primary([{"role": "user", "content": "test"}], max_tokens=5)
            status["primary"] = True
        except Exception as e:
            logger.debug(f"Primary LLM health check failed: {str(e)}")
        
        try:
            self._call_fallback([{"role": "user", "content": "test"}], max_tokens=5)
            status["fallback"] = True
        except Exception as e:
            logger.debug(f"Fallback LLM health check failed: {str(e)}")
        
        return status


# Global instance
_llm_wrapper: Optional[LLMWrapper] = None

def get_llm() -> LLMWrapper:
    """Get or create global LLM wrapper instance."""
    global _llm_wrapper
    if _llm_wrapper is None:
        _llm_wrapper = LLMWrapper()
    return _llm_wrapper
