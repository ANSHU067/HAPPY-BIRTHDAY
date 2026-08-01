"""Anthropic (Claude) provider stub.

This module will provide integration with Anthropic's Claude API.
TODO: Implement Anthropic provider following the BaseLLMProvider interface.
"""

from typing import AsyncIterator, Dict, List

from app.llm.providers.base import BaseLLMProvider, LLMConfig, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider - TO BE IMPLEMENTED."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # TODO: Initialize Anthropic client

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response - TO BE IMPLEMENTED."""
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def generate_async(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate async response - TO BE IMPLEMENTED."""
        raise NotImplementedError("Anthropic provider not yet implemented")

    async def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """Stream response - TO BE IMPLEMENTED."""
        raise NotImplementedError("Anthropic provider not yet implemented")
        yield ""  # Make it a generator

    def count_tokens(self, text: str) -> int:
        """Count tokens - TO BE IMPLEMENTED."""
        raise NotImplementedError("Anthropic provider not yet implemented")
