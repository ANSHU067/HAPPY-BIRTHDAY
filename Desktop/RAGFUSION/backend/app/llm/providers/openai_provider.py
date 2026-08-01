"""OpenAI provider implementation.

This module provides integration with OpenAI's API.
"""

from typing import AsyncIterator, Dict, List

import tiktoken
from openai import AsyncOpenAI, OpenAI

from app.llm.providers.base import (BaseLLMProvider, LLMConfig, LLMProvider,
                                    LLMResponse, ProviderRegistry)


@ProviderRegistry.register(LLMProvider.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key)
        self.async_client = AsyncOpenAI(api_key=config.api_key)

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(config.model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from OpenAI.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse object
        """
        # Merge config with kwargs
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.config.presence_penalty
            ),
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Call API
        response = self.client.chat.completions.create(**params)

        # Parse response
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "id": response.id,
                "created": response.created,
            },
        )

    async def generate_async(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate a response asynchronously.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse object
        """
        # Merge config with kwargs
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.config.presence_penalty
            ),
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Call API
        response = await self.async_client.chat.completions.create(**params)

        # Parse response
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=response.choices[0].finish_reason,
            metadata={
                "id": response.id,
                "created": response.created,
            },
        )

    async def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[str]:
        """Stream response from OpenAI.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional OpenAI parameters

        Yields:
            Response chunks
        """
        # Merge config with kwargs
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        # Call API with streaming
        stream = await self.async_client.chat.completions.create(**params)

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        return len(self.encoding.encode(text))


if __name__ == "__main__":
    # Example usage
    import os

    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
    )

    provider = OpenAIProvider(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is RAG?"},
    ]

    response = provider.generate(messages)
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage}")
