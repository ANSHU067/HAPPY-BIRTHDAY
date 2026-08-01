"""Base provider interface for LLM integrations.

This module defines the abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """Standard response format from LLM providers."""

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_models: Optional[List[str]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize the provider.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.provider_name = config.provider.value

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    async def generate_async(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate a response asynchronously.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """Stream response from the LLM.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional provider-specific parameters

        Yields:
            Response chunks
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        pass

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            True if configuration is valid
        """
        if not self.config.model:
            raise ValueError(f"Model not specified for {self.provider_name}")
        return True

    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for the provider.

        Args:
            messages: Standard message format

        Returns:
            Provider-specific message format
        """
        # Default implementation returns messages as-is
        return messages

    def parse_response(self, response: Any) -> LLMResponse:
        """Parse provider response to standard format.

        Args:
            response: Provider-specific response

        Returns:
            LLMResponse object
        """
        # To be implemented by subclasses
        raise NotImplementedError


class ProviderRegistry:
    """Registry for LLM providers."""

    _providers: Dict[str, type] = {}

    @classmethod
    def register(cls, provider: LLMProvider):
        """Decorator to register a provider.

        Args:
            provider: Provider enum value
        """

        def decorator(provider_class):
            cls._providers[provider.value] = provider_class
            return provider_class

        return decorator

    @classmethod
    def get_provider(cls, provider: LLMProvider) -> type:
        """Get provider class by name.

        Args:
            provider: Provider enum value

        Returns:
            Provider class
        """
        if provider.value not in cls._providers:
            raise ValueError(f"Provider {provider.value} not registered")
        return cls._providers[provider.value]

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
