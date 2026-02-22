"""Multi-provider LLM client with support for Anthropic, OpenAI, and custom APIs."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import httpx
from ..config import settings


class LLMError(Exception):
    """Custom exception for LLM errors."""

    pass


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        # temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate text from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            # temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_text, usage) where usage has input_tokens and output_tokens
        """
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self, api_key: str, model: str, base_url: str = "https://api.anthropic.com"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        # temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, Dict[str, int]]:
        """Generate text using Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            # "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from response
                if "content" in data and len(data["content"]) > 0:
                    usage = data.get("usage", {})
                    return data["content"][0]["text"], {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    }
                else:
                    raise LLMError("No content in response")

        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise LLMError(f"Request error: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error: {e}")


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self, api_key: str, model: str, base_url: str = "https://api.openai.com"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        # temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, Dict[str, int]]:
        """Generate text using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            # "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from response
                if "choices" in data and len(data["choices"]) > 0:
                    usage = data.get("usage", {})
                    return data["choices"][0]["message"]["content"], {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    }
                else:
                    raise LLMError("No choices in response")

        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise LLMError(f"Request error: {e}")
        except Exception as e:
            raise LLMError(f"Unexpected error: {e}")


class CustomClient(LLMClient):
    """Custom/third-party API client (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        # temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[str, Dict[str, int]]:
        """Generate text using custom API (OpenAI-compatible format)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            # "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                # Debug: Check response content
                response_text = response.text
                if not response_text:
                    raise LLMError(
                        f"Empty response from API. Status: {response.status_code}"
                    )

                try:
                    data = response.json()
                except Exception as json_error:
                    raise LLMError(
                        f"Invalid JSON response. Content: {response_text[:500]}"
                    )

                # Extract text from response (OpenAI format)
                if "choices" in data and len(data["choices"]) > 0:
                    usage = data.get("usage", {})
                    return data["choices"][0]["message"]["content"], {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    }
                else:
                    raise LLMError(f"No choices in response. Response: {data}")

        except httpx.HTTPStatusError as e:
            raise LLMError(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise LLMError(f"Request error: {e}")
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Unexpected error: {e}")


def get_llm_client() -> LLMClient:
    """
    Get LLM client based on configuration.

    Returns:
        Configured LLM client instance
    """
    if settings.is_anthropic:
        return AnthropicClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    elif settings.is_openai:
        return OpenAIClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
    else:  # custom
        return CustomClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )
