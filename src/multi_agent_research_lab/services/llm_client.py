"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass, field
from typing import Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

# Token pricing per 1M tokens (approximate, update as needed)
TOKEN_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Provider-agnostic LLM client using OpenAI.

    Falls back to a mock response when no API key is configured.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self.temperature = temperature
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        """Lazy-init OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=get_settings().openai_api_key,
                timeout=get_settings().timeout_seconds,
            )
        return self._client

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on token usage."""
        prices = TOKEN_PRICES.get(self.model, TOKEN_PRICES["gpt-4o-mini"])
        total = (input_tokens / 1_000_000) * prices["input"] + (
            output_tokens / 1_000_000
        ) * prices["output"]
        return round(total, 6)

    def _mock_response(self, user_prompt: str) -> LLMResponse:
        """Return a mock response when no API key is available."""
        return LLMResponse(
            content=(
                f"[MOCK] Received query: {user_prompt[:100]}...\n"
                "Configure OPENAI_API_KEY in .env to get real responses."
            ),
            model=self.model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> LLMResponse:
        """Return a model completion with retry and cost tracking.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query.
            **kwargs: Additional parameters passed to the chat completion API.
        """
        api_key = get_settings().openai_api_key

        if not api_key:
            return self._mock_response(user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            usage = response.usage

            return LLMResponse(
                content=content,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                cost_usd=self._estimate_cost(
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
                if usage
                else None,
                model=self.model,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                },
            )
        except openai.AuthenticationError:
            raise RuntimeError(
                f"Authentication failed. Check your OPENAI_API_KEY in .env."
            ) from None
        except openai.RateLimitError as exc:
            raise RuntimeError(
                "Rate limit exceeded. Consider adding exponential backoff or "
                "using a slower model."
            ) from exc
        except openai.APIError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc
