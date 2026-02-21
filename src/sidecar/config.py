"""Configuration management for the multi-agent tutor system."""

from typing import Literal
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Provider Configuration
    llm_provider: Literal["custom", "anthropic", "openai"] = "custom"
    llm_base_url: str = ""
    llm_api_key: str
    llm_model: str = "claude-3-5-sonnet-20241022"

    @model_validator(mode="after")
    def _fill_default_base_url(self) -> "Settings":
        """Fill in a sensible base URL if the caller left it blank."""
        if not self.llm_base_url:
            self.llm_base_url = _PROVIDER_DEFAULT_BASE_URLS.get(self.llm_provider, "")
        return self

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Session Configuration
    max_input_length: int = 10000
    batch_error_log_every_n_turns: int = 5
    max_attempts_before_unlock: int = 3  # K value for instruction packet unlock

    # Memory / Context Configuration
    max_context_chars: int = 100000
    memory_reserved_chars: int = 50000
    memory_total_budget_chars: int = 20000
    memory_long_term_budget_chars: int = 4000
    memory_mid_term_budget_chars: int = 8000
    memory_short_term_budget_chars: int = 6000
    memory_long_term_max_chars: int = 4000
    memory_mid_term_chunk_max_chars: int = 1200
    memory_mid_term_input_chars: int = 12000
    memory_recent_turns: int = 6
    memory_mid_term_chunk_turns: int = 6
    memory_mid_term_max_chunks: int = 8
    memory_recent_turn_chars: int = 1500
    memory_mid_term_turn_chars: int = 1200

    # File Upload Configuration (v3.2.0)
    max_upload_size_mb: int = 5
    max_uploads_per_session: int = 10
    allowed_file_extensions: list[str] = ['.csv', '.xlsx', '.xls', '.json']

    @property
    def is_anthropic(self) -> bool:
        """Check if using Anthropic provider."""
        return self.llm_provider == "anthropic"

    @property
    def is_openai(self) -> bool:
        """Check if using OpenAI provider."""
        return self.llm_provider == "openai"

    @property
    def is_custom(self) -> bool:
        """Check if using custom provider."""
        return self.llm_provider == "custom"


# Global settings instance
settings = Settings()
