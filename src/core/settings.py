from enum import Enum
from typing import Any, List, Optional

from dotenv import find_dotenv
from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class OpenAIModel(str, Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"


class AnthropicModel(str, Enum):
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class GoogleModel(str, Enum):
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"


# All available models
ModelType = OpenAIModel | AnthropicModel | GoogleModel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )
    
    # General configuration
    MODE: Optional[str] = None
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    AUTH_SECRET: Optional[SecretStr] = None

    # API keys for different providers
    OPENAI_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    GOOGLE_API_KEY: Optional[SecretStr] = None

    # Default model configuration
    DEFAULT_MODEL: str = OpenAIModel.GPT_4O_MINI.value
    AVAILABLE_MODELS: List[str] = []
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize available models based on provided API keys."""
        api_keys = {
            ModelProvider.OPENAI: self.OPENAI_API_KEY,
            ModelProvider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            ModelProvider.GOOGLE: self.GOOGLE_API_KEY,
        }
        
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        # Add available models based on provided API keys
        for provider in active_keys:
            if provider == ModelProvider.OPENAI:
                self.AVAILABLE_MODELS.extend([m.value for m in OpenAIModel])
            elif provider == ModelProvider.ANTHROPIC:
                self.AVAILABLE_MODELS.extend([m.value for m in AnthropicModel])
            elif provider == ModelProvider.GOOGLE:
                self.AVAILABLE_MODELS.extend([m.value for m in GoogleModel])
        
        # Set default model based on available models
        if self.DEFAULT_MODEL not in self.AVAILABLE_MODELS:
            if self.AVAILABLE_MODELS:
                self.DEFAULT_MODEL = self.AVAILABLE_MODELS[0]
    
    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"
    
    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()