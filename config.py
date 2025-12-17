"""Configuration for the multi-agent system."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings."""

    # OpenAI settings
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-mini")
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

    # Agent-specific models (can be overridden at runtime)
    PM_MODEL: str = os.getenv("PM_MODEL", "gpt-5-mini")  # Defaults to LLM_MODEL if None
    DEV_MODEL: str = os.getenv("DEV_MODEL", "gpt-5-mini")
    QA_MODEL: str = os.getenv("QA_MODEL", "gpt-5-mini")

    # Agent-specific temperatures
    AGENT_TEMPERATURES: dict[str, float] = {
        "pm": float(os.getenv("PM_TEMPERATURE", "0.7")),
        "dev": float(os.getenv("DEV_TEMPERATURE", "0.3")),
        "qa": float(os.getenv("QA_TEMPERATURE", "0.2")),
    }

    # Workflow settings
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Please set it in .env file or environment variables.")


# Validate configuration on import
Config.validate()
