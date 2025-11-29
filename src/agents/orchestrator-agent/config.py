import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def _parse_port(raw_value, default):
    """Handle plain ints or docker/k8s style tcp://host:port strings."""
    if raw_value in (None, ""):
        return default
    parsed = urlparse(str(raw_value))
    if parsed.scheme and parsed.port:
        return parsed.port
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default

class Config:
    # Google API Key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key")
    GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemma-3n-e4b-it")
    GOOGLE_LLM_TEMPERATURE = os.getenv("GOOGLE_LLM_TEMPERATURE", 0.2)
    GOOGLE_LLM_MAX_OUTPUT_TOKENS = os.getenv("GOOGLE_LLM_MAX_OUTPUT_TOKENS", 2048)


    # Server
    ORCHESTRATOR_AGENT_HOST = os.getenv("ORCHESTRATOR_AGENT_HOST", "0.0.0.0")
    ORCHESTRATOR_AGENT_PORT = _parse_port(os.getenv("ORCHESTRATOR_AGENT_PORT"), 7010)
    RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:7005")


    #REDIS
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", 6379)
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = os.getenv("REDIS_DB", 0)

    # POSTGRES
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = _parse_port(os.getenv("POSTGRES_PORT"), 5432)
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbotdb")

    # Short-term memory settings
    HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", 50))
    REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", 604800))  # 7 days
    MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", 2000))
