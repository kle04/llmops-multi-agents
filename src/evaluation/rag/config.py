import os
from dotenv import load_dotenv
load_dotenv()



class Config:
    # Google API Key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key")
    GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemma-3n-e4b-it")
    GOOGLE_LLM_TEMPERATURE = os.getenv("GOOGLE_LLM_TEMPERATURE", 0.4)
    GOOGLE_LLM_MAX_OUTPUT_TOKENS = os.getenv("GOOGLE_LLM_MAX_OUTPUT_TOKENS", 2048)
    
    # Vector Database and Retrieval Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mental_health_advisor")
    TOP_K_DOCUMENTS = os.getenv("TOP_K_DOCUMENTS",5)
    SIMILARITY_THRESHOLD = os.getenv("SIMILARITY_THRESHOLD",0.7)

    RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:7005")

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 7005))
    
    # Hugging Face Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")