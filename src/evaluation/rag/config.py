import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mental_health_advisor")
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    
    # Retrieval settings
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # RAGAs Evaluation settings
    RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gemini-1.5-flash")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

