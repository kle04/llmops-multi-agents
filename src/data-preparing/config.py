import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mental_health_advisor")
    
    # Embedding Model - Optimized for Vietnamese + Mental Health Terms
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    
    # Chunking Strategy - Optimized for Medical/Psychological Content
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    
    # Retrieval settings - Tuned for Mental Health Consultation
    TOP_K_DOCUMENTS = int(os.getenv("TOP_K_DOCUMENTS", "3"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
    
    # Advanced chunking options
    CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "recursive")
    OVERLAP_METHOD = os.getenv("OVERLAP_METHOD", "sentence")
    
    # Embedding optimization
    NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))  # Smaller for Vietnamese model
    
    # Domain-specific settings
    DOMAIN = "mental_health_advisor"

# Emergency contact information
EMERGENCY_CONTACTS = {
    "suicide_prevention_hotline": "113",
    "mental_health_hotline": "18001567",
    "youth_support_hotline": "15009",
    "unicef_support": "18001524"
}
