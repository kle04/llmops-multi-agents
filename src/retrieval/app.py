#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import logging
import time
from typing import List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mental Health Embedding API",
    description="Vietnamese text embedding service for mental health domain",
    version="1.0.0"
)

# Global variables for model
model = None
device = None

class TextRequest(BaseModel):
    text: str = Field(..., description="Text to vectorize", min_length=1, max_length=5000)

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to vectorize", min_items=1, max_items=100)

class EmbeddingResponse(BaseModel):
    vector: List[float]
    dimension: int
    processing_time: float

class BatchEmbeddingResponse(BaseModel):
    vectors: List[List[float]]
    dimension: int
    processing_time: float
    count: int

def detect_device() -> str:
    """Detect the best available device"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name}")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon) device")
        return "mps"
    else:
        logger.info("Using CPU device")
        return "cpu"

def load_model():
    """Load the embedding model with proper error handling"""
    global model, device
    
    model_name = os.getenv("EMBED_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
    device = detect_device()
    
    try:
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

def text2vec(text: str, normalize: bool = True) -> np.ndarray:
    """
    Convert text to vector using the loaded model
    
    Args:
        text: Input text
        normalize: Whether to L2-normalize the embedding
        
    Returns:
        Numpy array of embedding vector
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Use SentenceTransformers for better handling
        vector = model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Validate embedding
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            raise ValueError("Invalid embedding generated (NaN or Inf)")
            
        return vector
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise ValueError(f"Failed to generate embedding: {e}")

def batch_text2vec(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Convert multiple texts to vectors efficiently
    
    Args:
        texts: List of input texts
        normalize: Whether to L2-normalize the embeddings
        
    Returns:
        Numpy array of embedding vectors
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)
        
        # Validate embeddings
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            raise ValueError("Invalid embeddings generated (NaN or Inf)")
            
        return vectors
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        raise ValueError(f"Failed to generate embeddings: {e}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
        logger.info("Embedding service ready")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "embedding_dim": model.get_sentence_embedding_dimension() if model else None
    }

@app.post("/vectorize", response_model=EmbeddingResponse)
async def vectorize(request: TextRequest):
    """
    Generate embedding vector for a single text
    """
    start_time = time.time()
    
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        vector = text2vec(request.text.strip())
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            vector=vector.tolist(),
            dimension=len(vector),
            processing_time=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in vectorize: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/vectorize_batch", response_model=BatchEmbeddingResponse)
async def vectorize_batch(request: BatchTextRequest):
    """
    Generate embedding vectors for multiple texts efficiently
    """
    start_time = time.time()
    
    try:
        # Filter out empty texts
        valid_texts = [text.strip() for text in request.texts if text.strip()]
        
        if not valid_texts:
            raise HTTPException(status_code=400, detail="No valid texts provided")
            
        vectors = batch_text2vec(valid_texts)
        processing_time = time.time() - start_time
        
        return BatchEmbeddingResponse(
            vectors=vectors.tolist(),
            dimension=vectors.shape[1],
            processing_time=processing_time,
            count=len(vectors)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in vectorize_batch: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return {
        "model_name": os.getenv("EMBED_MODEL", "AITeamVN/Vietnamese_Embedding_v2"),
        "device": device,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_sequence_length": getattr(model, 'max_seq_length', 'unknown')
    }

if __name__ == '__main__':
    # For development
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 5000)),
        log_level="info"
    )
