#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class EmbeddingError(Exception):
    """Raised when embedding generation fails"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class QdrantError(Exception):
    """Raised when Qdrant operations fail"""
    pass

def detect_device() -> str:
    """
    Detect the best available device for embedding generation.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using CUDA device: {gpu_name}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon) device")
        else:
            device = "cpu"
            logger.info("Using CPU device")
        return device
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        return "cpu"

def load_chunks(chunks_path: Path) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load chunks from JSONL file with error handling.
    
    Args:
        chunks_path: Path to chunks JSONL file
        
    Returns:
        Tuple of (ids, texts, payloads)
        
    Raises:
        EmbeddingError: If file is invalid or empty
    """
    if not chunks_path.exists():
        raise EmbeddingError(f"Chunks file not found: {chunks_path}")
    
    ids, texts, payloads = [], [], []
    
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    
                    # Validate required fields
                    if "id" not in obj:
                        logger.warning(f"{chunks_path}:{line_num} - Missing 'id' field")
                        continue
                    if "text" not in obj:
                        logger.warning(f"{chunks_path}:{line_num} - Missing 'text' field")
                        continue
                    
                    text = obj["text"].strip()
                    if not text:
                        logger.warning(f"{chunks_path}:{line_num} - Empty text for id {obj['id']}")
                        continue
                    
                    ids.append(obj["id"])
                    texts.append(text)
                    payloads.append(obj)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"{chunks_path}:{line_num} - Invalid JSON: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"{chunks_path}:{line_num} - Error processing line: {e}")
                    continue
                    
    except Exception as e:
        raise EmbeddingError(f"Failed to read chunks file {chunks_path}: {e}")
    
    if not ids:
        raise EmbeddingError(f"No valid chunks found in {chunks_path}")
    
    logger.debug(f"Loaded {len(ids)} chunks from {chunks_path}")
    return ids, texts, payloads

def download_model_with_progress(model_name: str) -> str:
    """
    Download model with progress tracking and caching.
    
    Args:
        model_name: Name/path of the embedding model
        
    Returns:
        Local path to model
    """
    import os
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    
    # Check if model is already cached
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Create a safe model directory name
    safe_model_name = model_name.replace("/", "--")
    model_cache_path = cache_dir / f"models--{safe_model_name}"
    
    if model_cache_path.exists():
        logger.info(f"Model {model_name} found in cache: {model_cache_path}")
        return model_name
    
    logger.info(f"Downloading model {model_name} (this may take several minutes for first time)...")
    
    # Set environment variable to show download progress
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    
    try:
        # Pre-download the model
        logger.info("Starting model download...")
        test_model = SentenceTransformer(model_name)
        logger.info(f"Model download completed and cached at: {test_model._modules['0'].auto_model.config.name_or_path}")
        del test_model
        return model_name
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        raise

def validate_embedding_model(model_name: str, device: str) -> Tuple[str, int]:
    """
    Validate that the embedding model can be loaded and return model info.
    
    Args:
        model_name: Name/path of the embedding model
        device: Device to load model on
        
    Returns:
        Tuple of (model_name, embedding_dimension)
        
    Raises:
        ConfigurationError: If model cannot be loaded
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Download model first if needed
        model_name = download_model_with_progress(model_name)
        
        logger.info(f"Validating model: {model_name}")
        # Quick validation load
        test_model = SentenceTransformer(model_name, device=device)
        embedding_dim = test_model.get_sentence_embedding_dimension()
        logger.info(f"Model validation successful. Embedding dimension: {embedding_dim}")
        del test_model  # Free memory
        return model_name, embedding_dim
        
    except Exception as e:
        raise ConfigurationError(f"Failed to load embedding model '{model_name}': {e}")

def debug_text_issues(texts: List[str], doc_id: str) -> None:
    """Debug and log potential text issues"""
    logger.debug(f"Debugging text issues for {doc_id}")
    
    for i, text in enumerate(texts[:5]):  # Check first 5 texts
        text_len = len(text)
        char_count = len(text.encode('utf-8'))
        
        # Check for problematic characters
        has_null = '\x00' in text
        has_control_chars = any(ord(c) < 32 and c not in '\n\r\t' for c in text)
        
        logger.debug(f"Text {i}: length={text_len}, bytes={char_count}, "
                    f"null_bytes={has_null}, control_chars={has_control_chars}")
        
        if text_len > 1000:
            logger.debug(f"Text {i} preview: {repr(text[:200])}...")
        else:
            logger.debug(f"Text {i}: {repr(text)}")
        
        if i >= 2:  # Limit debug output
            break

class QdrantManager:
    """Manager for Qdrant operations"""
    
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
    
    def setup_collection(self, vector_size: int, recreate: bool = False) -> bool:
        """Setup or recreate Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                if recreate:
                    logger.info(f"Recreating collection '{self.collection_name}'...")
                    self.client.delete_collection(self.collection_name)
                else:
                    # Check vector size
                    collection_info = self.client.get_collection(self.collection_name)
                    current_size = collection_info.config.params.vectors.size
                    
                    if current_size != vector_size:
                        logger.warning(f"Collection has wrong vector size: {current_size}, expected: {vector_size}")
                        logger.info(f"Recreating collection with correct size...")
                        self.client.delete_collection(self.collection_name)
                    else:
                        logger.info(f"Collection '{self.collection_name}' exists with correct size: {current_size}")
                        return True
            
            # Create collection
            logger.info(f"Creating collection '{self.collection_name}' with vector size {vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Verify collection
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection created successfully: {collection_info.points_count} points")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            return False
    
    def upsert_points(self, points: List[PointStruct]) -> bool:
        """Upsert points to Qdrant collection"""
        try:
            if not points:
                return True
                
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert points: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

def process_single_document(file_path: Path, out_dir: Path, model, 
                          batch_size: int, normalize: bool, 
                          device: str, qdrant_manager: Optional[QdrantManager] = None) -> Dict[str, Any]:
    """
    Process a single document for embedding generation.
    
    Args:
        file_path: Path to chunks file
        out_dir: Output directory
        model: Loaded embedding model
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings
        device: Device being used
        
    Returns:
        Processing statistics dictionary
    """
    start_time = time.time()
    doc_id = file_path.stem
    
    stats = {
        "doc_id": doc_id,
        "status": "success",
        "chunks": 0,
        "chunks_upserted": 0,
        "embedding_dim": 0,
        "processing_time": 0,
        "throughput": 0,
        "errors": []
    }
    
    try:
        # Load chunks
        ids, texts, payloads = load_chunks(file_path)
        stats["chunks"] = len(texts)
        
        if not texts:
            raise EmbeddingError(f"No texts to embed in {file_path}")
        
        logger.info(f"Processing {doc_id}: {len(texts)} chunks")
        
        # Debug text issues if in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            debug_text_issues(texts, doc_id)
        
        # Clean and validate texts before embedding
        cleaned_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            # Clean text
            clean_text = text.strip()
            if not clean_text:
                logger.warning(f"{doc_id}: Empty text at index {i}, skipping")
                continue
            
            # Check text length (sentence-transformers usually has 512 token limit)
            # Truncate if too long
            if len(clean_text) > 8000:  # Rough character limit
                logger.warning(f"{doc_id}: Text too long at index {i} ({len(clean_text)} chars), truncating")
                clean_text = clean_text[:8000]
            
            # Advanced text cleaning for Vietnamese text
            try:
                # Step 1: Remove null bytes and other problematic bytes
                clean_text = clean_text.replace('\x00', ' ')
                
                # Step 2: Normalize Unicode (Vietnamese text often has combining characters)
                import unicodedata
                clean_text = unicodedata.normalize('NFC', clean_text)
                
                # Step 3: Remove control characters but keep essential whitespace
                clean_text = ''.join(char for char in clean_text 
                                    if ord(char) >= 32 or char in '\t\n\r')
                
                # Step 4: Remove problematic Unicode categories but keep Vietnamese letters
                allowed_categories = {'Ll', 'Lu', 'Lt', 'Lm', 'Lo', 'Nd', 'Nl', 'No', 
                                     'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po', 'Zs'}
                clean_text = ''.join(char for char in clean_text 
                                    if unicodedata.category(char) in allowed_categories or char in '\t\n\r')
                
                # Step 5: Handle specific problematic sequences for Vietnamese
                # Remove standalone numbers that might be page numbers
                if clean_text.strip() and len(clean_text.strip().split()) <= 5:
                    # Check if it's just numbers/short metadata
                    words = clean_text.strip().split()
                    if all(word.isdigit() or len(word) <= 2 for word in words):
                        logger.warning(f"{doc_id}: Skipping probable metadata at index {i}: {repr(clean_text[:50])}")
                        continue
                
                # Step 6: Normalize whitespace
                clean_text = ' '.join(clean_text.split())
                
                # Step 7: Final validation
                if len(clean_text.strip()) < 10:  # Skip very short texts
                    logger.warning(f"{doc_id}: Skipping very short text at index {i}: {repr(clean_text)}")
                    continue
                
                # Test encode/decode to catch any remaining problematic characters
                clean_text.encode('utf-8').decode('utf-8')
                
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError) as e:
                logger.warning(f"{doc_id}: Unicode error at index {i}, skipping: {e}")
                continue
            except Exception as e:
                logger.warning(f"{doc_id}: Text cleaning error at index {i}, skipping: {e}")
                continue
            
            if clean_text and len(clean_text.strip()) >= 10:
                cleaned_texts.append(clean_text)
                valid_indices.append(i)
        
        if not cleaned_texts:
            raise EmbeddingError(f"No valid texts after cleaning for {doc_id}")
        
        logger.info(f"{doc_id}: Processing {len(cleaned_texts)}/{len(texts)} valid texts")
        
        # Generate embeddings with error handling
        try:
            vectors = model.encode(
                cleaned_texts,
                batch_size=batch_size,
                normalize_embeddings=False,  # Manual normalization for consistency
                show_progress_bar=True,
                convert_to_numpy=True,
                convert_to_tensor=False
            ).astype(np.float32)
            
        except Exception as e:
            # Try with smaller batch size if failed
            logger.warning(f"{doc_id}: Failed with batch_size={batch_size}, trying with batch_size=1")
            try:
                vectors = model.encode(
                    cleaned_texts,
                    batch_size=1,
                    normalize_embeddings=False,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    convert_to_tensor=False
                ).astype(np.float32)
            except Exception as e2:
                raise EmbeddingError(f"Failed to generate embeddings even with batch_size=1: {e2}")
        
        # Update data to match cleaned texts
        ids = [ids[i] for i in valid_indices]
        texts = cleaned_texts
        payloads = [payloads[i] for i in valid_indices]
        
        if vectors.shape[0] != len(texts):
            raise EmbeddingError(f"Mismatch between input texts ({len(texts)}) and output vectors ({vectors.shape[0]})")
        
        stats["embedding_dim"] = vectors.shape[1]
        
        # Normalize if requested
        if normalize:
            logger.debug(f"Normalizing embeddings for {doc_id}")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms
            
            # Verify normalization
            sample_norms = np.linalg.norm(vectors[:min(10, len(vectors))], axis=1)
            if not np.allclose(sample_norms, 1.0, atol=1e-6):
                logger.warning(f"Normalization may have failed for {doc_id}")
        
        # Validate embeddings quality
        if np.any(np.isnan(vectors)) or np.any(np.isinf(vectors)):
            raise EmbeddingError(f"Invalid embeddings detected (NaN or Inf) for {doc_id}")
        
        # Check for reasonable embedding ranges
        vector_min, vector_max = vectors.min(), vectors.max()
        if abs(vector_min) > 100 or abs(vector_max) > 100:
            logger.warning(f"Unusual embedding values detected for {doc_id}: min={vector_min:.3f}, max={vector_max:.3f}")
        
        # Save embeddings to file (optional)
        if out_dir:
            out_path = out_dir / f"{doc_id}.npz"
            try:
                np.savez_compressed(
                    out_path, 
                    ids=np.array(ids, dtype='<U64'),  # Unicode string array
                    vectors=vectors
                )
                logger.debug(f"Saved embeddings to {out_path}")
            except Exception as e:
                logger.warning(f"Failed to save embeddings to {out_path}: {e}")
        
        # Upsert to Qdrant if available
        if qdrant_manager:
            try:
                # Create Qdrant points
                import uuid
                import hashlib
                
                points = []
                for i, (chunk_id, vector, payload) in enumerate(zip(ids, vectors, payloads)):
                    # Generate a valid Qdrant point ID (UUID from chunk_id hash)
                    # This ensures consistent IDs for the same chunk across runs
                    chunk_id_hash = hashlib.md5(chunk_id.encode('utf-8')).hexdigest()
                    point_id = str(uuid.UUID(chunk_id_hash))
                    
                    # Ensure proper data types for payload
                    clean_payload = {
                        "original_id": str(payload.get("id", chunk_id)),
                        "chunk_id": str(payload.get("chunk_id", chunk_id)),
                        "doc_id": str(payload.get("doc_id", "")),
                        "title": str(payload.get("title", "")),
                        "source": str(payload.get("source", "")),
                        "year": float(payload["year"]) if payload.get("year") and str(payload["year"]) not in ["", "nan", "None"] else None,
                        "language": str(payload.get("language", "vi")),
                        "audience": str(payload.get("audience", "")),
                        "grade_range": str(payload.get("grade_range", "")),
                        "topics": str(payload.get("topics", "")),
                        "section": str(payload.get("section", "")),
                        "text": str(payload.get("text", "")),
                        "token_count": int(payload.get("token_count", 0))
                    }
                    
                    point = PointStruct(
                        id=point_id,  # Use UUID instead of string
                        vector=vector.tolist(),  # Convert numpy array to list
                        payload=clean_payload
                    )
                    points.append(point)
                
                # Upsert in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    if qdrant_manager.upsert_points(batch):
                        stats["chunks_upserted"] += len(batch)
                    else:
                        raise QdrantError(f"Failed to upsert batch {i//batch_size + 1}")
                
                logger.info(f"Successfully upserted {stats['chunks_upserted']} points to Qdrant for {doc_id}")
                
            except Exception as e:
                logger.error(f"Failed to upsert to Qdrant for {doc_id}: {e}")
                # Continue processing even if Qdrant fails
        
        # Calculate statistics
        stats["processing_time"] = time.time() - start_time
        stats["throughput"] = len(texts) / max(stats["processing_time"], 0.001)  # chunks/second
        
        logger.info(f"Successfully processed {doc_id}: {len(texts)} chunks, "
                   f"dim={vectors.shape[1]}, {stats['processing_time']:.2f}s, "
                   f"{stats['throughput']:.1f} chunks/sec")
        
    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        stats["processing_time"] = time.time() - start_time
        logger.error(f"Failed to process {doc_id}: {e}")
        
        # Log full traceback for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full traceback for {doc_id}:\n{traceback.format_exc()}")
    
    return stats

def main():
    """Main function with improved error handling and device optimization."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from text chunks with optimized performance"
    )
    parser.add_argument("--chunks_dir", default="data/processed/chunks",
                       help="Directory containing chunk JSONL files")
    parser.add_argument("--out_dir", default="data/embeddings",
                       help="Output directory for embedding files")
    parser.add_argument("--model", default="AITeamVN/Vietnamese_Embedding",
                       help="Embedding model name or path")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding")
    parser.add_argument("--normalize", action="store_true",
                       help="L2-normalize embedding vectors")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1 for GPU)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                       help="Device to use for embeddings")
    parser.add_argument("--qdrant_url", default="http://localhost:6333",
                       help="Qdrant server URL (use 'none' to disable)")
    parser.add_argument("--collection_name", default="mental_health_vi",
                       help="Qdrant collection name")
    parser.add_argument("--recreate_collection", action="store_true",
                       help="Recreate collection even if it exists")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Convert to Path objects
    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)
    
    try:
        # Validate configuration
        if not chunks_dir.exists():
            raise ConfigurationError(f"Chunks directory not found: {chunks_dir}")
        
        if args.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive: {args.batch_size}")
        
        if args.max_workers <= 0:
            raise ConfigurationError(f"max_workers must be positive: {args.max_workers}")
        
        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        if args.device == "auto":
            device = detect_device()
        else:
            device = args.device
            logger.info(f"Using specified device: {device}")
        
        # Validate model and get info
        model_name, embedding_dim = validate_embedding_model(args.model, device)
        
        # Find chunk files
        chunk_files = sorted(chunks_dir.glob("*.jsonl"))
        if not chunk_files:
            logger.error(f"No chunk files found in {chunks_dir}. Run chunking step first.")
            return 1
        
        logger.info(f"Found {len(chunk_files)} chunk files to process")
        
        # Load model once (already validated and downloaded)
        logger.info(f"Loading embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
            actual_dim = model.get_sentence_embedding_dimension()
            if actual_dim != embedding_dim:
                logger.warning(f"Dimension mismatch: expected {embedding_dim}, got {actual_dim}")
                embedding_dim = actual_dim
            logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load model: {e}")
        
        # Initialize Qdrant manager if enabled
        qdrant_manager = None
        if args.qdrant_url.lower() != "none":
            try:
                qdrant_manager = QdrantManager(args.qdrant_url, args.collection_name)
                if not qdrant_manager.setup_collection(embedding_dim, recreate=args.recreate_collection):
                    logger.error("Failed to setup Qdrant collection")
                    return 1
                logger.info(f"Qdrant collection '{args.collection_name}' ready with {embedding_dim}D vectors")
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant: {e}")
                return 1
        else:
            logger.info("Qdrant disabled, only saving to files")
        
        # Process files
        all_stats = []
        
        # For embedding models, parallel processing on same GPU can be problematic
        # Use sequential processing by default, but allow parallel for CPU
        if args.max_workers == 1 or device in ["cuda", "mps"]:
            # Sequential processing (recommended for GPU)
            logger.info("Using sequential processing")
            for file_path in tqdm(chunk_files, desc="Embedding documents"):
                stats = process_single_document(
                    file_path, out_dir, model, args.batch_size, 
                    args.normalize, device, qdrant_manager
                )
                all_stats.append(stats)
        else:
            # Parallel processing (only for CPU)
            logger.info(f"Using parallel processing with {args.max_workers} workers")
            logger.warning("Parallel processing with embedding models can be memory intensive")
            
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        process_single_document, file_path, out_dir, model,
                        args.batch_size, args.normalize, device, qdrant_manager
                    ): file_path.stem
                    for file_path in chunk_files
                }
                
                for future in tqdm(as_completed(future_to_file), 
                                 total=len(future_to_file),
                                 desc="Embedding documents"):
                    doc_id = future_to_file[future]
                    try:
                        stats = future.result()
                        all_stats.append(stats)
                    except Exception as e:
                        logger.error(f"Task failed for {doc_id}: {e}")
                        all_stats.append({
                            "doc_id": doc_id,
                            "status": "failed",
                            "error": str(e)
                        })
        
        # Print summary
        successful = [s for s in all_stats if s["status"] == "success"]
        failed = [s for s in all_stats if s["status"] != "success"]
        
        total_chunks = sum(s.get("chunks", 0) for s in successful)
        total_chunks_upserted = sum(s.get("chunks_upserted", 0) for s in successful)
        total_time = sum(s.get("processing_time", 0) for s in successful)
        avg_throughput = sum(s.get("throughput", 0) for s in successful) / max(len(successful), 1)
        
        logger.info(f"\n=== EMBEDDING SUMMARY ===")
        logger.info(f"Total documents: {len(all_stats)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total chunks embedded: {total_chunks}")
        if qdrant_manager:
            logger.info(f"Total chunks upserted to Qdrant: {total_chunks_upserted}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Average throughput: {avg_throughput:.1f} chunks/sec")
        logger.info(f"Device used: {device}")
        logger.info(f"Model: {args.model}")
        
        if successful:
            sample_dim = successful[0].get("embedding_dim", "unknown")
            logger.info(f"Embedding dimension: {sample_dim}")
        
        # Get final Qdrant collection stats
        if qdrant_manager:
            final_stats = qdrant_manager.get_collection_stats()
            if final_stats:
                logger.info(f"Final Qdrant collection stats: {final_stats}")
        
        if failed:
            logger.warning(f"Failed documents: {[s['doc_id'] for s in failed]}")
        
        return 0 if not failed else 1
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())