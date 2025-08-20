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

def validate_embedding_model(model_name: str, device: str) -> None:
    """
    Validate that the embedding model can be loaded.
    
    Args:
        model_name: Name/path of the embedding model
        device: Device to load model on
        
    Raises:
        ConfigurationError: If model cannot be loaded
    """
    try:
        from sentence_transformers import SentenceTransformer
        # Test load without actually loading to save memory
        logger.info(f"Validating model: {model_name}")
        # Just instantiate to check if model exists
        test_model = SentenceTransformer(model_name, device=device)
        embedding_dim = test_model.get_sentence_embedding_dimension()
        logger.info(f"Model validation successful. Embedding dimension: {embedding_dim}")
        del test_model  # Free memory
    except Exception as e:
        raise ConfigurationError(f"Failed to load embedding model '{model_name}': {e}")

def process_single_document(file_path: Path, out_dir: Path, model, 
                          batch_size: int, normalize: bool, 
                          device: str) -> Dict[str, Any]:
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
        
        # Generate embeddings with progress bar
        try:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=False,  # Manual normalization for consistency
                show_progress_bar=True,
                convert_to_numpy=True
            ).astype(np.float32)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
        
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
        
        # Save embeddings
        out_path = out_dir / f"{doc_id}.npz"
        try:
            np.savez_compressed(
                out_path, 
                ids=np.array(ids, dtype='<U64'),  # Unicode string array
                vectors=vectors
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to save embeddings to {out_path}: {e}")
        
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
    parser.add_argument("--model", default="AITeamVN/Vietnamese_Embedding_v2",
                       help="Embedding model name or path")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for encoding")
    parser.add_argument("--normalize", action="store_true",
                       help="L2-normalize embedding vectors")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1 for GPU)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                       help="Device to use for embeddings")
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
        
        # Validate model
        validate_embedding_model(args.model, device)
        
        # Find chunk files
        chunk_files = sorted(chunks_dir.glob("*.jsonl"))
        if not chunk_files:
            logger.error(f"No chunk files found in {chunks_dir}. Run chunking step first.")
            return 1
        
        logger.info(f"Found {len(chunk_files)} chunk files to process")
        
        # Load model once
        logger.info(f"Loading embedding model: {args.model}")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.model, device=device)
            embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load model: {e}")
        
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
                    args.normalize, device
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
                        args.batch_size, args.normalize, device
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
        total_time = sum(s.get("processing_time", 0) for s in successful)
        avg_throughput = sum(s.get("throughput", 0) for s in successful) / max(len(successful), 1)
        
        logger.info(f"\n=== EMBEDDING SUMMARY ===")
        logger.info(f"Total documents: {len(all_stats)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total chunks embedded: {total_chunks}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Average throughput: {avg_throughput:.1f} chunks/sec")
        logger.info(f"Device used: {device}")
        logger.info(f"Model: {args.model}")
        
        if successful:
            sample_dim = successful[0].get("embedding_dim", "unknown")
            logger.info(f"Embedding dimension: {sample_dim}")
        
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