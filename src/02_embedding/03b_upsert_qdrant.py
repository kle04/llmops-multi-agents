#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upsert embeddings + payloads vào Qdrant với optimized performance và error handling.

YÊU CẦU DỮ LIỆU:
- data/embeddings/<doc_id>.npz
    - chứa: ids[] (list[str]), vectors (np.ndarray: N x D)
- data/processed/chunks/<doc_id>.jsonl
    - mỗi dòng là 1 payload JSON, có key "id" khớp với ids[]

VÍ DỤ CHẠY (local):
python src/02_embedding/03b_upsert_qdrant.py \
  --emb_dir data/embeddings \
  --chunks_dir data/processed/chunks \
  --qdrant_url http://localhost:6333 \
  --collection mental_health_vi \
  --distance cosine \
  --batch_size 256

VÍ DỤ CHẠY (docker compose):
cd src
docker compose build upsert
docker compose run --rm upsert
"""

import argparse
import json
import logging
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterator

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
class QdrantUpsertError(Exception):
    """Raised when Qdrant upsert operations fail"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo
    from qdrant_client.http import models
except ImportError as e:
    logger.error(
        "Missing qdrant-client library. Install with:\n"
        "  pip install qdrant-client\n"
    )
    raise ConfigurationError(f"Missing qdrant-client: {e}")

# Constants
DEFAULT_UUID_NAMESPACE = uuid.NAMESPACE_URL
MAX_RETRIES = 3
RETRY_DELAY = 1.0

# --------------------------
# Helpers
# --------------------------

def to_uuid5_str(s: str, namespace: uuid.UUID = DEFAULT_UUID_NAMESPACE) -> str:
    """
    Convert string to deterministic UUID v5 string.
    
    Args:
        s: Input string to convert
        namespace: UUID namespace to use
        
    Returns:
        UUID string
    """
    try:
        # If already a valid UUID, return standardized string
        return str(uuid.UUID(s))
    except ValueError:
        return str(uuid.uuid5(namespace, s))

def validate_vectors(vectors: np.ndarray, doc_id: str) -> None:
    """
    Validate vector data quality.
    
    Args:
        vectors: Vector array to validate
        doc_id: Document ID for error reporting
        
    Raises:
        DataValidationError: If vectors are invalid
    """
    if vectors.size == 0:
        raise DataValidationError(f"{doc_id}: Empty vectors array")
    
    if np.any(np.isnan(vectors)):
        raise DataValidationError(f"{doc_id}: Vectors contain NaN values")
    
    if np.any(np.isinf(vectors)):
        raise DataValidationError(f"{doc_id}: Vectors contain infinite values")
    
    # Check reasonable value ranges
    vector_min, vector_max = vectors.min(), vectors.max()
    if abs(vector_min) > 1000 or abs(vector_max) > 1000:
        logger.warning(f"{doc_id}: Unusual vector values detected: min={vector_min:.3f}, max={vector_max:.3f}")

def load_embeddings(npz_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings from NPZ file with validation.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Tuple of (ids, vectors)
        
    Raises:
        QdrantUpsertError: If file cannot be loaded
        DataValidationError: If data is invalid
    """
    if not npz_path.exists():
        raise QdrantUpsertError(f"Embeddings file not found: {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if "ids" not in data:
            raise DataValidationError(f"Missing 'ids' in {npz_path}")
        if "vectors" not in data:
            raise DataValidationError(f"Missing 'vectors' in {npz_path}")
        
        ids = data["ids"].tolist()
        vectors = data["vectors"].astype(np.float32)
        
        if len(ids) != len(vectors):
            raise DataValidationError(f"IDs count ({len(ids)}) != vectors count ({len(vectors)}) in {npz_path}")
        
        if len(ids) == 0:
            raise DataValidationError(f"No embeddings found in {npz_path}")
        
        # Validate vector quality
        validate_vectors(vectors, npz_path.stem)
        
        logger.debug(f"Loaded {len(ids)} embeddings from {npz_path}")
        return ids, vectors
        
    except Exception as e:
        if isinstance(e, (DataValidationError, QdrantUpsertError)):
            raise
        raise QdrantUpsertError(f"Failed to load embeddings from {npz_path}: {e}")

def load_payloads(chunks_path: Path) -> Dict[str, dict]:
    """
    Load payloads from JSONL file with error handling.
    
    Args:
        chunks_path: Path to chunks JSONL file
        
    Returns:
        Dictionary mapping chunk IDs to payloads
        
    Raises:
        QdrantUpsertError: If file cannot be loaded
    """
    if not chunks_path.exists():
        logger.warning(f"Chunks file not found: {chunks_path}")
        return {}
    
    payloads: Dict[str, dict] = {}
    
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    obj = json.loads(line)
                    chunk_id = obj.get("id")
                    
                    if not chunk_id:
                        logger.warning(f"{chunks_path}:{line_num} - Missing 'id' field")
                        continue
                    
                    payloads[chunk_id] = obj
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"{chunks_path}:{line_num} - Invalid JSON: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"{chunks_path}:{line_num} - Error processing line: {e}")
                    continue
                    
    except Exception as e:
        raise QdrantUpsertError(f"Failed to load payloads from {chunks_path}: {e}")
    
    logger.debug(f"Loaded {len(payloads)} payloads from {chunks_path}")
    return payloads

def validate_qdrant_connection(client: QdrantClient) -> None:
    """
    Validate Qdrant connection and health.
    
    Args:
        client: Qdrant client instance
        
    Raises:
        QdrantUpsertError: If connection fails
    """
    try:
        # Test connection by getting collections list
        collections = client.get_collections()
        logger.info(f"Qdrant connection successful. Found {len(collections.collections)} collections")
    except Exception as e:
        raise QdrantUpsertError(f"Failed to connect to Qdrant: {e}")

def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    distance: str = "cosine",
    recreate: bool = False,
) -> CollectionInfo:
    """
    Ensure collection exists with proper configuration.
    
    Args:
        client: Qdrant client instance
        name: Collection name
        dim: Vector dimension
        distance: Distance metric
        recreate: Whether to recreate if exists
        
    Returns:
        Collection info
        
    Raises:
        QdrantUpsertError: If collection operations fail
    """
    distance_map = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
        "euclidean": Distance.EUCLID,
        "l2": Distance.EUCLID,
        "manhattan": Distance.MANHATTAN,
    }
    
    distance_metric = distance_map.get(distance.lower())
    if distance_metric is None:
        raise ConfigurationError(f"Unsupported distance metric: {distance}. "
                               f"Supported: {list(distance_map.keys())}")

    try:
        # Check if collection exists
        exists = False
        try:
            if hasattr(client, 'collection_exists'):
                exists = client.collection_exists(name)
            else:
                # Fallback for older clients
                client.get_collection(name)
                exists = True
        except Exception:
            exists = False

        # Handle recreation
        if recreate and exists:
            try:
                client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
                exists = False
            except Exception as e:
                logger.warning(f"Failed to delete collection {name}: {e}")

        # Create collection if needed
        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=distance_metric),
            )
            logger.info(f"Created collection: {name} (dim={dim}, distance={distance})")
        else:
            logger.info(f"Collection already exists: {name}")

        # Verify collection configuration
        collection_info = client.get_collection(name)
        
        # Validate dimension
        actual_dim = collection_info.config.params.vectors.size  # type: ignore
        if actual_dim != dim:
            raise QdrantUpsertError(
                f"Collection {name} has dimension {actual_dim}, expected {dim}. "
                f"Use --recreate to fix this."
            )
        
        return collection_info
        
    except Exception as e:
        if isinstance(e, (QdrantUpsertError, ConfigurationError)):
            raise
        raise QdrantUpsertError(f"Failed to ensure collection {name}: {e}")

def retry_on_failure(func, max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """
    Retry wrapper for network operations.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries
        
    Returns:
        Function result
        
    Raises:
        Exception: Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                delay *= 1.5  # Exponential backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception

def batched(iterable: Iterator, batch_size: int) -> Iterator[List]:
    """
    Create batches from an iterable.
    
    Args:
        iterable: Input iterable
        batch_size: Maximum batch size
        
    Yields:
        Batches of items
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_single_document(npz_path: Path, chunks_dir: Path, client: QdrantClient,
                          collection_name: str, batch_size: int, 
                          namespace: uuid.UUID = DEFAULT_UUID_NAMESPACE) -> Dict[str, Any]:
    """
    Process a single document for upsert.
    
    Args:
        npz_path: Path to embeddings NPZ file
        chunks_dir: Directory containing chunk files
        client: Qdrant client instance
        collection_name: Target collection name
        batch_size: Batch size for upsert
        namespace: UUID namespace for ID generation
        
    Returns:
        Processing statistics
    """
    start_time = time.time()
    doc_id = npz_path.stem
    
    stats = {
        "doc_id": doc_id,
        "status": "success",
        "points_upserted": 0,
        "missing_payloads": 0,
        "processing_time": 0,
        "throughput": 0,
        "errors": []
    }
    
    try:
        # Load embeddings
        ids, vectors = load_embeddings(npz_path)
        
        # Load payloads
        chunks_path = chunks_dir / f"{doc_id}.jsonl"
        payloads = load_payloads(chunks_path)
        
        # Build points
        points: List[PointStruct] = []
        missing_count = 0
        
        for i, chunk_id in enumerate(ids):
            # Generate deterministic UUID
            point_id = to_uuid5_str(chunk_id, namespace)
            
            # Get payload
            payload = payloads.get(chunk_id)
            if payload is None:
                missing_count += 1
                # Create minimal payload
                payload = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_index": i
                }
            
            # Create point
            points.append(PointStruct(
                id=point_id,
                vector=vectors[i].tolist(),
                payload=payload
            ))
        
        stats["missing_payloads"] = missing_count
        if missing_count > 0:
            logger.warning(f"{doc_id}: {missing_count}/{len(ids)} payloads missing, using minimal payloads")
        
        # Upsert in batches with retry
        points_sent = 0
        for batch in tqdm(list(batched(points, batch_size)), 
                         desc=f"Upserting {doc_id}",
                         leave=False):
            
            def upsert_batch():
                return client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            
            try:
                retry_on_failure(upsert_batch)
                points_sent += len(batch)
            except Exception as e:
                error_msg = f"Failed to upsert batch: {e}"
                stats["errors"].append(error_msg)
                logger.error(f"{doc_id}: {error_msg}")
                # Continue with next batch
        
        stats["points_upserted"] = points_sent
        stats["processing_time"] = time.time() - start_time
        stats["throughput"] = points_sent / max(stats["processing_time"], 0.001)
        
        logger.info(f"Successfully processed {doc_id}: {points_sent} points, "
                   f"{stats['processing_time']:.2f}s, {stats['throughput']:.1f} points/sec")
        
    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        stats["processing_time"] = time.time() - start_time
        logger.error(f"Failed to process {doc_id}: {e}")
        
        # Log full traceback for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full traceback for {doc_id}:\n{traceback.format_exc()}")
    
    return stats

def get_collection_stats(client: QdrantClient, collection_name: str) -> Dict[str, Any]:
    """
    Get collection statistics.
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        
    Returns:
        Collection statistics
    """
    try:
        collection_info = client.get_collection(collection_name)
        return {
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status,
        }
    except Exception as e:
        logger.warning(f"Failed to get collection stats: {e}")
        return {}

# --------------------------
# Main
# --------------------------

def main():
    """Main function with improved error handling and monitoring."""
    parser = argparse.ArgumentParser(
        description="Upsert embeddings to Qdrant with optimized performance and monitoring"
    )
    parser.add_argument("--emb_dir", default="data/embeddings",
                       help="Directory containing NPZ embedding files")
    parser.add_argument("--chunks_dir", default="data/processed/chunks",
                       help="Directory containing chunk JSONL files")
    parser.add_argument("--qdrant_url", default="http://localhost:6333",
                       help="Qdrant server URL")
    parser.add_argument("--qdrant_key", default=None,
                       help="Qdrant API key (optional)")
    parser.add_argument("--collection", default="mental_health_vi",
                       help="Target collection name")
    parser.add_argument("--distance", default="cosine",
                       help="Distance metric (cosine, dot, euclid, manhattan)")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate collection if it exists")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for upsert operations")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers")
    parser.add_argument("--dry_run", action="store_true",
                       help="Dry run mode - validate data without upserting")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Convert to Path objects
    emb_dir = Path(args.emb_dir)
    chunks_dir = Path(args.chunks_dir)
    
    try:
        # Validate configuration
        if not emb_dir.exists():
            raise ConfigurationError(f"Embeddings directory not found: {emb_dir}")
        
        if not chunks_dir.exists():
            raise ConfigurationError(f"Chunks directory not found: {chunks_dir}")
        
        if args.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive: {args.batch_size}")
        
        # Find embedding files
        embedding_files = sorted(emb_dir.glob("*.npz"))
        if not embedding_files:
            logger.error(f"No NPZ files found in {emb_dir}")
            return 1
        
        logger.info(f"Found {len(embedding_files)} embedding files to process")
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {args.qdrant_url}")
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_key)
        
        # Validate connection
        validate_qdrant_connection(client)
        
        # Determine vector dimension from first file
        first_file = embedding_files[0]
        _, first_vectors = load_embeddings(first_file)
        vector_dim = first_vectors.shape[1]
        logger.info(f"Vector dimension: {vector_dim}")
        
        # Validate dimension consistency
        for npz_path in embedding_files[1:]:
            _, vectors = load_embeddings(npz_path)
            if vectors.shape[1] != vector_dim:
                raise DataValidationError(
                    f"Inconsistent dimensions: {npz_path.stem} has dim={vectors.shape[1]}, "
                    f"expected {vector_dim}"
                )
        
        if not args.dry_run:
            # Ensure collection exists
            collection_info = ensure_collection(
                client, args.collection, vector_dim, args.distance, args.recreate
            )
            logger.info(f"Collection ready: {args.collection}")
            
            # Get initial collection stats
            initial_stats = get_collection_stats(client, args.collection)
            if initial_stats:
                logger.info(f"Initial collection stats: {initial_stats['points_count']} points")
        else:
            logger.info("DRY RUN MODE - No data will be upserted")
        
        # Process files
        all_stats = []
        namespace = uuid.uuid5(DEFAULT_UUID_NAMESPACE, args.collection)
        
        if args.max_workers == 1:
            # Sequential processing
            logger.info("Using sequential processing")
            for npz_path in tqdm(embedding_files, desc="Processing documents"):
                if args.dry_run:
                    # Just validate data
                    try:
                        load_embeddings(npz_path)
                        load_payloads(chunks_dir / f"{npz_path.stem}.jsonl")
                        logger.info(f"Validated: {npz_path.stem}")
                    except Exception as e:
                        logger.error(f"Validation failed for {npz_path.stem}: {e}")
                else:
                    stats = process_single_document(
                        npz_path, chunks_dir, client, args.collection, 
                        args.batch_size, namespace
                    )
                    all_stats.append(stats)
        else:
            # Parallel processing (use with caution)
            logger.info(f"Using parallel processing with {args.max_workers} workers")
            logger.warning("Parallel processing may overwhelm Qdrant server")
            
            if not args.dry_run:
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            process_single_document, npz_path, chunks_dir, client,
                            args.collection, args.batch_size, namespace
                        ): npz_path.stem
                        for npz_path in embedding_files
                    }
                    
                    for future in tqdm(as_completed(future_to_file), 
                                     total=len(future_to_file),
                                     desc="Processing documents"):
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
        
        if not args.dry_run:
            # Print summary
            successful = [s for s in all_stats if s["status"] == "success"]
            failed = [s for s in all_stats if s["status"] != "success"]
            
            total_points = sum(s.get("points_upserted", 0) for s in successful)
            total_time = sum(s.get("processing_time", 0) for s in successful)
            avg_throughput = sum(s.get("throughput", 0) for s in successful) / max(len(successful), 1)
            
            logger.info(f"\n=== UPSERT SUMMARY ===")
            logger.info(f"Total documents: {len(all_stats)}")
            logger.info(f"Successful: {len(successful)}")
            logger.info(f"Failed: {len(failed)}")
            logger.info(f"Total points upserted: {total_points}")
            logger.info(f"Total processing time: {total_time:.2f}s")
            logger.info(f"Average throughput: {avg_throughput:.1f} points/sec")
            logger.info(f"Collection: {args.collection}")
            
            # Get final collection stats
            final_stats = get_collection_stats(client, args.collection)
            if final_stats:
                logger.info(f"Final collection stats: {final_stats['points_count']} points")
            
            if failed:
                logger.warning(f"Failed documents: {[s['doc_id'] for s in failed]}")
            
            return 0 if not failed else 1
        else:
            logger.info("Dry run completed successfully")
            return 0
            
    except (ConfigurationError, DataValidationError, QdrantUpsertError) as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
