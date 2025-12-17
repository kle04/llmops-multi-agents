#!/usr/bin/env python3
"""
Prepare dataset for RAGAs evaluation.
This script creates a dataset with questions, ground truth answers, and retrieved contexts.
"""

import re
import json
import sys
import logging
from pathlib import Path
from typing import List, Dict
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data-preparing"))
from config import Config as DataConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QA_PATTERN = re.compile(
    r"(?P<label>Q|A)\s*(?P<index>\d+)\s*:\s*(?P<content>.+?)(?=(?:\nQ\d+:|\nA\d+:|\Z))",
    re.DOTALL,
)

def parse_raw_qa(raw_text: str) -> List[Dict[str, str]]:
    """Parse raw Q/A text into list of question-answer pairs."""
    matches = QA_PATTERN.findall(raw_text)
    buffer = {}
    dataset = []
    
    for label, idx, content in matches:
        content = content.strip()
        key = (label, idx)
        buffer[key] = content
    
    indices = sorted({int(idx) for (_, idx) in buffer.keys()})
    
    for idx in indices:
        question = buffer.get(("Q", str(idx)))
        answer = buffer.get(("A", str(idx)))
        if question:
            # Clean up answer - remove page references
            if answer:
                answer = re.sub(r'\([Tt]rang?\s+\d+.*?\)', '', answer).strip()
                answer = re.sub(r'\(trang\s+\d+.*?\)', '', answer).strip()
            
            dataset.append({
                "id": idx,
                "question": question,
                "ground_truth": answer if answer else "",
            })
    
    return dataset

def retrieve_contexts_for_question(
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    question: str,
    top_k: int = 5,
    threshold: float = 0.4
) -> List[str]:
    """Retrieve relevant contexts from Qdrant for a question."""
    try:
        # Generate query embedding
        normalize = getattr(DataConfig, 'NORMALIZE_EMBEDDINGS', True)
        query_embedding = embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        # Search Qdrant
        search_results = qdrant_client.search(
            collection_name=DataConfig.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=threshold,
            with_payload=True,
            with_vectors=False
        )
        
        # Extract contexts
        contexts = []
        for hit in search_results:
            content = hit.payload.get("content", "")
            if content:
                contexts.append(content)
        
        return contexts
        
    except (ValueError, RuntimeError, ConnectionError) as e:
        logger.error("Error retrieving contexts for question: %s", e)
        return []

def prepare_ragas_dataset(
    raw_qa_file: str,
    output_file: str,
    top_k: int = 5,
    threshold: float = 0.4
) -> None:
    """Prepare dataset for RAGAs evaluation."""
    
    logger.info("🚀 Starting RAGAs dataset preparation...")
    
    # 1. Load Q/A pairs
    logger.info("📖 Loading Q/A pairs from %s...", raw_qa_file)
    raw_qa_path = Path(raw_qa_file)
    if not raw_qa_path.exists():
        logger.error("❌ File not found: %s", raw_qa_path)
        return
    
    with open(raw_qa_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    qa_pairs = parse_raw_qa(raw_text)
    logger.info("✅ Parsed %d Q/A pairs", len(qa_pairs))
    
    # 2. Initialize Qdrant client
    logger.info("🔌 Connecting to Qdrant...")
    try:
        qdrant_client = QdrantClient(
            url=DataConfig.QDRANT_URL,
            api_key=DataConfig.QDRANT_API_KEY if DataConfig.QDRANT_API_KEY else None,
        )
        
        # Test connection
        collections = qdrant_client.get_collections()
        logger.info("✅ Connected to Qdrant. Found %d collections", len(collections.collections))
        
    except (ConnectionError, ValueError, RuntimeError) as e:
        logger.error("❌ Failed to connect to Qdrant: %s", e)
        return
    
    # 3. Initialize embedding model
    logger.info("🧮 Loading embedding model: %s...", DataConfig.EMBEDDING_MODEL)
    try:
        embedding_model = SentenceTransformer(
            DataConfig.EMBEDDING_MODEL,
            trust_remote_code=True
        )
        logger.info("✅ Embedding model loaded")
    except (ValueError, RuntimeError, OSError) as e:
        logger.error("❌ Failed to load embedding model: %s", e)
        return
    
    # 4. Retrieve contexts for each question
    logger.info("🔍 Retrieving contexts for %d questions...", len(qa_pairs))
    logger.info("   Top K: %d, Threshold: %.2f", top_k, threshold)
    
    dataset = []
    for i, item in enumerate(qa_pairs, 1):
        question = item["question"]
        logger.info("\n[%d/%d] Processing: %s...", i, len(qa_pairs), question[:60])
        
        # Retrieve contexts
        contexts = retrieve_contexts_for_question(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            question=question,
            top_k=top_k,
            threshold=threshold
        )
        
        if not contexts:
            logger.warning("   ⚠️  No contexts retrieved for this question")
        else:
            logger.info("   ✅ Retrieved %d contexts", len(contexts))
            logger.info("   📊 Context lengths: %s", [len(c) for c in contexts])
        
        # Prepare dataset entry
        # RAGAs format: question, contexts, ground_truth
        # Note: 'answer' will be generated during evaluation by RAG system
        dataset_entry = {
            "question": question,
            "ground_truth": item["ground_truth"],
            "contexts": contexts,
            # Additional metadata
            "question_id": item["id"],
            "num_contexts": len(contexts)
        }
        
        dataset.append(dataset_entry)
    
    # 5. Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info("\n✅ Dataset saved to %s", output_path)
    logger.info("📊 Dataset statistics:")
    logger.info("   - Total questions: %d", len(dataset))
    logger.info("   - Questions with contexts: %d", sum(1 for d in dataset if d['contexts']))
    logger.info("   - Questions without contexts: %d", sum(1 for d in dataset if not d['contexts']))
    avg_contexts = sum(d['num_contexts'] for d in dataset) / len(dataset) if dataset else 0
    logger.info("   - Average contexts per question: %.2f", avg_contexts)
    
    # Show sample
    if dataset:
        logger.info("\n📝 Sample entry:")
        sample = dataset[0]
        logger.info("   Question: %s...", sample['question'][:80])
        logger.info("   Ground truth length: %d chars", len(sample['ground_truth']))
        logger.info("   Contexts: %d", sample['num_contexts'])

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare dataset for RAGAs evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset with default settings
  python prepare_ragas_dataset.py
  
  # Use custom Q/A file
  python prepare_ragas_dataset.py --qa-file ../rag/raw_qa.txt
  
  # Custom retrieval settings
  python prepare_ragas_dataset.py --top-k 10 --threshold 0.3
        """
    )
    
    parser.add_argument(
        "--qa-file",
        type=str,
        default="raw_qa.txt",
        help="Path to raw Q/A file (default: raw_qa.txt in current directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="ragas_evaluation_dataset.json",
        help="Output dataset file (default: ragas_evaluation_dataset.json)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of contexts to retrieve per question (default: 5)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Similarity threshold for retrieval (default: 0.4)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    qa_file = script_dir / args.qa_file if not Path(args.qa_file).is_absolute() else Path(args.qa_file)
    output_file = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    # If qa_file doesn't exist in current dir, try parent rag folder
    if not qa_file.exists():
        parent_qa = script_dir.parent / "rag" / "raw_qa.txt"
        if parent_qa.exists():
            qa_file = parent_qa
            logger.info("Using Q/A file from parent directory: %s", qa_file)
    
    prepare_ragas_dataset(
        raw_qa_file=str(qa_file),
        output_file=str(output_file),
        top_k=args.top_k,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()

