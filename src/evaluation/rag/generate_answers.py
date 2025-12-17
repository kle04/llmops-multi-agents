#!/usr/bin/env python3
"""
Generate answers for RAGAs evaluation dataset using RAG Agent.
This script calls the RAG Agent for each question to generate answers.
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "agents" / "rag-agent"))
from agent import RAGAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_answers_for_dataset(
    dataset_file: str,
    output_file: str,
    use_existing_answers: bool = False
) -> None:
    """
    Generate answers for all questions in the dataset using RAG Agent.
    
    Args:
        dataset_file: Path to input dataset JSON file
        output_file: Path to output dataset JSON file with answers
        use_existing_answers: If True, skip questions that already have answers
    """
    logger.info("🚀 Starting answer generation...")
    
    # 1. Load dataset
    logger.info("📖 Loading dataset from %s...", dataset_file)
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        logger.error("❌ Dataset file not found: %s", dataset_path)
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info("✅ Loaded %d questions", len(dataset))
    
    # 2. Initialize RAG Agent
    logger.info("🤖 Initializing RAG Agent...")
    try:
        rag_agent = RAGAgent()
        logger.info("✅ RAG Agent initialized successfully")
    except Exception as e:
        logger.error("❌ Failed to initialize RAG Agent: %s", e)
        return
    
    # 3. Generate answers for each question
    logger.info("💬 Generating answers for %d questions...", len(dataset))
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, item in enumerate(dataset, 1):
        question = item.get("question", "")
        question_id = item.get("question_id", i)
        
        # Check if answer already exists
        if use_existing_answers and item.get("answer"):
            logger.info("[%d/%d] Question %d: Answer already exists, skipping...", 
                       i, len(dataset), question_id)
            skipped_count += 1
            continue
        
        logger.info("\n[%d/%d] Processing Question %d: %s...", 
                   i, len(dataset), question_id, question[:60])
        
        try:
            # Call RAG Agent
            result = rag_agent.invoke(query=question, user_context={})
            
            # Extract answer
            answer = result.get("answer", "")
            status = result.get("status", "unknown")
            
            if status == "error" or not answer:
                logger.warning("   ⚠️  Failed to generate answer (status: %s)", status)
                error_count += 1
                item["answer"] = ""  # Set empty answer
                item["answer_status"] = status
            else:
                logger.info("   ✅ Generated answer (%d chars)", len(answer))
                item["answer"] = answer
                item["answer_status"] = status
                item["answer_metadata"] = {
                    "relevant_documents_count": result.get("relevant_documents_count", 0),
                    "total_retrieved_count": result.get("total_retrieved_count", 0),
                    "processing_time": result.get("processing_time", 0.0)
                }
                updated_count += 1
            
        except Exception as e:
            logger.error("   ❌ Error generating answer: %s", e)
            error_count += 1
            item["answer"] = ""
            item["answer_status"] = "error"
            item["answer_error"] = str(e)
    
    # 4. Save updated dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info("\n✅ Dataset with answers saved to %s", output_path)
    logger.info("📊 Summary:")
    logger.info("   - Total questions: %d", len(dataset))
    logger.info("   - Answers generated: %d", updated_count)
    logger.info("   - Skipped (existing): %d", skipped_count)
    logger.info("   - Errors: %d", error_count)
    logger.info("   - Questions with answers: %d", sum(1 for d in dataset if d.get("answer")))
    
    # Show sample
    if dataset:
        sample = next((d for d in dataset if d.get("answer")), dataset[0])
        logger.info("\n📝 Sample entry:")
        logger.info("   Question: %s...", sample['question'][:80])
        logger.info("   Answer length: %d chars", len(sample.get('answer', '')))
        logger.info("   Answer status: %s", sample.get('answer_status', 'N/A'))

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate answers for RAGAs evaluation dataset using RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate answers for all questions
  python generate_answers.py --dataset ragas_evaluation_dataset.json --output ragas_evaluation_dataset_with_answers.json
  
  # Skip questions that already have answers
  python generate_answers.py --dataset ragas_evaluation_dataset.json --output ragas_evaluation_dataset_with_answers.json --use-existing
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ragas_evaluation_dataset.json",
        help="Input dataset file (default: ragas_evaluation_dataset.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="ragas_evaluation_dataset_with_answers.json",
        help="Output dataset file with answers (default: ragas_evaluation_dataset_with_answers.json)"
    )
    
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Skip questions that already have answers"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    dataset_file = script_dir / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_file = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    generate_answers_for_dataset(
        dataset_file=str(dataset_file),
        output_file=str(output_file),
        use_existing_answers=args.use_existing
    )

if __name__ == "__main__":
    main()

