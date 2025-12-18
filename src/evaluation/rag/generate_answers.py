#!/usr/bin/env python3
"""
Generate answers for RAGAs evaluation dataset using Orchestrator Agent endpoint.
This script calls the Orchestrator Agent API for each question to generate answers.
"""

import json
import sys
import logging
import httpx
from pathlib import Path
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_orchestrator_agent(
    question: str,
    base_url: str = "http://localhost:7010",
    timeout: float = 60.0
) -> Dict:
    """
    Call Orchestrator Agent endpoint to get answer for a question.
    
    Args:
        question: The question to ask
        base_url: Base URL of Orchestrator Agent
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary with answer and metadata
    """
    url = f"{base_url}/chat"
    
    payload = {
        "message": question,
        "user_id": "ragas_evaluation",
        "session_id": "evaluation_session"
    }
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            sources = result.get("sources")
            if sources is None:
                sources = []
            
            return {
                "answer": result.get("response", ""),
                "selected_agent": result.get("selected_agent"),
                "sources": sources,
                "error": result.get("error"),
                "status": "success" if not result.get("error") else "error"
            }
    except httpx.TimeoutException:
        logger.error("Request timeout")
        return {
            "answer": "",
            "selected_agent": None,
            "sources": [],
            "error": "Request timeout",
            "status": "timeout"
        }
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error: %s - %s", e.response.status_code, e.response.text)
        return {
            "answer": "",
            "selected_agent": None,
            "sources": [],
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
            "status": "error"
        }
    except Exception as e:
        logger.error("Error calling Orchestrator Agent: %s", e)
        return {
            "answer": "",
            "selected_agent": None,
            "sources": [],
            "error": str(e),
            "status": "error"
        }

def generate_answers_for_dataset(
    dataset_file: str,
    output_file: str,
    orchestrator_url: str = "http://localhost:7010",
    use_existing_answers: bool = False,
    timeout: float = 60.0
) -> None:
    """
    Generate answers for all questions in the dataset using Orchestrator Agent endpoint.
    
    Args:
        dataset_file: Path to input dataset JSON file
        output_file: Path to output dataset JSON file with answers
        orchestrator_url: Base URL of Orchestrator Agent
        use_existing_answers: If True, skip questions that already have answers
        timeout: Request timeout in seconds
    """
    logger.info("🚀 Starting answer generation...")
    
    # 1. Check Orchestrator Agent health
    logger.info("🏥 Checking Orchestrator Agent health at %s...", orchestrator_url)
    try:
        with httpx.Client(timeout=10.0) as client:
            health_response = client.get(f"{orchestrator_url}/health")
            health_response.raise_for_status()
            health_data = health_response.json()
            logger.info("✅ Orchestrator Agent is healthy: %s", health_data.get("status", "unknown"))
    except Exception as e:
        logger.warning("⚠️  Could not verify Orchestrator Agent health: %s", e)
        logger.warning("   Continuing anyway...")
    
    # 2. Load dataset
    logger.info("📖 Loading dataset from %s...", dataset_file)
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        logger.error("❌ Dataset file not found: %s", dataset_path)
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info("✅ Loaded %d questions", len(dataset))
    
    # 3. Generate answers for each question
    logger.info("💬 Generating answers for %d questions...", len(dataset))
    logger.info("   Orchestrator URL: %s", orchestrator_url)
    logger.info("   Timeout: %.1f seconds", timeout)
    
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
            # Call Orchestrator Agent endpoint
            result = call_orchestrator_agent(
                question=question,
                base_url=orchestrator_url,
                timeout=timeout
            )
            
            # Extract answer
            answer = result.get("answer", "")
            status = result.get("status", "unknown")
            
            if status in ["error", "timeout"] or not answer:
                logger.warning("   ⚠️  Failed to generate answer (status: %s)", status)
                if result.get("error"):
                    logger.warning("   Error: %s", result.get("error"))
                error_count += 1
                item["answer"] = ""  # Set empty answer
                item["answer_status"] = status
                item["answer_error"] = result.get("error", "")
            else:
                logger.info("   ✅ Generated answer (%d chars)", len(answer))
                logger.info("   Selected agent: %s", result.get("selected_agent", "unknown"))
                item["answer"] = answer
                item["answer_status"] = status
                sources = result.get("sources") or []
                item["answer_metadata"] = {
                    "selected_agent": result.get("selected_agent"),
                    "sources": sources,
                    "num_sources": len(sources)
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
        if sample.get('answer_metadata'):
            logger.info("   Selected agent: %s", sample['answer_metadata'].get('selected_agent', 'N/A'))

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate answers for RAGAs evaluation dataset using Orchestrator Agent endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate answers for all questions
  python generate_answers.py --dataset ragas_evaluation_dataset.json --output ragas_evaluation_dataset_with_answers.json
  
  # Use custom Orchestrator Agent URL
  python generate_answers.py --dataset ragas_evaluation_dataset.json --output ragas_evaluation_dataset_with_answers.json --url http://localhost:7010
  
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
        "--url",
        type=str,
        default="http://localhost:7010",
        help="Orchestrator Agent base URL (default: http://localhost:7010)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)"
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
        orchestrator_url=args.url,
        use_existing_answers=args.use_existing,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main()

