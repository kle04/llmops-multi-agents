
import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

# RAGAs wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval using RAGAs.")
    parser.add_argument("--dataset-file", default="src/evaluation/rag/retrieval_dataset.json", help="Path to retrieval dataset JSON")
    parser.add_argument("--output-file", default="src/evaluation/rag/evaluation_results.csv", help="Path to output CSV")
    
    args = parser.parse_args()

    # 1. Load Dataset
    file_path = Path(args.dataset_file)
    if not file_path.exists():
        logger.error(f"Dataset file not found: {file_path}")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")

    # 2. Prepare for RAGAs
    # RAGAs expects "user_input" (or "question"), "retrieved_contexts" (or "contexts"), "response" (or "answer"), "reference" (or "ground_truth")
    # Mapping keys if necessary. My json has: question, ground_truth, contexts.
    # context_precision requires: question, ground_truth, contexts
    # context_recall requires: ground_truth, contexts
    
    # RAGAs < 0.2 used 'question', 'contexts', 'ground_truth'. 
    # RAGAs >= 0.2 usually standardizes on 'user_input', 'retrieved_contexts', 'reference'.
    # But it also supports 'question', 'contexts', 'ground_truth' for backward compat or auto-mapping.
    # Let's ensure strict naming for safety:
    formatted_data = []
    for item in data:
        formatted_data.append({
            "user_input": item["question"],
            "retrieved_contexts": item["contexts"],
            "reference": item["ground_truth"]
        })
    
    ragas_dataset = Dataset.from_list(formatted_data)

    # 3. Init LLM & Embeddings
    logger.info("Initializing LLM & Embeddings for RAGAs...")
    
    # LLM (Gemini)
    google_llm = ChatGoogleGenerativeAI(
        model=Config.GOOGLE_LLM_MODEL,
        temperature=0,
        google_api_key=Config.GOOGLE_API_KEY
    )
    evaluator_llm = LangchainLLMWrapper(google_llm)

    # Embeddings (HF)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={'normalize_embeddings': True}
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # 4. Evaluate
    logger.info("Running RAGAs evaluation...")
    metrics = [context_precision, context_recall]
    
    results = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 5. Process Results & Calculate F1
    df = results.to_pandas()
    
    # Calculate F1 Score (Harmonic Mean of Precision & Recall)
    # F1 = 2 * (P * R) / (P + R)
    # RAGAs columns usually: 'context_precision', 'context_recall'
    
    def calculate_f1(row):
        p = row.get('context_precision', 0)
        r = row.get('context_recall', 0)
        if (p + r) == 0:
            return 0
        return 2 * (p * r) / (p + r)

    df['f1_score'] = df.apply(calculate_f1, axis=1)

    # 6. Save & Report
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Evaluation complete. Results saved to {output_path}")
    
    # Print summary
    mean_scores = df[['context_precision', 'context_recall', 'f1_score']].mean()
    print("\n--- Evaluation Summary ---")
    print(mean_scores)

if __name__ == "__main__":
    main()
