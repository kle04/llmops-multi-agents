import re
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QA_PATTERN = re.compile(
    r"(?P<label>Q|A)\s*(?P<index>\d+)\s*:\s*(?P<content>.+?)(?=(?:\nQ\d+:|\nA\d+:|\Z))",
    re.DOTALL,
)

def parse_raw_qa(raw_text: str) -> List[Dict[str, str]]:
    """Parse raw text into list of {"question":..., "ground_truth":...}."""
    matches = QA_PATTERN.findall(raw_text)
    buffer = {}
    dataset = []
    for label, idx, content in matches:
        content = content.strip()
        key = (label, idx)
        buffer[key] = content

    indices = sorted(
        {int(idx) for (_, idx) in buffer.keys()},
    )

    for idx in indices:
        question = buffer.get(("Q", str(idx)))
        answer = buffer.get(("A", str(idx)))
        if question: # Relaxed condition: ground_truth might be missing in some raw files, but here we have it.
            dataset.append(
                {
                    "id": idx,
                    "question": question,
                    "ground_truth": answer if answer else "",
                }
            )
    return dataset

def filter_documents(llm, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter documents using LLM grading."""
    relevant_docs = []
    logger.info(f"Filtering {len(docs)} documents with LLM grading")
    
    for doc in docs:
        check_prompt = f"""
        Have a look at the text below and decide if it is relevant to the question?

        Question: {query}

        Text: {doc["content"][:500]}...
        
        Answer "YES" if relevant, "NO" if not.
        """
        
        try:
            messages = [HumanMessage(content=check_prompt)]
            response = llm.invoke(messages)
            grade = response.content.strip().upper()

            if "YES" in grade:
                logger.info(f"Document accepted (score: {doc.get('score', 0):.3f})")
                relevant_docs.append(doc)
            elif "NO" in grade:
                logger.info(f"Document rejected (score: {doc.get('score', 0):.3f})")
            else:
                logger.warning(f"Ambiguous LLM response: '{grade}', keeping document")
                relevant_docs.append(doc)
                
        except Exception as e:
            logger.exception(f"LLM grading failed: {e}, keeping document")
            relevant_docs.append(doc)
    
    logger.info(f"Filtering complete: {len(relevant_docs)}/{len(docs)} passed")
    return relevant_docs

def main():
    parser = argparse.ArgumentParser(description="Fetch retrieval context from Qdrant for RAG evaluation.")
    parser.add_argument("--raw-file", default="raw_qa.txt", help="Path to raw QA text file")
    parser.add_argument("--output-file", default="retrieval_dataset.json", help="Path to output JSON file")
    parser.add_argument("--top-k", type=int, default=Config.TOP_K_DOCUMENTS, help="Number of documents to retrieve")
    parser.add_argument("--threshold", type=float, default=Config.SIMILARITY_THRESHOLD, help="Similarity threshold")
    parser.add_argument("--use-llm-filter", action="store_true", help="Enable LLM grading filter (default: False)")
    
    args = parser.parse_args()

    # 1. Load Raw QA
    raw_path = Path(args.raw_file)
    if not raw_path.exists():
        logger.error(f"File not found: {raw_path}")
        return
    
    raw_text = raw_path.read_text(encoding="utf-8")
    qa_pairs = parse_raw_qa(raw_text)
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {raw_path}")

    # 2. Init Retrieval Components
    logger.info("Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
    )

    logger.info("Connecting to Qdrant...")
    qdrant_client = QdrantClient(url=Config.QDRANT_URL)

    # Init LLM if filtering is enabled
    llm = None
    if args.use_llm_filter:
        logger.info("Initializing LLM for grading...")
        llm = ChatGoogleGenerativeAI(
            model=Config.GOOGLE_LLM_MODEL,
            temperature=0.2,
            google_api_key=Config.GOOGLE_API_KEY
        )

    output_data = []

    # 3. Retrieve
    for item in qa_pairs:
        question = item["question"]
        logger.info(f"Retrieving for Q{item['id']}: {question[:30]}...")
        
        # Embed query
        query_vector = embeddings.embed_query(question)
        
        # Search Qdrant
        search_results = qdrant_client.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=query_vector,
            limit=args.top_k,
            score_threshold=args.threshold,
            with_payload=True
        )
        
        if search_results:
            logger.info(f"Top hit score: {search_results[0].score}")
        else:
            logger.warning("No results found from Qdrant.")

        # Convert to list of dicts for processing
        hits = []
        for hit in search_results:
            hits.append({
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                # Add other metadata if needed
            })

        # Apply LLM Filter
        if llm and args.use_llm_filter:
            hits = filter_documents(llm, question, hits)

        # Extract contexts
        contexts = [h["content"] for h in hits if h["content"]]
        
        output_data.append({
            "question": question,
            "ground_truth": item["ground_truth"],
            "contexts": contexts
        })
    
    # 4. Save
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved retrieval dataset to {output_file}")

if __name__ == "__main__":
    main()
