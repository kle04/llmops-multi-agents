
import os
import glob
import uuid
from pathlib import Path
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from config import Config
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize Vietnamese text."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    for p in ['.', ',', ';', ':', '?', '!']:
        text = text.replace(f' {p}', p).replace(f'{p}', f'{p} ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += clean_text(page_text) + "\n"
        return text
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        return ""

def main():
    # 0. Setup Paths & Config
    if os.path.exists("data"):
        data_dir = "data"
    else:
        data_dir = "src/data-preparing/data"
    
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {data_dir}")
        return

    # User Configuration
    CHUNK_SIZE = 1000
    # Embedding Model (from Config)
    EMBEDDING_MODEL_NAME = Config.EMBEDDING_MODEL
    QDRANT_URL = Config.QDRANT_URL
    QDRANT_API_KEY = Config.QDRANT_API_KEY
    COLLECTION_NAME = Config.COLLECTION_NAME
    
    logger.info(f"Configuration:")
    logger.info(f" - Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f" - Embedding Model: {EMBEDDING_MODEL_NAME}")
    logger.info(f" - Qdrant URL: {QDRANT_URL}")
    logger.info(f" - Collection: {COLLECTION_NAME}")

    # 1. Initialize Clients
    logger.info("Initializing components...")
    
    # Embedding Model
    # "trust_remote_code=True" is often needed for newer huggingface models
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    embedding_size = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding initialized. Dimension: {embedding_size}")

    # Qdrant Client (Using prefer_grpc=False as requested/researched to avoid Ingress issues)
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        prefer_grpc=False 
    )
    
    # Recreate Collection
    logger.info(f"Recreating collection '{COLLECTION_NAME}'...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
    )

    # 2. Process Files
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_points = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        full_text = extract_text_from_pdf(pdf_path)
        if not full_text:
            continue
            
        chunks = text_splitter.split_text(full_text)
        filename = Path(pdf_path).name
        
        # Batch embedding for efficiency
        if not chunks:
            continue
            
        logger.info(f"File {filename}: Generating embeddings for {len(chunks)} chunks...")
        
        # Embed in smaller batches to avoid hanging with large models
        embedding_batch_size = 16 # Small batch size for large models (0.6B params)
        embeddings = []
        
        for i in range(0, len(chunks), embedding_batch_size):
            batch_chunks = chunks[i:i + embedding_batch_size]
            logger.info(f"  - Embedding batch {i//embedding_batch_size + 1}/{(len(chunks)-1)//embedding_batch_size + 1}")
            try:
                batch_embeddings = model.encode(batch_chunks, show_progress_bar=False, normalize_embeddings=Config.NORMALIZE_EMBEDDINGS)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i}: {e}")
                # Fallback: try one by one or skip?
                # For now, just skip this batch to avoid crashing everything
                continue
                
        if len(embeddings) != len(chunks):
            logger.warning(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings. Some chunks may have failed.")
        
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            
            point_id = str(uuid.uuid4())
            
            payload = {
                "source": filename,
                "chunk_index": i,
                "text": chunk_text, # Important: Qdrant payload stores the text for retrieval
                "content": chunk_text # Duplicate key for compatibility with existing scripts if any
            }
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=payload
            )
            all_points.append(point)

    # 3. Upload to Qdrant
    if all_points:
        logger.info(f"Uploading {len(all_points)} points to Qdrant...")
        # Upload in batches to avoid payload too large errors
        batch_size = 50 
        for i in tqdm(range(0, len(all_points), batch_size), desc="Uploading Batches"):
            batch = all_points[i:i+batch_size]
            try:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
            except Exception as e:
                logger.error(f"Failed to upload batch {i}: {e}")
        
        logger.info("✅ Data ingestion complete!")
        
        # Verify count
        count = qdrant_client.count(collection_name=COLLECTION_NAME)
        logger.info(f"Total vectors in collection: {count.count}")
        
    else:
        logger.warning("No data generated to upload.")

if __name__ == "__main__":
    main()
