#!/usr/bin/env python3
"""
Script embedding cơ bản với LangChain để tạo embeddings và lưu vào Qdrant
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEmbeddingProcessor:
    """Processor đơn giản để tạo embeddings và lưu vào Qdrant"""
    
    def __init__(self, model_name: str = "dangvantuan/vietnamese-embedding", 
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "mental_health_vi"):
        self.model_name = model_name
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Setup collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup hoặc tạo collection trong Qdrant"""
        try:
            # Kiểm tra collection có tồn tại không
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                # Tạo collection mới với 768 dimensions (cho vietnamese-embedding)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
            # Kiểm tra thông tin collection
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection info: {collection_info.points_count} points")
            
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def load_chunks_from_json(self, json_file: Path) -> List[Document]:
        """Load chunks từ file JSON và convert thành LangChain Documents"""
        logger.info(f"Loading chunks from: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        documents = []
        doc_name = json_file.stem  # Tên file làm document name
        
        for i, chunk in enumerate(chunks_data):
            # Tạo Document với title và context
            title = chunk.get('title', '').strip()
            context = chunk.get('context', '').strip()
            
            if not context:
                logger.warning(f"Skipping chunk {i} - empty context")
                continue
            
            # Kết hợp title và context
            if title:
                full_text = f"{title}\n\n{context}"
            else:
                full_text = context
            
            # Tạo metadata
            metadata = {
                "document": doc_name,
                "chunk_id": f"{doc_name}_{i:05d}",
                "title": title,
                "source": "json_chunks"
            }
            
            doc = Document(
                page_content=full_text,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {json_file}")
        return documents
    
    def process_single_file(self, json_file: Path) -> int:
        """Xử lý một file JSON và lưu vào Qdrant"""
        try:
            # Load documents
            documents = self.load_chunks_from_json(json_file)
            
            if not documents:
                logger.warning(f"No documents found in {json_file}")
                return 0
            
            # Process in smaller batches to avoid "index out of range" error
            batch_size = 50  # Smaller batch size
            total_processed = 0
            
            logger.info(f"Creating embeddings for {len(documents)} documents in batches of {batch_size}...")
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}: {len(batch)} documents")
                
                try:
                    # Create embeddings for this batch
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    
                    # Generate embeddings
                    embeddings_vectors = self.embeddings.embed_documents(texts)
                    
                    # Create points for Qdrant
                    from qdrant_client.models import PointStruct
                    import uuid
                    
                    points = []
                    for j, (text, metadata, vector) in enumerate(zip(texts, metadatas, embeddings_vectors)):
                        point_id = str(uuid.uuid4())
                        
                        # Clean metadata for Qdrant (ensure all values are serializable)
                        clean_metadata = {
                            "text": text[:1000],  # Limit text length in metadata
                            "document": str(metadata.get("document", "")),
                            "chunk_id": str(metadata.get("chunk_id", "")),
                            "title": str(metadata.get("title", ""))[:200],  # Limit title length
                            "source": str(metadata.get("source", ""))
                        }
                        
                        point = PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=clean_metadata
                        )
                        points.append(point)
                    
                    # Upload to Qdrant
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    total_processed += len(batch)
                    logger.info(f"✅ Batch {i//batch_size + 1} completed: {len(batch)} chunks")
                    
                except Exception as batch_error:
                    logger.error(f"❌ Error processing batch {i//batch_size + 1}: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
            
            logger.info(f"✅ Successfully processed {json_file.name}: {total_processed}/{len(documents)} chunks")
            return total_processed
            
        except Exception as e:
            logger.error(f"❌ Error processing {json_file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0
    
    def process_all_files(self, chunks_dir: Path) -> Dict[str, Any]:
        """Xử lý tất cả file JSON trong thư mục chunks"""
        json_files = list(chunks_dir.glob("*.json"))
        
        if not json_files:
            logger.error(f"No JSON files found in {chunks_dir}")
            return {"status": "error", "message": "No files found"}
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        total_chunks = 0
        processed_files = 0
        failed_files = []
        
        for json_file in tqdm(json_files, desc="Processing files"):
            logger.info(f"\n📄 Processing: {json_file.name}")
            
            chunks_processed = self.process_single_file(json_file)
            
            if chunks_processed > 0:
                total_chunks += chunks_processed
                processed_files += 1
            else:
                failed_files.append(json_file.name)
        
        # Lấy thông tin cuối cùng từ collection
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            final_points = collection_info.points_count
        except:
            final_points = "unknown"
        
        summary = {
            "status": "completed",
            "total_files": len(json_files),
            "processed_files": processed_files,
            "failed_files": len(failed_files),
            "total_chunks_processed": total_chunks,
            "final_points_in_qdrant": final_points,
            "failed_file_names": failed_files
        }
        
        logger.info(f"\n🎉 SUMMARY:")
        logger.info(f"   Total files: {summary['total_files']}")
        logger.info(f"   Processed: {summary['processed_files']}")
        logger.info(f"   Failed: {summary['failed_files']}")
        logger.info(f"   Total chunks: {summary['total_chunks_processed']}")
        logger.info(f"   Points in Qdrant: {summary['final_points_in_qdrant']}")
        
        if failed_files:
            logger.warning(f"   Failed files: {failed_files}")
        
        return summary

def main():
    """Main function"""
    # Configuration
    chunks_dir = Path("../../data/processed/chunks")
    model_name = "dangvantuan/vietnamese-embedding"
    qdrant_url = "http://localhost:6333"
    collection_name = "mental_health_vi"
    
    try:
        # Kiểm tra thư mục chunks
        if not chunks_dir.exists():
            logger.error(f"Chunks directory not found: {chunks_dir}")
            return 1
        
        logger.info(f"🚀 Starting embedding process...")
        logger.info(f"   Chunks directory: {chunks_dir}")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Qdrant URL: {qdrant_url}")
        logger.info(f"   Collection: {collection_name}")
        
        # Initialize processor
        processor = SimpleEmbeddingProcessor(
            model_name=model_name,
            qdrant_url=qdrant_url,
            collection_name=collection_name
        )
        
        # Process all files
        summary = processor.process_all_files(chunks_dir)
        
        if summary["status"] == "completed":
            logger.info(f"✅ Embedding process completed successfully!")
            return 0
        else:
            logger.error(f"❌ Embedding process failed: {summary.get('message', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())