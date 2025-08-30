#!/usr/bin/env python3
"""
Test embedding với 1 file để đảm bảo mọi thứ hoạt động
"""

import json
import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_file():
    """Test với 1 file JSON"""
    # Configuration
    json_file = Path("../../data/processed/chunks/MOET_SoTay_ThucHanh_CTXH_TrongTruongHoc_vi.json")
    qdrant_url = "http://localhost:6333"
    collection_name = "test_mental_health"
    
    try:
        logger.info(f"🧪 Testing with file: {json_file}")
        
        # Check file exists
        if not json_file.exists():
            logger.error(f"File not found: {json_file}")
            return False
        
        # Load JSON data
        logger.info("📂 Loading JSON data...")
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        logger.info(f"📊 Found {len(chunks_data)} chunks")
        
        # Take only first 3 chunks for testing
        test_chunks = chunks_data[:3]
        logger.info(f"🔬 Testing with {len(test_chunks)} chunks")
        
        # Create documents
        documents = []
        for i, chunk in enumerate(test_chunks):
            title = chunk.get('title', '').strip()
            context = chunk.get('context', '').strip()
            
            if not context:
                continue
            
            # Combine title and context
            if title:
                full_text = f"{title}\n\n{context}"
            else:
                full_text = context
            
            # Create metadata
            metadata = {
                "document": "test_document",
                "chunk_id": f"test_{i:03d}",
                "title": title[:100],  # Limit title length
                "source": "test"
            }
            
            doc = Document(page_content=full_text, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"📝 Created {len(documents)} documents")
        
        # Initialize embeddings
        logger.info("🤖 Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="dangvantuan/vietnamese-embedding",
            model_kwargs={"device": "cpu"}
        )
        logger.info("✅ Model loaded successfully")
        
        # Test embedding creation
        logger.info("🔢 Creating embeddings...")
        sample_text = documents[0].page_content[:200]
        logger.info(f"Sample text: {sample_text}...")
        
        sample_embedding = embeddings.embed_query(sample_text)
        logger.info(f"✅ Embedding created: dimension {len(sample_embedding)}")
        
        # Create Qdrant vector store
        logger.info(f"💾 Saving to Qdrant collection: {collection_name}")
        vectorstore = Qdrant.from_documents(
            documents,
            embeddings,
            url=qdrant_url,
            prefer_grpc=False,
            collection_name=collection_name
        )
        
        logger.info("✅ Successfully saved to Qdrant!")
        
        # Test search
        logger.info("🔍 Testing similarity search...")
        query = "tâm lý học sinh"
        results = vectorstore.similarity_search(query, k=2)
        
        logger.info(f"🎯 Search results for '{query}':")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result.page_content[:100]}...")
        
        logger.info("🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_single_file()
    if success:
        print("\n✅ Test passed! Ready to run full embedding script.")
    else:
        print("\n❌ Test failed. Please check the errors above.")
