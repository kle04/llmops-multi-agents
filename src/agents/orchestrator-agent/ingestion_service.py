import sys
import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import specific components from local utils
try:
    from utils.pdf_processor import PDFProcessor
    from utils.markdown_processor import MarkdownProcessor
    from utils.embedding_manager import EmbeddingManager
    from utils.qdrant_manager import QdrantManager
    from config import Config as DataConfig
except ImportError as e:
    logger.error(f"Failed to import ingestion utilities: {e}")
    logger.error("Ensure local utils directory exists and dependencies are installed.")
    raise

class IngestionService:
    """
    Service to handle document ingestion for the Orchestrator Agent.
    This is a specific implementation for the agent that reuses the core logic 
    from data-preparing/utils but focuses on single-file API processing.
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Ingestion Service...")
        try:
            # Initialize managers
            logger.info("   [1/4] Initializing Embedding Manager (this may download models)...")
            self.embedding_manager = EmbeddingManager()
            logger.info("   ✅ Embedding Manager Initialized")
            
            logger.info("   [2/4] Initializing Qdrant Manager...")
            self.qdrant_manager = QdrantManager()
            logger.info("   ✅ Qdrant Manager Initialized")
            
            # Initialize processors
            logger.info("   [3/4] Initializing PDF Processor...")
            self.pdf_processor = PDFProcessor(embedding_manager=self.embedding_manager)
            
            logger.info("   [4/4] Initializing Markdown Processor...")
            self.markdown_processor = MarkdownProcessor(embedding_manager=self.embedding_manager)
            
            # Check connection immediately
            logger.info("   Checking health status...")
            self.health_check()
            logger.info("✅ Ingestion Service Ready")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ingestion Service: {e}", exc_info=True)
            raise

    def health_check(self) -> dict:
        """Check if dependent services (Qdrant, Models) are healthy."""
        status = {"status": "healthy", "qdrant": "unknown", "models": "unknown"}
        
        # Check Qdrant
        try:
            q_health = self.qdrant_manager.health_check()
            status["qdrant"] = "connected" if q_health.get("status") == "healthy" else "disconnected"
        except Exception as e:
            status["qdrant"] = f"error: {str(e)}"
            status["status"] = "unhealthy"
            
        # Check Embedding Model
        try:
            # Simple check if model is loaded
            if self.embedding_manager.model:
                 status["models"] = "loaded"
            else:
                 status["models"] = "not_loaded"
                 status["status"] = "unhealthy"
        except Exception as e:
            status["models"] = f"error: {str(e)}"
            status["status"] = "unhealthy"
            
        return status

    def ingest_file(self, file_path: str, original_filename: str) -> bool:
        """
        Process and ingest a single file.
        
        Args:
            file_path: Path to the temporary file on disk.
            original_filename: Original name of the uploaded file (detects extension).
            
        Returns:
            bool: True if success, False otherwise.
        """
        logger.info(f"🔄 Processing file: {original_filename}")
        
        try:
            ext = Path(original_filename).suffix.lower()
            documents = []
            
            # 1. Process File
            if ext == '.pdf':
                # PDFProcessor expects a path
                documents = self.pdf_processor.process_pdf(file_path)
            elif ext == '.md':
                documents = self.markdown_processor.process_markdown(file_path)
            elif ext == '.txt':
                # Basic TXT handling (reuse markdown processor or simple split)
                # For now, let's treat as MD without sections
                documents = self.markdown_processor.process_markdown(file_path)
            else:
                logger.error(f"Unsupported file type: {ext}")
                return False
                
            if not documents:
                logger.warning(f"⚠️  No content extracted from {original_filename}")
                return False
                
            # FIX: Post-process documents to restore original filename in metadata
            # The processors use the temp file path (e.g. tmp123.pdf) for 'source' and 'doc_id'
            temp_filename = Path(file_path).name
            
            for doc in documents:
                # Fix source
                if doc.get("source") == temp_filename:
                    doc["source"] = original_filename
                
                # Fix doc_id if it starts with temp filename
                # e.g. tmp123.md_SECTION_... -> original.md_SECTION_...
                if "doc_id" in doc and doc.get("doc_id", "").startswith(temp_filename):
                     doc["doc_id"] = doc["doc_id"].replace(temp_filename, original_filename, 1)

            logger.info(f"   Generated {len(documents)} chunks from {original_filename}")
            
            # 2. Create Embeddings
            # embed_documents expects list of dicts with 'content' key
            documents_with_embeddings = self.embedding_manager.embed_documents(documents)
            
            if not documents_with_embeddings:
                logger.error("❌ Failed to generate embeddings")
                return False
                
            # 3. Store in Qdrant
            # We don't verify collection existence here as QdrantManager handles it?
            # Actually QdrantManager.add_documents checks existence.
            # But we should ensure collection exists initially.
            # Lazy creation:
            try:
                # Check if collection exists
                self.qdrant_manager.client.get_collection(DataConfig.COLLECTION_NAME)
            except:
                logger.info(f"Collection {DataConfig.COLLECTION_NAME} not found, creating...")
                vector_size = self.embedding_manager.embedding_dimension
                self.qdrant_manager.create_collection(vector_size)
            
            success = self.qdrant_manager.add_documents(documents_with_embeddings)
            
            if success:
                logger.info(f"✅ Successfully ingested {original_filename}")
            else:
                logger.error(f"❌ Failed to store documents in Qdrant")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Error during ingestion of {original_filename}: {e}", exc_info=True)
            return False
