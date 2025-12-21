#!/usr/bin/env python3
"""
PDF Processor for Mental Health RAG Agent.
Handles PDF extraction, cleaning, and Semantic Chunking for better retrieval.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import PyPDF2
import numpy as np
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Splits text based on semantic similarity between sentences.
    """
    def __init__(self, embedding_manager, buffer_size: int = 1, breakpoint_percentile_threshold: int = 95):
        self.embedding_manager = embedding_manager
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter for Vietnamese/English
        # Handles commonly used endings.
        text = re.sub(r'([.?!])\s+', r'\1\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences

    def _combine_sentences(self, sentences: List[str]) -> List[Dict[str, str]]:
        # Create a sliding buffer window (sentences with context)
        # Actually, let's keep it simple: compare sentence i with sentence i+1
        # To reduce noise, we can embed 'combined_sentence' which is context window.
        
        sentences_with_context = []
        for i in range(len(sentences)):
            combined = ""
            # Context before
            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined += sentences[j] + " "
            combined += sentences[i]
            # Context after
            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined += " " + sentences[j]
            
            sentences_with_context.append({
                "sentence": sentences[i],
                "combined": combined.strip(),
                "index": i
            })
        return sentences_with_context

    def split_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # 1. Prepare sentences with context
        sentences_with_context = self._combine_sentences(sentences)
        
        # 2. Embed
        combined_texts = [s["combined"] for s in sentences_with_context]
        # Use embedding manager (assumes it has a batch encode method or similar)
        # using the public model attribute if available, or a method
        if hasattr(self.embedding_manager, 'model'):
             # Embed in batches to prevent hanging and provide feedback
             embeddings = []
             batch_size = 16 # Safe batch size for large models
             total_batches = (len(combined_texts) + batch_size - 1) // batch_size
             
             logger.info(f"Embedding {len(combined_texts)} sentences for semantic splitting (in {total_batches} batches)...")
             
             for i in range(0, len(combined_texts), batch_size):
                 batch_texts = combined_texts[i:i+batch_size]
                 if i % (batch_size * 5) == 0: # Log every 5 batches to avoid spam but show life
                     logger.info(f"  - Semantic Embed Batch {i//batch_size + 1}/{total_batches}")
                 
                 try:
                     batch_embeddings = self.embedding_manager.model.encode(
                         batch_texts, 
                         convert_to_numpy=True,
                         show_progress_bar=False,
                         batch_size=batch_size
                     )
                     if len(batch_embeddings.shape) == 1:
                         batch_embeddings = batch_embeddings.reshape(1, -1)
                     embeddings.extend(batch_embeddings)
                 except Exception as e:
                     logger.error(f"Error embedding semantic batch {i}: {e}")
                     # Fallback: create zero embeddings for this batch to keep index alignment?
                     # Or just panic. Let's append zeros to maintain alignment which is critical for splitting
                     # Assuming dimension is available or can be inferred
                     dim = 1024 # Default fallback or try to get from model
                     try:
                         dim = self.embedding_manager.embedding_dimension
                     except:
                         pass
                     embeddings.extend(np.zeros((len(batch_texts), dim)))
             
             embeddings = np.array(embeddings)
        else:
             print("Warning: Embedding manager does not expose model directly.")
             return [text]

        # 3. Calculate Cosine Distances
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            dist = 1 - sim
            distances.append(dist)
        
        # 4. Calculate Threshold
        breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
        
        # 5. Split
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, dist in enumerate(distances):
            if dist > breakpoint_distance_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i+1]]
            else:
                current_chunk.append(sentences[i+1])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

class PDFProcessor:
    def __init__(self, embedding_manager=None):
        """
        Initialize PDF processor.
        :param embedding_manager: Optional, used for Semantic Chunking.
        """
        self.embedding_manager = embedding_manager
        if self.embedding_manager:
            logger.info("Semantic Chunking ENABLED")
            self.semantic_chunker = SemanticChunker(embedding_manager)
        else:
            logger.info("Semantic Chunking DISABLED (using default RecursiveCharacter)")
            # Fallback import
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with robust cleaning."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"Reading PDF: {pdf_path} ({len(pdf_reader.pages)} pages)")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        page_text = self.clean_text(page_text)
                        # Add page marker for metadata (removed before embedding)
                        text += f"\n[PAGE_{page_num + 1}]\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error reading page {page_num+1}: {e}")
                        continue
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF {pdf_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text."""
        if not text:
            return ""
        
        # Remove control chars
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix Vietnamese punctuation spacing
        for p in ['.', ',', ';', ':', '?', '!']:
            text = text.replace(f' {p}', p).replace(f'{p}', f'{p} ')
            
        text = re.sub(r'\s+', ' ', text) # cleanup again
        return text.strip()

    def create_chunks(self, text: str, source_file: str) -> List[Dict]:
        """Split text into chunks using Semantic Chunking if available."""
        if not text.strip():
            return []

        doc_id_base = str(uuid.uuid4())
        chunks_data = []

        # 1. Remove Page Markers for splitting (or keep them?) 
        # For semantic splitting, page markers might add noise. Let's strip them but keep track? 
        # Complicated. Let's just strip markers for the content.
        clean_content = re.sub(r'\[PAGE_\d+\]', '', text)
        
        chunks = []
        if self.embedding_manager:
            chunks = self.semantic_chunker.split_text(clean_content)
        else:
            chunks = self.text_splitter.split_text(clean_content)

        # 2. Create Chunk Dicts
        for i, chunk_content in enumerate(chunks):
             # Ensure chunk isn't too small
            if len(chunk_content) < 50:
                continue
                
            chunk_dict = {
                "content": chunk_content.strip(),
                "source": source_file,
                "chunk_index": i,
                "doc_id": doc_id_base,
                "section": "General" # TODO: Implement better section extraction
            }
            chunks_data.append(chunk_dict)
            
        return chunks_data

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a single PDF file."""
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []
            
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
            
        chunks = self.create_chunks(text, Path(pdf_path).name)
        logger.info(f"Processed {Path(pdf_path).name}: {len(chunks)} chunks generated.")
        return chunks
