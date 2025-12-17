#!/usr/bin/env python3
"""
Markdown Processor for Mental Health RAG Agent.
Handles markdown extraction, cleaning, and semantic chunking for better retrieval.
"""

import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Splits text based on semantic similarity between sentences.
    Reused from PDFProcessor for consistency.
    """
    def __init__(self, embedding_manager, buffer_size: int = 1, breakpoint_percentile_threshold: int = 95):
        self.embedding_manager = embedding_manager
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for Vietnamese/English."""
        text = re.sub(r'([.?!])\s+', r'\1\n', text)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return sentences

    def _combine_sentences(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Create sentences with context window for better semantic understanding."""
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
        """Split text using semantic similarity."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        sentences_with_context = self._combine_sentences(sentences)
        
        if not hasattr(self.embedding_manager, 'model'):
            logger.warning("Embedding manager does not expose model directly.")
            return [text]

        combined_texts = [s["combined"] for s in sentences_with_context]
        embeddings = []
        batch_size = 16
        total_batches = (len(combined_texts) + batch_size - 1) // batch_size
        
        logger.info(f"Embedding {len(combined_texts)} sentences for semantic splitting (in {total_batches} batches)...")
        
        for i in range(0, len(combined_texts), batch_size):
            batch_texts = combined_texts[i:i+batch_size]
            if i % (batch_size * 5) == 0:
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
                dim = 1024
                try:
                    dim = self.embedding_manager.embedding_dimension
                except:
                    pass
                embeddings.extend(np.zeros((len(batch_texts), dim)))
        
        embeddings = np.array(embeddings)

        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            dist = 1 - sim
            distances.append(dist)
        
        breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
        
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


class MarkdownProcessor:
    """
    Enhanced Markdown Processor with semantic chunking support.
    """
    
    def __init__(self, embedding_manager=None):
        """
        Initialize Markdown processor.
        :param embedding_manager: Optional, used for Semantic Chunking.
        """
        self.embedding_manager = embedding_manager
        if self.embedding_manager:
            logger.info("Semantic Chunking ENABLED for markdown")
            self.semantic_chunker = SemanticChunker(embedding_manager)
        else:
            logger.info("Semantic Chunking DISABLED (using default RecursiveCharacter)")
        
        # Initialize text splitter with Config values
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text."""
        if not text:
            return ""
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace but preserve markdown structure
        # First, preserve list markers and headers
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces/tabs
        
        # Fix Vietnamese punctuation spacing
        for p in ['.', ',', ';', ':', '?', '!']:
            text = text.replace(f' {p}', p).replace(f'{p}', f'{p} ')
        
        text = re.sub(r'\s+', ' ', text)  # Final cleanup
        return text.strip()

    def _extract_header_hierarchy(self, metadata: Dict) -> Dict:
        """Extract and organize header hierarchy from metadata."""
        headers = {
            "h1": metadata.get("header_1", "").strip(),
            "h2": metadata.get("header_2", "").strip(),
            "h3": metadata.get("header_3", "").strip(),
            "h4": metadata.get("header_4", "").strip(),
        }
        
        # Build section path
        section_parts = [h for h in [headers["h1"], headers["h2"], headers["h3"], headers["h4"]] if h]
        section_path = " > ".join(section_parts) if section_parts else "Introduction"
        
        return {
            "headers": headers,
            "section_path": section_path,
            "section_level": len(section_parts)
        }

    def _create_chunk_metadata(self, content: str, file_name: str, 
                               section_info: Dict, chunk_index: int, 
                               global_chunk_index: int) -> Dict:
        """Create comprehensive metadata for a chunk."""
        # Generate unique doc_id
        section_slug = re.sub(r'[^\w\s-]', '', section_info["section_path"])
        section_slug = re.sub(r'[-\s]+', '_', section_slug)[:50]  # Limit length
        doc_id = f"{file_name}_{section_slug}_{global_chunk_index}"
        
        return {
            "content": content,
            "source": file_name,
            "chunk_index": chunk_index,  # Index within section
            "global_chunk_index": global_chunk_index,  # Global index across all chunks
            "section": section_info["section_path"],
            "section_level": section_info["section_level"],
            "header_1": section_info["headers"]["h1"],
            "header_2": section_info["headers"]["h2"],
            "header_3": section_info["headers"]["h3"],
            "doc_id": doc_id
        }

    def process_markdown(self, file_path: str) -> List[Dict]:
        """
        Process a markdown file and return a list of chunks with enhanced metadata.
        """
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"Empty file: {file_path}")
                return []

            # Only do minimal cleaning before header splitting (preserve markdown structure)
            # Remove control characters but keep newlines and markdown structure
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
            text = re.sub(r'\r', '\n', text)  # Handle old Mac line endings

            # Split by headers - support up to h4
            # Must do this BEFORE aggressive cleaning to preserve markdown structure
            headers_to_split_on = [
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
                ("####", "header_4"),
            ]
            
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(text)

            if not md_header_splits:
                logger.warning(f"No header splits found in {file_path}")
                return []

            final_documents = []
            file_name = Path(file_path).name
            global_chunk_index = 0

            for section_chunk in md_header_splits:
                # Extract header hierarchy
                section_info = self._extract_header_hierarchy(section_chunk.metadata)
                
                # Get content and clean it (now safe to clean since headers are already extracted)
                content = section_chunk.page_content.strip()
                if not content or len(content) < 20:  # Skip very short sections
                    continue
                
                # Clean the content now (after header extraction)
                content = self.clean_text(content)
                if not content or len(content) < 20:
                    continue
                
                # Determine chunking strategy
                chunks = []
                if self.embedding_manager and len(content) > Config.CHUNK_SIZE:
                    # Use semantic chunking for longer sections
                    try:
                        chunks = self.semantic_chunker.split_text(content)
                        logger.debug(f"Semantic chunking: {len(chunks)} chunks for section {section_info['section_path']}")
                    except Exception as e:
                        logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
                        chunks = self.text_splitter.split_text(content)
                else:
                    # Use recursive chunking
                    chunks = self.text_splitter.split_text(content)
                
                # If no chunks created, use the whole content
                if not chunks:
                    chunks = [content] if content else []
                
                # Create documents for each chunk
                for i, chunk_content in enumerate(chunks):
                    # Skip very small chunks
                    if len(chunk_content.strip()) < 50:
                        continue
                    
                    # Create metadata
                    doc = self._create_chunk_metadata(
                        content=chunk_content.strip(),
                        file_name=file_name,
                        section_info=section_info,
                        chunk_index=i,
                        global_chunk_index=global_chunk_index
                    )
                    
                    final_documents.append(doc)
                    global_chunk_index += 1

            logger.info(f"Processed {file_name}: {len(final_documents)} chunks generated from {len(md_header_splits)} sections")
            return final_documents

        except Exception as e:
            logger.error(f"Error processing markdown file {file_path}: {e}", exc_info=True)
            return []
