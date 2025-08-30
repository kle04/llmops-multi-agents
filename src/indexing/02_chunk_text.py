#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class ChunkingError(Exception):
    """Raised when chunking fails"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

# ---------- Token counter ----------
class TokenCounter:
    """Thread-safe token counter with caching"""
    
    def __init__(self):
        self._encoder = None
        self._cache = {}
    
    def _get_encoder(self):
        """Lazy load tiktoken encoder"""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not available, using fallback estimation")
                self._encoder = "fallback"
        return self._encoder
    
    @lru_cache(maxsize=1024)
    def count_tokens(self, text: str) -> int:
        """Count tokens with caching for performance"""
        if not text:
            return 0
            
        encoder = self._get_encoder()
        if encoder == "fallback":
            # Fallback: estimate 4 chars ~ 1 token
            return max(1, math.ceil(len(text) / 4))
        else:
            return len(encoder.encode(text))

# Global token counter instance
token_counter = TokenCounter()

# Pre-compile regex patterns for Vietnamese educational documents
PATTERNS = {
    'toc_dots': re.compile(r"\.{2,}\s*\d{1,4}$"),
    'heading_part': re.compile(r"^(PHẦN|CHƯƠNG|MỤC)\s+\d+[\.\:\s]", re.IGNORECASE),
    'heading_numbered': re.compile(r"^\d+(\.\d+)*\.?\s+"),
    'heading_intro': re.compile(r"^(LỜI MỞ ĐẦU|MỤC LỤC|GIỚI THIỆU|GIẢI THÍCH THUẬT NGỮ|TÀI LIỆU THAM KHẢO|PHỤ LỤC)", re.IGNORECASE),
    'whitespace': re.compile(r'\s+'),
    'page_number': re.compile(r"^\d{1,4}\s*$"),
    'danh_muc': re.compile(r"^(DANH MỤC|BẢNG BIỂU)", re.IGNORECASE),
}

# ---------- Helpers ----------
def parse_skip_pages(cell: str) -> Set[int]:
    """
    Parse skip pages string like '1-3,5,8-9' -> set({1,2,3,5,8,9})
    """
    if not isinstance(cell, str) or not cell.strip():
        return set()
    
    pages = set()
    try:
        for part in cell.split(","):
            part = part.strip()
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start, end = int(start_str), int(end_str)
                pages.update(range(min(start, end), max(start, end) + 1))
            else:
                pages.add(int(part))
    except ValueError as e:
        raise ValueError(f"Invalid skip_pages format '{cell}': {e}")
    
    return pages

def is_toc_page(text: str) -> bool:
    """
    Detect table of contents pages using heuristics for Vietnamese documents.
    """
    if not text or not text.strip():
        return False
        
    # Check for explicit TOC keywords
    upper_text = text.upper()
    if any(keyword in upper_text for keyword in ["MỤC LỤC", "MUC LUC", "DANH MỤC"]):
        return True
    
    # Check for TOC-like patterns (many lines ending with page numbers)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
        
    matched_lines = sum(1 for line in lines if PATTERNS['toc_dots'].search(line))
    threshold = max(3, len(lines) // 3)
    
    return matched_lines >= threshold

def is_heading(line: str) -> bool:
    """
    Detect if a line is likely a heading/title for Vietnamese educational documents.
    """
    line = line.strip()
    if not line:
        return False
    
    # Check common heading patterns for Vietnamese educational documents
    if PATTERNS['heading_part'].match(line):
        return True
    if PATTERNS['heading_numbered'].match(line):
        return True
    if PATTERNS['heading_intro'].match(line):
        return True
    if PATTERNS['danh_muc'].match(line):
        return True
    
    # Check for ALL CAPS short lines (likely headings)
    if (len(line) <= 70 and 
        line == line.upper() and 
        any(ch.isalpha() for ch in line)):
        return True
    
    return False

def extract_section_title(paragraph: str) -> str:
    """
    Extract section title from paragraph for Vietnamese educational documents.
    """
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return ""
    
    first_line = lines[0]
    
    # Check for different heading patterns
    if PATTERNS['heading_part'].match(first_line):
        return first_line
    elif PATTERNS['heading_numbered'].match(first_line):
        return first_line
    elif PATTERNS['heading_intro'].match(first_line):
        return first_line
    elif PATTERNS['danh_muc'].match(first_line):
        return first_line
    elif is_heading(first_line):
        return first_line
    
    return ""

def extract_page_info(section: str, text: str) -> Optional[str]:
    """
    Trích xuất thông tin trang từ section hoặc text
    
    Args:
        section: Section info từ chunk
        text: Nội dung text
        
    Returns:
        String mô tả trang hoặc None
    """
    # Pattern tìm số trang
    page_patterns = [
        r'(?:^|\s)(\d+)(?:\s|$)',  # Số đơn lẻ
        r'(?:^|\s)(\d+)-(\d+)(?:\s|$)',  # Phạm vi trang
        r'(?:^|\s)(\d+),\s*(\d+)(?:\s|$)',  # Nhiều trang
    ]
    
    pages_found = set()
    
    # Tìm trong section
    if section and isinstance(section, str):
        for pattern in page_patterns:
            matches = re.findall(pattern, section)
            for match in matches:
                if isinstance(match, tuple):
                    pages_found.update(match)
                else:
                    pages_found.add(match)
    
    # Tìm trong text (chỉ ở đầu text)
    text_start = text[:200] if text else ""
    for pattern in page_patterns:
        matches = re.findall(pattern, text_start)
        for match in matches:
            if isinstance(match, tuple):
                pages_found.update(match)
            else:
                pages_found.add(match)
            break  # Chỉ lấy match đầu tiên trong text
    
    if pages_found:
        # Loại bỏ số quá lớn (không phải trang)
        valid_pages = [int(p) for p in pages_found if p.isdigit() and 1 <= int(p) <= 1000]
        if valid_pages:
            valid_pages.sort()
            if len(valid_pages) == 1:
                return f"tr.{valid_pages[0]}"
            elif len(valid_pages) == 2:
                return f"tr.{valid_pages[0]}-{valid_pages[1]}"
            else:
                return f"tr.{valid_pages[0]}-{valid_pages[-1]}"
    
    return None

def paragraphs_from_pages(pages: List[str]) -> List[str]:
    """
    Convert pages to paragraphs with better handling for Vietnamese text.
    """
    if not pages:
        return []
        
    # Join all pages with double newlines
    text = "\n\n".join(page for page in pages if page.strip())
    
    # Split into paragraphs and filter empty ones
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Filter out page numbers and very short meaningless paragraphs
    filtered_paragraphs = []
    for p in paragraphs:
        if len(p.strip()) < 10:  # Skip very short paragraphs
            continue
        if PATTERNS['page_number'].match(p.strip()):  # Skip standalone page numbers
            continue
        filtered_paragraphs.append(p)
    
    return filtered_paragraphs

def chunk_with_overlap(paras: List[str], max_tokens: int, overlap: int, min_tokens: int) -> List[str]:
    """
    Create chunks from paragraphs with overlap and token limits, optimized for Vietnamese educational content.
    """
    if max_tokens <= 0 or min_tokens <= 0 or overlap < 0:
        raise ChunkingError(f"Invalid chunking parameters: max_tokens={max_tokens}, min_tokens={min_tokens}, overlap={overlap}")
    
    if overlap >= max_tokens:
        raise ChunkingError(f"Overlap ({overlap}) must be less than max_tokens ({max_tokens})")
    
    chunks = []
    current_buffer = []
    current_tokens = 0
    
    for paragraph in paras:
        if not paragraph.strip():
            continue
            
        para_tokens = token_counter.count_tokens(paragraph)
        
        # Handle extremely long paragraphs by splitting them at sentence boundaries
        if para_tokens > max_tokens * 1.5:
            logger.warning(f"Very long paragraph ({para_tokens} tokens), splitting by sentences")
            # Split by Vietnamese sentence endings
            sentences = re.split(r'[.!?](?:\s|$)', paragraph)
            sentence_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                test_chunk = sentence_chunk + ". " + sentence if sentence_chunk else sentence
                if token_counter.count_tokens(test_chunk) > max_tokens:
                    if sentence_chunk and token_counter.count_tokens(sentence_chunk) >= min_tokens:
                        chunks.append(sentence_chunk)
                    sentence_chunk = sentence
                else:
                    sentence_chunk = test_chunk
            
            if sentence_chunk and token_counter.count_tokens(sentence_chunk) >= min_tokens:
                chunks.append(sentence_chunk)
            continue
        
        # Check if we can add this paragraph to current chunk
        if current_tokens + para_tokens <= max_tokens:
            current_buffer.append(paragraph)
            current_tokens += para_tokens
        else:
            # Finalize current chunk if it meets minimum requirements
            if current_tokens >= min_tokens and current_buffer:
                chunks.append("\n\n".join(current_buffer))
                
                # Create overlap for next chunk
                if overlap > 0 and current_buffer:
                    overlap_buffer = []
                    overlap_tokens = 0
                    
                    # Take paragraphs from the end for overlap
                    for para in reversed(current_buffer):
                        para_tokens = token_counter.count_tokens(para)
                        if overlap_tokens + para_tokens > overlap:
                            break
                        overlap_buffer.append(para)
                        overlap_tokens += para_tokens
                    
                    current_buffer = list(reversed(overlap_buffer))
                    current_tokens = overlap_tokens
                else:
                    current_buffer = []
                    current_tokens = 0
            else:
                # Current buffer doesn't meet minimum, start fresh
                current_buffer = []
                current_tokens = 0
            
            # Add current paragraph to buffer
            current_buffer.append(paragraph)
            current_tokens += para_tokens
    
    # Add final chunk if it exists and meets requirements
    if current_buffer and current_tokens >= min_tokens:
        chunks.append("\n\n".join(current_buffer))
    
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]

def attach_sections(paras: List[str]) -> List[Tuple[str, str]]:
    """
    Attach section headings to paragraphs for Vietnamese educational documents.
    """
    if not paras:
        return []
        
    result = []
    current_section = ""
    
    for paragraph in paras:
        if not paragraph.strip():
            continue
            
        # Extract section title from this paragraph
        section_title = extract_section_title(paragraph)
        
        if section_title:
            current_section = section_title
        
        result.append((current_section, paragraph))
    
    return result

def validate_configuration(catalog_path: Path, text_dir: Path, max_tokens: int, min_tokens: int, overlap: int) -> None:
    """
    Validate configuration parameters.
    """
    if not catalog_path.exists():
        raise ConfigurationError(f"Catalog file not found: {catalog_path}")
    
    if not text_dir.exists():
        raise ConfigurationError(f"Text directory not found: {text_dir}")
    
    if max_tokens <= 0:
        raise ConfigurationError(f"max_tokens must be positive: {max_tokens}")
    
    if min_tokens <= 0:
        raise ConfigurationError(f"min_tokens must be positive: {min_tokens}")
    
    if overlap < 0:
        raise ConfigurationError(f"overlap cannot be negative: {overlap}")
    
    if overlap >= max_tokens:
        raise ConfigurationError(f"overlap ({overlap}) must be less than max_tokens ({max_tokens})")
    
    if min_tokens > max_tokens:
        raise ConfigurationError(f"min_tokens ({min_tokens}) cannot be greater than max_tokens ({max_tokens})")

def process_single_document(row: pd.Series, text_dir: Path, out_dir: Path, 
                          max_tokens: int, overlap: int, min_tokens: int, 
                          auto_toc: bool) -> Dict[str, Any]:
    """
    Process a single document for chunking with improved Vietnamese support.
    """
    start_time = time.time()
    doc_id = str(row["doc_id"])
    
    stats = {
        "doc_id": doc_id,
        "status": "success",
        "pages_processed": 0,
        "pages_skipped": 0,
        "paragraphs": 0,
        "chunks": 0,
        "processing_time": 0,
        "errors": []
    }
    
    try:
        text_path = text_dir / f"{doc_id}.pages.jsonl"
        if not text_path.exists():
            raise ChunkingError(f"Text file not found: {text_path}")
        
        # Load pages
        pages = []
        with text_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    page_num = int(obj["page"])
                    text = obj.get("text", "")
                    pages.append((page_num, text))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    error_msg = f"Invalid JSON on line {line_num}: {e}"
                    stats["errors"].append(error_msg)
                    logger.warning(f"{doc_id}: {error_msg}")
        
        if not pages:
            raise ChunkingError(f"No valid pages found in {text_path}")
        
        # Parse skip pages from catalog
        try:
            skip_pages = parse_skip_pages(str(row.get("skip_pages", "")))
        except ValueError as e:
            error_msg = f"Invalid skip_pages format: {e}"
            stats["errors"].append(error_msg)
            logger.warning(f"{doc_id}: {error_msg}")
            skip_pages = set()
        
        # Filter pages
        original_count = len(pages)
        if skip_pages:
            pages = [(pg, txt) for (pg, txt) in pages if pg not in skip_pages]
            logger.debug(f"{doc_id}: Skipped {original_count - len(pages)} pages by catalog")
        
        # Auto-skip TOC pages
        if auto_toc:
            before_toc = len(pages)
            pages = [(pg, txt) for (pg, txt) in pages if not is_toc_page(txt)]
            toc_skipped = before_toc - len(pages)
            if toc_skipped > 0:
                logger.debug(f"{doc_id}: Auto-skipped {toc_skipped} TOC pages")
        
        stats["pages_processed"] = len(pages)
        stats["pages_skipped"] = original_count - len(pages)
        
        if not pages:
            raise ChunkingError("No pages remaining after filtering")
        
        # Convert to paragraphs
        ordered_texts = [txt for _, txt in sorted(pages, key=lambda x: x[0])]
        paragraphs = paragraphs_from_pages(ordered_texts)
        stats["paragraphs"] = len(paragraphs)
        
        if not paragraphs:
            raise ChunkingError("No paragraphs extracted")
        
        # Attach sections
        section_paragraphs = attach_sections(paragraphs)
        
        # Create chunks
        paragraph_texts = [p for (sec, p) in section_paragraphs]
        chunks = chunk_with_overlap(paragraph_texts, max_tokens, overlap, min_tokens)
        stats["chunks"] = len(chunks)
        
        if not chunks:
            raise ChunkingError("No chunks created")
        
        # Build chunk payloads with improved section matching
        chunk_records = []
        for i, chunk_text in enumerate(chunks):
            # Find section for this chunk by matching the first meaningful paragraph
            chunk_paragraphs = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
            section = ""
            
            if chunk_paragraphs:
                first_para = chunk_paragraphs[0]
                # Find the section this chunk belongs to
                for (sec, para) in section_paragraphs:
                    if para and first_para.startswith(para[:min(100, len(para))]):
                        section = sec
                        break
                
                # If no exact match, try to find section by partial content match
                if not section:
                    for (sec, para) in section_paragraphs:
                        if para and any(first_para in chunk_para for chunk_para in chunk_paragraphs):
                            section = sec
                            break
            
            chunk_id = f"{doc_id}_{i:05d}"
            
            # Tạo tên tài liệu ngắn gọn
            title = str(row.get("title", ""))
            source = str(row.get("source", ""))
            if title and source:
                document = f"{title} ({source})"
            elif title:
                document = title
            else:
                document = source if source else "Tài liệu không xác định"
            
            # Trích xuất thông tin trang từ section hoặc chunk_text
            page_info = extract_page_info(section, chunk_text)
            
            # Tạo record đơn giản
            record = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "document": document
            }
            
            # Chỉ thêm page_info nếu có
            if page_info:
                record["page_info"] = page_info
            chunk_records.append(record)
        
        # Write output
        out_file = out_dir / f"{doc_id}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for record in chunk_records:
                # Clean and normalize text before saving
                if "text" in record and record["text"]:
                    # Remove problematic characters and normalize
                    clean_text = record["text"]
                    # Remove null bytes and control characters
                    clean_text = clean_text.replace('\x00', ' ')
                    clean_text = ' '.join(clean_text.split())  # Normalize whitespace
                    record["text"] = clean_text
                
                # Clean document field
                if "document" in record and record["document"]:
                    clean_field = str(record["document"])
                    clean_field = clean_field.replace('\x00', ' ')
                    clean_field = ' '.join(clean_field.split())
                    record["document"] = clean_field
                
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        
        stats["processing_time"] = time.time() - start_time
        logger.info(f"Successfully processed {doc_id}: {stats['chunks']} chunks, "
                   f"{stats['paragraphs']} paragraphs, {stats['processing_time']:.2f}s")
        
    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        stats["processing_time"] = time.time() - start_time
        logger.error(f"Failed to process {doc_id}: {e}")
    
    return stats

# ---------- Main ----------
def main():
    """Main function with improved Vietnamese document processing."""
    parser = argparse.ArgumentParser(
        description="Create text chunks from extracted text with improved Vietnamese support"
    )
    parser.add_argument("--catalog", default="../../data/metadata/catalog.csv",
                       help="Path to catalog CSV file")
    parser.add_argument("--text_dir", default="../../data/processed/text",
                       help="Directory containing extracted text files")
    parser.add_argument("--out_dir", default="../../data/processed/chunks",
                       help="Output directory for chunk files")
    parser.add_argument("--max_tokens", type=int, default=800,
                       help="Maximum tokens per chunk")
    parser.add_argument("--overlap", type=int, default=120,
                       help="Number of overlap tokens between chunks")
    parser.add_argument("--min_tokens", type=int, default=120,
                       help="Minimum tokens per chunk")
    parser.add_argument("--auto_toc", action="store_true",
                       help="Automatically skip table of contents pages")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Convert to Path objects
    catalog_path = Path(args.catalog)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.out_dir)
    
    try:
        # Validate configuration
        validate_configuration(catalog_path, text_dir, args.max_tokens, 
                             args.min_tokens, args.overlap)
        
        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load catalog
        catalog = pd.read_csv(catalog_path)
        logger.info(f"Loaded catalog with {len(catalog)} documents")
        
        # Filter valid documents
        valid_rows = []
        for _, row in catalog.iterrows():
            doc_id = str(row["doc_id"])
            text_path = text_dir / f"{doc_id}.pages.jsonl"
            if text_path.exists():
                valid_rows.append(row)
            else:
                logger.warning(f"Text file not found for {doc_id}: {text_path}")
        
        if not valid_rows:
            logger.error("No valid documents found")
            return 1
        
        logger.info(f"Processing {len(valid_rows)} valid documents")
        
        # Process documents
        max_workers = min(args.max_workers, len(valid_rows))
        all_stats = []
        
        if max_workers == 1:
            # Sequential processing
            for row in tqdm(valid_rows, desc="Chunking documents"):
                stats = process_single_document(
                    row, text_dir, out_dir, args.max_tokens, 
                    args.overlap, args.min_tokens, args.auto_toc
                )
                all_stats.append(stats)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_doc = {
                    executor.submit(
                        process_single_document, row, text_dir, out_dir,
                        args.max_tokens, args.overlap, args.min_tokens, args.auto_toc
                    ): str(row["doc_id"])
                    for row in valid_rows
                }
                
                for future in tqdm(as_completed(future_to_doc), 
                                 total=len(future_to_doc),
                                 desc="Chunking documents"):
                    doc_id = future_to_doc[future]
                    try:
                        stats = future.result()
                        all_stats.append(stats)
                    except Exception as e:
                        logger.error(f"Task failed for {doc_id}: {e}")
                        all_stats.append({
                            "doc_id": doc_id,
                            "status": "failed",
                            "error": str(e)
                        })
        
        # Print summary
        successful = [s for s in all_stats if s["status"] == "success"]
        failed = [s for s in all_stats if s["status"] != "success"]
        
        total_chunks = sum(s.get("chunks", 0) for s in successful)
        total_paragraphs = sum(s.get("paragraphs", 0) for s in successful)
        total_time = sum(s.get("processing_time", 0) for s in successful)
        
        logger.info(f"\n=== CHUNKING SUMMARY ===")
        logger.info(f"Total documents: {len(all_stats)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total chunks created: {total_chunks}")
        logger.info(f"Total paragraphs processed: {total_paragraphs}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Average chunks per document: {total_chunks / max(len(successful), 1):.1f}")
        
        if failed:
            logger.warning(f"Failed documents: {[s['doc_id'] for s in failed]}")
        
        return 0 if not failed else 1
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
