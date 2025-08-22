#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class PDFExtractionError(Exception):
    """Raised when all PDF extraction methods fail"""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass

# Pre-compile regex patterns for performance
PATTERNS = {
    'line_break': re.compile(r"(?<!\n)\n(?!\n)"),
    'whitespace': re.compile(r"[ \t]+"),
    'multiple_newlines': re.compile(r"\n{3,}"),
    'page_number': re.compile(r"^\d{1,4}$"),
    'empty_line': re.compile(r"^\s*$")
}

def extract_pages(pdf_path: Path) -> Tuple[List[str], str]:
    """
    Extract text from PDF using available libraries.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (pages_text, extraction_method)
        
    Raises:
        PDFExtractionError: When all extraction methods fail
    """
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF file not found: {pdf_path}")
    
    # Try PyMuPDF first (more robust and faster)
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf_path) as doc:
            pages = []
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")
                    pages.append(text if text else "")
                except Exception as e:
                    logger.warning(f"PyMuPDF failed on page {page_num + 1} of {pdf_path}: {e}")
                    pages.append("")
        logger.debug(f"Successfully extracted {len(pages)} pages using PyMuPDF from {pdf_path}")
        return pages, "pymupdf"
        
    except ImportError:
        logger.warning("PyMuPDF not available, falling back to PyPDF2")
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
    
    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        pages = []
        
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                pages.append(text if text else "")
            except Exception as e:
                logger.warning(f"PyPDF2 failed on page {i + 1} of {pdf_path}: {e}")
                pages.append("")
                
        logger.debug(f"Successfully extracted {len(pages)} pages using PyPDF2 from {pdf_path}")
        return pages, "pypdf2"
        
    except ImportError:
        raise PDFExtractionError(f"No PDF extraction library available. Install PyMuPDF or PyPDF2")
    except Exception as e:
        raise PDFExtractionError(f"All extraction methods failed for {pdf_path}: {e}")

@lru_cache(maxsize=256)
def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text with caching for performance.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text or not text.strip():
        return ""
    
    # Join broken lines (not double newlines) - be more careful with Vietnamese text
    text = PATTERNS['line_break'].sub(" ", text)
    # Compress multiple spaces/tabs
    text = PATTERNS['whitespace'].sub(" ", text)
    # Replace 3+ newlines with 2 newlines (preserve paragraphs)
    text = PATTERNS['multiple_newlines'].sub("\n\n", text)
    
    # Remove any null bytes or control characters that might interfere with JSON
    text = text.replace('\x00', ' ')
    
    return text.strip()

def strip_headers_footers(pages: List[str]) -> List[str]:
    """
    Remove headers and footers from pages.
    
    Args:
        pages: List of page texts
        
    Returns:
        List of cleaned page texts
    """
    if not pages:
        return pages
    
    cleaned = []
    first_line_count = {}
    
    # First pass: collect statistics and remove page numbers
    for page_text in pages:
        if not page_text.strip():
            cleaned.append("")
            continue
            
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        
        # Remove lines that are just page numbers
        lines = [line for line in lines if not PATTERNS['page_number'].fullmatch(line)]
        
        if not lines:
            cleaned.append("")
            continue
            
        # Count first line occurrences for header detection
        first_line = lines[0]
        first_line_count[first_line] = first_line_count.get(first_line, 0) + 1
        cleaned.append("\n".join(lines))
    
    # Identify common headers (appear in >50% of pages)
    threshold = len(pages) * 0.5
    common_headers = {header for header, count in first_line_count.items() if count > threshold}
    
    if common_headers:
        logger.info(f"Detected {len(common_headers)} common headers to remove")
    
    # Second pass: remove detected headers
    final_pages = []
    for page_text in cleaned:
        if not page_text:
            final_pages.append("")
            continue
            
        lines = page_text.splitlines()
        if lines and lines[0] in common_headers:
            lines = lines[1:]
            
        final_pages.append("\n".join(lines))
    
    return final_pages

def process_single_pdf(pdf_path: Path, out_dir: Path, doc_id: str) -> Dict[str, Any]:
    """
    Process a single PDF file with error handling and statistics.
    
    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory
        doc_id: Document ID
        
    Returns:
        Processing statistics dictionary
    """
    start_time = time.time()
    stats = {
        "doc_id": doc_id,
        "status": "success",
        "pages": 0,
        "method": "unknown",
        "errors": 0,
        "processing_time": 0,
        "file_size": 0
    }
    
    try:
        # Get file size for stats
        stats["file_size"] = pdf_path.stat().st_size
        
        # Extract pages
        raw_pages, extraction_method = extract_pages(pdf_path)
        stats["method"] = extraction_method
        stats["pages"] = len(raw_pages)
        
        # Process pages with streaming to reduce memory usage
        out_path = out_dir / f"{doc_id}.pages.jsonl"
        processed_pages = []
        
        with out_path.open("w", encoding="utf-8") as f:
            for i, raw_text in enumerate(raw_pages, start=1):
                try:
                    # Normalize text
                    normalized_text = normalize_whitespace(raw_text)
                    
                    # Write immediately to reduce memory usage
                    page_data = {
                        "doc_id": doc_id,
                        "page": i,
                        "text": normalized_text
                    }
                    json.dump(page_data, f, ensure_ascii=False)
                    f.write("\n")
                    
                    processed_pages.append(normalized_text)
                    
                except Exception as e:
                    logger.error(f"Error processing page {i} of {doc_id}: {e}")
                    stats["errors"] += 1
                    processed_pages.append("")
        
        # Clean headers/footers and create merged file
        cleaned_pages = strip_headers_footers(processed_pages)
        merged_text = "\n\n".join(cleaned_pages)
        
        merged_path = out_dir / f"{doc_id}.merged.txt"
        merged_path.write_text(merged_text, encoding="utf-8")
        
        stats["processing_time"] = time.time() - start_time
        logger.info(f"Successfully processed {doc_id}: {stats['pages']} pages, "
                   f"{stats['errors']} errors, {stats['processing_time']:.2f}s")
        
    except PDFExtractionError as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        logger.error(f"Failed to process {doc_id}: {e}")
        
    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        logger.error(f"Unexpected error processing {doc_id}: {e}")
    
    finally:
        stats["processing_time"] = time.time() - start_time
    
    return stats

def validate_configuration(catalog_path: Path, raw_dir: Path) -> None:
    """
    Validate configuration and inputs.
    
    Args:
        catalog_path: Path to catalog CSV
        raw_dir: Path to raw PDF directory
        
    Raises:
        ConfigurationError: When configuration is invalid
    """
    if not catalog_path.exists():
        raise ConfigurationError(f"Catalog file not found: {catalog_path}")
    
    if not raw_dir.exists():
        raise ConfigurationError(f"Raw directory not found: {raw_dir}")
    
    try:
        df = pd.read_csv(catalog_path)
        required_columns = ["doc_id", "filename"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ConfigurationError(f"Missing required columns in catalog: {missing_columns}")
    except Exception as e:
        raise ConfigurationError(f"Invalid catalog file: {e}")

def main():
    """Main function with improved error handling and parallel processing."""
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files with optimized performance and error handling"
    )
    parser.add_argument("--catalog", default="data/metadata/catalog.csv", 
                       help="Path to catalog CSV file")
    parser.add_argument("--raw_dir", default="data/raw", 
                       help="Directory containing source PDF files")
    parser.add_argument("--out_dir", default="data/processed/text", 
                       help="Output directory for processed text files")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel workers (default: 4)")
    parser.add_argument("--log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Convert to Path objects
    catalog_path = Path(args.catalog)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    
    try:
        # Validate configuration
        validate_configuration(catalog_path, raw_dir)
        
        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load catalog
        df = pd.read_csv(catalog_path)
        logger.info(f"Loaded catalog with {len(df)} documents")
        
        # Filter existing files
        valid_rows = []
        for _, row in df.iterrows():
            pdf_path = raw_dir / row["filename"]
            if pdf_path.exists():
                valid_rows.append((pdf_path, row["doc_id"]))
            else:
                logger.warning(f"PDF file not found: {pdf_path}")
        
        if not valid_rows:
            logger.error("No valid PDF files found")
            return
        
        logger.info(f"Processing {len(valid_rows)} valid PDF files")
        
        # Process files in parallel
        max_workers = min(args.max_workers, len(valid_rows))
        all_stats = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(process_single_pdf, pdf_path, out_dir, doc_id): doc_id
                for pdf_path, doc_id in valid_rows
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_doc), 
                             total=len(future_to_doc), 
                             desc="Extracting PDFs"):
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
        
        # Print summary statistics
        successful = [s for s in all_stats if s["status"] == "success"]
        failed = [s for s in all_stats if s["status"] != "success"]
        
        total_pages = sum(s.get("pages", 0) for s in successful)
        total_time = sum(s.get("processing_time", 0) for s in successful)
        total_size = sum(s.get("file_size", 0) for s in successful)
        
        logger.info(f"\n=== EXTRACTION SUMMARY ===")
        logger.info(f"Total documents: {len(all_stats)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total pages extracted: {total_pages}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Total file size: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"Average speed: {total_pages / max(total_time, 0.1):.1f} pages/second")
        
        if failed:
            logger.warning(f"Failed documents: {[s['doc_id'] for s in failed]}")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
