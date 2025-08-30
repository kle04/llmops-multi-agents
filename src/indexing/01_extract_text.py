import json
import re
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_with_pypdf2(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_with_fitz(pdf_path):
    """Extract text from a PDF file using fitz (PyMuPDF)."""
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):  # Start from 0 to include all pages
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def preprocess_and_chunk_text(text):
    """
    Preprocess the text and split it into chunks with titles and contexts.
    - Handles structure like: "PHẦN 1", "1.1. Nội dung", "1.2. Nội dung"
    - Titles include both the current part and section.
    - Contexts contain the text under each section.
    """
    # Define regex patterns for identifying parts and sections
    # Pattern for "Phần X" or "PHẦN X" with optional title 
    part_pattern = r"(PHẦN\s+\d+(?:\.|:)?\s*[^\n]*)"
    # Pattern for numbered sections like "1.1.", "1.2.", "2.1.1." etc.
    # Also handles Roman numerals and letters like "I.", "A.", "a)", etc.
    section_pattern = r"(\d+(?:\.\d+)*\.?\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ][^\n]*)"
    
    # Split the text into sections based on parts and sections
    sections = re.split(f"({part_pattern}|{section_pattern})", text, flags=re.DOTALL)
    
    # Initialize variables for processing
    current_part = None
    current_section = None
    chunks = []
    buffer = ""
    
    for section in sections:
        # Skip None or empty sections
        if section is None or not section.strip():
            continue
        
        # Check if the section is a part title (PHẦN X)
        part_match = re.match(part_pattern, section.strip())
        if part_match:
            # If there's a previous section, save its content as a chunk
            if current_section and buffer.strip():
                # Create title in format: "Phần X: Title, Section"
                chunk_title = f"{current_part}, {current_section}"
                
                chunk = {
                    "title": chunk_title,
                    "context": buffer.strip()
                }
                chunks.append(chunk)
            
            # Update the current part
            current_part = section.strip()
            current_section = None  # Reset section when a new part starts
            buffer = ""  # Reset buffer for new part
            continue
        
        # Check if the section is a numbered section (1.1., 1.2., etc.)
        section_match = re.match(section_pattern, section.strip())
        if section_match:
            # If there's a previous section, save its content as a chunk
            if current_section and buffer.strip():
                # Create title in format: "Phần X: Title, Section"
                chunk_title = f"{current_part}, {current_section}" if current_part else current_section
                    
                chunk = {
                    "title": chunk_title,
                    "context": buffer.strip()
                }
                chunks.append(chunk)
            
            # Update the current section
            current_section = section.strip()
            buffer = ""  # Reset buffer for new section
            continue
        
        # If it's neither a part nor a section, it's part of the current section's content
        if current_section:
            buffer += " " + section.strip()
    
    # Add the last chunk if there's any remaining content
    if current_section and buffer.strip():
        # Create title in format: "Phần X: Title, Section"
        chunk_title = f"{current_part}, {current_section}" if current_part else current_section
            
        chunk = {
            "title": chunk_title,
            "context": buffer.strip()
        }
        chunks.append(chunk)
    
    return chunks

def split_long_context(title, context, max_length=800):
    """
    Split long context into smaller chunks using RecursiveCharacterTextSplitter.
    Each smaller chunk retains the same title.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=300,  # Overlap to ensure continuity between chunks
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    sub_chunks = splitter.split_text(context)
    return [{"title": title, "context": sub_chunk.strip()} for sub_chunk in sub_chunks]

def process_pdf(pdf_path, output_json, extraction_method="fitz"):
    """Process the PDF and save the output as JSON."""
    print(f"🔄 Processing PDF: {pdf_path}")
    print(f"📁 Output: {output_json}")
    print(f"🔧 Method: {extraction_method}")
    print("=" * 60)
    
    # Step 1: Extract text from the PDF using the specified method
    if extraction_method == "pypdf2":
        raw_text = extract_text_with_pypdf2(pdf_path)
    elif extraction_method == "fitz":
        raw_text = extract_text_with_fitz(pdf_path)
    else:
        raise ValueError("Unsupported extraction method. Choose 'pypdf2' or 'fitz'.")
    
    print(f"📊 Total text length: {len(raw_text):,} characters")
    print(f"📄 Text preview (first 300 chars):")
    print("-" * 40)
    print(raw_text[:300])
    print("-" * 40)
    
    # Step 2: Preprocess and chunk the text
    print("\n🔍 Extracting sections and chunks...")
    chunks = preprocess_and_chunk_text(raw_text)
    
    # Step 3: Split long contexts into smaller chunks
    print(f"📋 Initial chunks found: {len(chunks)}")
    final_chunks = []
    
    for i, chunk in enumerate(chunks):
        title = chunk["title"]
        context = chunk["context"]
        
        print(f"  Chunk {i+1}: '{title}' ({len(context)} chars)")
        
        if len(context) > 800:  # If context is too long, split it
            sub_chunks = split_long_context(title, context)
            final_chunks.extend(sub_chunks)
            print(f"    → Split into {len(sub_chunks)} smaller chunks")
        else:
            final_chunks.append(chunk)
    
    print(f"\n📊 Final statistics:")
    print(f"   Total chunks: {len(final_chunks)}")
    print(f"   Average length: {sum(len(c['context']) for c in final_chunks) // len(final_chunks) if final_chunks else 0} chars")
    
    # Show sample chunks
    print(f"\n📝 Sample chunks:")
    for i, chunk in enumerate(final_chunks[:3], 1):
        print(f"   {i}. Title: {chunk['title']}")
        print(f"      Context: {chunk['context'][:80]}...")
    
    # Step 4: Save the chunks to a JSON file
    save_to_json(final_chunks, output_json)
    print(f"\n✅ Saved {len(final_chunks)} chunks to {output_json}")
    
    return final_chunks

def save_to_json(data, output_file):
    """Save the processed data to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def test_patterns():
    """Test the regex patterns with sample text."""
    sample_text = """
    PHẦN 1. TỔNG QUAN VỀ TÂM LÝ HỌC

    1.1. Khái niệm về tâm lý học
    Tâm lý học là khoa học nghiên cứu về hoạt động tâm lý của con người. Đây là một ngành khoa học quan trọng giúp hiểu được bản chất của con người.

    1.2. Đối tượng nghiên cứu của tâm lý học
    Đối tượng nghiên cứu của tâm lý học bao gồm các hiện tượng tâm lý như nhận thức, cảm xúc, hành vi và các quá trình tâm lý khác.

    PHẦN 2. CÁC PHƯƠNG PHÁP NGHIÊN CỨU

    2.1. Phương pháp quan sát
    Quan sát là phương pháp nghiên cứu cơ bản trong tâm lý học. Phương pháp này cho phép nghiên cứu viên quan sát hành vi tự nhiên của đối tượng nghiên cứu.

    2.2. Phương pháp thử nghiệm
    Thử nghiệm cho phép nghiên cứu mối quan hệ nhân quả giữa các biến số. Đây là phương pháp có tính khoa học cao và được sử dụng rộng rãi.
    """
    
    print("🧪 Testing extraction patterns...")
    print("=" * 60)
    chunks = preprocess_and_chunk_text(sample_text)
    
    print(f"📊 Number of chunks extracted: {len(chunks)}")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📝 Chunk {i}:")
        print(f"📌 Title: {chunk['title']}")
        print(f"📄 Context: {chunk['context'][:100]}...")
        print("-" * 40)
    
    print(f"\n✅ Test completed! Expected format:")
    print(f"   Title: 'PHẦN X: Title, Section'")
    print(f"   Context: 'Content of that section'")
    
    return chunks

# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Check if user wants to test patterns
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_patterns()
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        # Process all PDFs in data/raw/
        raw_dir = Path("../../data/raw/")
        pdf_files = list(raw_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ No PDF files found in data/raw/")
            sys.exit(1)
        
        print(f"📚 Found {len(pdf_files)} PDF files to process:")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        
        print("\n" + "="*80)
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n🔄 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            output_json = f"../../data/processed/chunks/{pdf_path.stem}.json"
            
            try:
                process_pdf(str(pdf_path), output_json, "fitz")
                print(f"✅ Completed: {pdf_path.name}")
            except Exception as e:
                print(f"❌ Failed: {pdf_path.name} - {e}")
            
            if i < len(pdf_files):
                print("\n" + "-"*60)
        
        print(f"\n🎉 Finished processing all {len(pdf_files)} PDF files!")
        
    else:
        # Process single PDF (default behavior)
        pdf_path = "../../data/raw/MOET_SoTay_ThucHanh_CTXH_TrongTruongHoc_vi.pdf"
        output_json = "output.json"
        
        # Choose the extraction method: "pypdf2" or "fitz"
        extraction_method = "fitz"  # Change to "pypdf2" if needed
        
        try:
            process_pdf(pdf_path, output_json, extraction_method)
        except FileNotFoundError:
            print(f"❌ PDF file not found: {pdf_path}")
            print("💡 Usage:")
            print("   python 01_extract_text.py       # Process default PDF")
            print("   python 01_extract_text.py test  # Test patterns")
            print("   python 01_extract_text.py all   # Process all PDFs in data/raw/")