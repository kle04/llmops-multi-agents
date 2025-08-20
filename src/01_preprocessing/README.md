# Text Preprocessing Pipeline

Text extraction và chunking pipeline cho Vietnamese mental health documents.

## 🏗️ Overview

Pipeline này chuyển đổi PDF documents thành text chunks sẵn sàng cho embedding:

```
📄 PDF Files → 📝 Text Extraction → ✂️ Chunking → 📦 JSONL Chunks
```

## 📋 Steps

### Step 1: Extract Text from PDFs
```bash
python 01_extract_text.py \
  --catalog data/metadata/catalog.csv \
  --raw_dir data/raw \
  --out_dir data/processed/text
```

**Input:** PDF files + catalog metadata  
**Output:** 
- `{doc_id}.pages.jsonl` - Text per page
- `{doc_id}.merged.txt` - Full document text

### Step 2: Create Text Chunks  
```bash
python 02_chunk_text.py \
  --catalog data/metadata/catalog.csv \
  --text_dir data/processed/text \
  --out_dir data/processed/chunks \
  --max_tokens 800 \
  --overlap 120 \
  --min_tokens 120 \
  --auto_toc
```

**Input:** Page texts from step 1  
**Output:** `{doc_id}.jsonl` - Chunks with metadata

## ⚙️ Configuration

### Text Extraction Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--catalog` | `data/metadata/catalog.csv` | Catalog CSV file |
| `--raw_dir` | `data/raw` | Directory with PDF files |
| `--out_dir` | `data/processed/text` | Output directory |
| `--max_workers` | `4` | Parallel workers |
| `--log_level` | `INFO` | Logging level |

### Chunking Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_tokens` | `800` | Maximum tokens per chunk |
| `--overlap` | `120` | Overlap tokens between chunks |
| `--min_tokens` | `120` | Minimum tokens per chunk |
| `--auto_toc` | `False` | Auto-skip table of contents |

## 📁 File Structure

```
src/01_preprocessing/
├── 01_extract_text.py     # PDF → Text extraction
├── 02_chunk_text.py       # Text → Chunks
├── requirements.txt       # Python dependencies
└── README.md             # This file

Input:
data/
├── metadata/catalog.csv   # Document metadata
└── raw/*.pdf             # Source PDF files

Output:
data/processed/
├── text/
│   ├── {doc_id}.pages.jsonl   # Text per page
│   └── {doc_id}.merged.txt    # Full document
└── chunks/
    └── {doc_id}.jsonl        # Final chunks
```

## 📊 Document Metadata

The `catalog.csv` contains document information:

```csv
doc_id,filename,title,source,year,language,audience,grade_range,topics
USSH_VaccineTinhThan_SoTay_ChamSoc_SKTT_SinhVien_vi,USSH_VaccineTinhThan_SoTay_ChamSoc_SKTT_SinhVien_vi.pdf,Sổ tay chăm sóc sức khỏe tinh thần cho sinh viên,USSH – ĐHQG-HCM,,vi,sinh_vien,,stress;lo_au;tram_cam;thich_ung
```

**Key fields:**
- `doc_id`: Unique document identifier
- `filename`: PDF file name
- `title`: Document title  
- `audience`: Target audience (sinh_vien, hoc_sinh_pho_thong, etc.)
- `topics`: Semicolon-separated topics
- `skip_pages`: Optional page ranges to skip (e.g., "1-3,5")

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full preprocessing pipeline
python 01_extract_text.py
python 02_chunk_text.py

# Check output
ls data/processed/chunks/
```

## 🔧 Advanced Usage

### Custom Chunking Strategy
```bash
# Larger chunks for better context
python 02_chunk_text.py --max_tokens 1200 --overlap 200

# Smaller chunks for precise retrieval  
python 02_chunk_text.py --max_tokens 400 --overlap 80
```

### Skip Specific Pages
Add `skip_pages` column to catalog:
```csv
doc_id,filename,skip_pages
doc1,file1.pdf,"1-2,10"  # Skip pages 1,2,10
```

### Parallel Processing
```bash
# Use more workers for faster processing
python 01_extract_text.py --max_workers 8
```

## 📈 Performance

**Text Extraction:**
- ~2-5 pages/second per worker
- Memory usage: ~100MB per PDF
- Supports PyMuPDF (faster) and PyPDF2 (fallback)

**Chunking:**
- ~1000 chunks/second
- Memory efficient streaming processing
- Automatic header/footer detection

## 🧹 Output Format

### Page JSONL (`{doc_id}.pages.jsonl`)
```json
{"doc_id": "doc1", "page": 1, "text": "Page 1 content..."}
{"doc_id": "doc1", "page": 2, "text": "Page 2 content..."}
```

### Chunk JSONL (`{doc_id}.jsonl`)
```json
{
  "id": "doc1_00001",
  "chunk_id": "doc1_00001", 
  "doc_id": "doc1",
  "title": "Document Title",
  "source": "Publisher",
  "year": "2023",
  "language": "vi",
  "audience": "sinh_vien",
  "grade_range": "",
  "topics": "stress;lo_au",
  "section": "PHẦN 1. GIỚI THIỆU",
  "text": "Chunk content here...",
  "token_count": 456
}
```

## 🛠️ Troubleshooting

**No PDFs found:**
```bash
# Check PDF directory
ls data/raw/*.pdf

# Verify catalog paths
head data/metadata/catalog.csv
```

**Memory issues:**
```bash
# Reduce parallel workers
python 01_extract_text.py --max_workers 2

# Process one file at a time
python 01_extract_text.py --max_workers 1
```

**Empty chunks:**
```bash
# Lower minimum token threshold
python 02_chunk_text.py --min_tokens 50

# Check text extraction output
ls data/processed/text/
```

## 🎯 Next Steps

After preprocessing, use the chunks for:
1. **Embedding generation** (`../02_embedding/`)
2. **Vector database storage** (Qdrant)
3. **RAG system integration**

The processed chunks are ready for the embedding pipeline!
