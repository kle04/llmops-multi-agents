# Preprocessing Pipeline

Pipeline để xử lý dữ liệu từ PDF thành vector embeddings và lưu trữ trong Qdrant vector database.

## 🚀 Tổng quan

Pipeline bao gồm 3 bước chính:
1. **Extract text** (`01_extract_text.py`) - Trích xuất text từ PDF files
2. **Create chunks** (`02_chunk_text.py`) - Chia text thành các chunks nhỏ
3. **Create embeddings** (`03_create_embeddings.py`) - Tạo vector embeddings và upsert lên Qdrant

## 📋 Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

### External Services
- **Qdrant vector database** - chạy ở `localhost:6333`
- **Embedding service** - chạy ở `localhost:5000`

## 🔧 Cấu hình Services

### 1. Khởi động Qdrant
```bash
cd src
docker-compose up qdrant -d
```

### 2. Khởi động Embedding Service
```bash
cd src
docker-compose up deploy-embedding -d
```

### 3. Kiểm tra Services
```bash
# Kiểm tra Qdrant
curl http://localhost:6333/collections

# Kiểm tra Embedding Service
curl http://localhost:5000/health
```

## 🚀 Sử dụng

### Option 1: Chạy toàn bộ pipeline
```bash
cd src/preprocessing

# Chạy pipeline hoàn chỉnh
python run_pipeline.py

# Với các tùy chọn
python run_pipeline.py \
  --max_tokens 800 \
  --overlap 120 \
  --min_tokens 120 \
  --auto_toc \
  --max_workers 4 \
  --recreate_collection \
  --check_services
```

### Option 2: Chạy từng bước riêng lẻ

#### Bước 1: Trích xuất text
```bash
python 01_extract_text.py \
  --catalog ../../data/metadata/catalog.csv \
  --raw_dir ../../data/raw \
  --out_dir ../../data/processed/text \
  --max_workers 4
```

#### Bước 2: Tạo chunks
```bash
python 02_chunk_text.py \
  --catalog ../../data/metadata/catalog.csv \
  --text_dir ../../data/processed/text \
  --out_dir ../../data/processed/chunks \
  --max_tokens 800 \
  --overlap 120 \
  --min_tokens 120 \
  --auto_toc \
  --max_workers 4
```

#### Bước 3: Tạo embeddings và upsert
```bash
python 03_create_embeddings.py \
  --chunks_dir ../../data/processed/chunks \
  --qdrant_url http://localhost:6333 \
  --embedding_service_url http://localhost:5000 \
  --collection_name mental_health_vi \
  --max_workers 4
```

## ⚙️ Tham số cấu hình

### Text Chunking
- `--max_tokens`: Số token tối đa mỗi chunk (mặc định: 800)
- `--min_tokens`: Số token tối thiểu mỗi chunk (mặc định: 120)
- `--overlap`: Số token overlap giữa các chunks (mặc định: 120)
- `--auto_toc`: Tự động bỏ qua trang mục lục

### Embedding & Vector DB
- `--qdrant_url`: URL của Qdrant server (mặc định: http://localhost:6333)
- `--embedding_service_url`: URL của embedding service (mặc định: http://localhost:5000)
- `--collection_name`: Tên collection trong Qdrant (mặc định: mental_health_vi)
- `--recreate_collection`: Tạo lại collection nếu đã tồn tại

### Performance
- `--max_workers`: Số luồng xử lý song song (mặc định: 4)
- `--log_level`: Mức độ logging (DEBUG, INFO, WARNING, ERROR)

## 📊 Vector Database Schema

Mỗi chunk được lưu trữ với cấu trúc:

```json
{
  "id": "doc_chunk_id",
  "vector": [0.1, -0.2, 0.3, ...],  // 768-dimensional
  "payload": {
    "id": "chunk_identifier",
    "chunk_id": "unique_chunk_id",
    "doc_id": "source_document_id",
    "title": "Document title",
    "source": "Publishing organization",
    "year": 2022,
    "language": "vi",
    "audience": "target_audience",
    "grade_range": "education_levels",
    "topics": "relevant_topics",
    "section": "document_section",
    "text": "chunk_content",
    "token_count": 500
  }
}
```

## 🔍 Monitoring & Debugging

### Kiểm tra trạng thái collection
```bash
curl http://localhost:6333/collections/mental_health_vi
```

### Tìm kiếm test
```bash
# Sử dụng context-retrieval service
curl -X POST http://localhost:5005/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sức khỏe tâm thần học sinh",
    "limit": 5,
    "score_threshold": 0.7
  }'
```

### Logs và debugging
```bash
# Chạy với debug logging
python run_pipeline.py --log_level DEBUG

# Chỉ chạy embedding step với recreate
python run_pipeline.py --skip_extract --skip_chunk --recreate_collection
```

## 📁 Cấu trúc dữ liệu

```
data/
├── metadata/
│   └── catalog.csv              # Metadata của documents
├── raw/
│   └── *.pdf                    # PDF files gốc
├── processed/
│   ├── text/
│   │   ├── *.pages.jsonl        # Text đã extract theo trang
│   │   └── *.merged.txt         # Text đã merge
│   └── chunks/
│       └── *.jsonl              # Chunks với metadata
└── embeddings/
    └── *.npz                    # Vector embeddings (nếu có)
```

## ⚠️ Lưu ý

1. **Vector dimensions**: Đảm bảo embedding service trả về vector 768 dimensions
2. **Memory usage**: Xử lý file lớn có thể cần nhiều memory
3. **Service dependencies**: Đảm bảo Qdrant và embedding service đang chạy
4. **Collection schema**: Recreate collection nếu thay đổi vector dimensions

## 🐛 Troubleshooting

### Lỗi vector dimension mismatch
```bash
# Recreate collection với đúng dimensions
python 03_create_embeddings.py --recreate_collection
```

### Embedding service không khả dụng
```bash
# Kiểm tra service
docker-compose logs deploy-embedding

# Restart service
docker-compose restart deploy-embedding
```

### Qdrant connection failed
```bash
# Kiểm tra Qdrant
docker-compose logs qdrant

# Restart Qdrant
docker-compose restart qdrant
```

## 📈 Performance Tips

1. **Parallel processing**: Tăng `--max_workers` cho máy có nhiều CPU
2. **Batch size**: Script tự động chia thành batches để tối ưu memory
3. **Skip steps**: Sử dụng `--skip_*` để bỏ qua các bước đã hoàn thành
4. **Incremental updates**: Chỉ xử lý files mới bằng cách filter catalog
