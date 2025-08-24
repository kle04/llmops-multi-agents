# LLMOps Multi-Agents - Source Code

 **Hệ thống LLMOps Multi-Agent cho tư vấn sức khỏe tâm thần học sinh với kiến trúc microservices**

Thư mục này chứa source code của hệ thống LLMOps Multi Agents với kiến trúc microservices để xây dựng một hệ thống tư vấn và hỗ trợ sức khỏe tâm thần học sinh thông minh.


## Cấu trúc thư mục

```
src/
├── README.md                      # Tài liệu này
├── docker-compose.yml             # Orchestration toàn bộ services
├── preprocessing/                 # Data pipeline & preprocessing
│   ├── 01_extract_text.py          # Trích xuất text từ PDF
│   ├── 02_chunk_text.py            # Chia text thành chunks
│   ├── 03_create_embeddings.py     # Tạo embeddings và lưu vector DB
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Chi tiết data pipeline
├── deploy_embedding/              # Vietnamese Embedding Service
│   ├── app.py                      # FastAPI embedding API
│   ├── Dockerfile                  # Container definition
│   ├── pyproject.toml              # Poetry dependencies
│   └── README.md                   # Chi tiết embedding service
├── context-retrieval/             # RAG Context Retrieval Service  
│   ├── app.py                      # FastAPI retrieval API
│   ├── Dockerfile                  # Container definition
│   └── pyproject.toml              # Poetry dependencies
```

## Quick Start

### 1. Khởi động toàn bộ hệ thống

```bash
# Clone và di chuyển vào thư mục
cd src/

# Khởi động tất cả services
docker-compose up -d

# Kiểm tra trạng thái
docker-compose ps
```

### 2. Xử lý dữ liệu (Data Pipeline)

```bash
cd preprocessing/

# Chạy pipeline hoàn chỉnh từ PDF đến Vector DB
python run_pipeline.py --check_services --recreate_collection

# Hoặc chạy từng bước riêng lẻ
python 01_extract_text.py --catalog ../../data/metadata/catalog.csv
python 02_chunk_text.py --auto_toc --max_tokens 800
python 03_create_embeddings.py --recreate_collection
```

### 3. Test hệ thống

```bash
# Test embedding service
curl http://localhost:5000/health

# Test context retrieval
curl -X POST http://localhost:5005/search \
  -H "Content-Type: application/json" \
  -d '{"query": "học sinh bị trầm cảm", "limit": 5}'

# Chạy integration tests
cd ../../test_scripts/
python test_integration.py --embedding_url http://localhost:5000 --retrieval_url http://localhost:5005
```

## Chi tiết các service

### Embedding Service (`deploy_embedding/`)
- **Model:** AITeamVN/Vietnamese_Embedding
- **API:** FastAPI với endpoint `/embed` và `/health`
- **Port:** 5000
- **Features:** Vietnamese text embedding, 1024-dimensional vectors

### Context Retrieval Service (`context-retrieval/`)
- **Database:** Qdrant vector database
- **API:** Search endpoints với filtering và scoring
- **Port:** 5005
- **Features:** Semantic search, metadata filtering, relevance scoring

### Data Processing Pipeline (`preprocessing/`)
- **Input:** PDF documents với metadata catalog
- **Output:** Vector embeddings trong Qdrant collection
- **Features:** Text extraction, intelligent chunking, batch processing

### Vector Database (Qdrant)
- **Port:** 6333
- **Collection:** `mental_health_vi`
- **Features:** Persistent storage, high-performance search

## Kiểm thử, đảm bảo chất lượng

### Comprehensive Test Suite
```bash
cd ../test_scripts/

# Test individual services
python test_embedding_server.py --base_url http://localhost:5000
python test_context_retrieval.py --base_url http://localhost:5005

# Full integration test
python test_integration.py --load_requests 50 --max_concurrent 10
```

### Các ngữ cảnh test
- ✅ **Health checks** cho tất cả services
- ✅ **Single query tests** với Vietnamese queries
- ✅ **Batch processing tests** với multiple requests
- ✅ **Load testing** với concurrent requests
- ✅ **Integration testing** end-to-end pipeline

