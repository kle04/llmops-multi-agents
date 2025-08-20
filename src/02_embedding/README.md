# Pipeline tạo embedding & qdrant
## Cách dùng

```bash
# Tạo Qdrant DB
docker-compose up qdrant -d

# Chạy embedding
docker-compose run --rm embedding
```

## Configuration

Biến môi trường:
- `CHUNKS_DIR`: Input chunks directory (default: `/workspace/data/processed/chunks`)
- `OUT_DIR`: Output embeddings directory (default: `/workspace/data/embeddings`)
- `MODEL`: Embedding model (default: `AITeamVN/Vietnamese_Embedding_v2`)
- `QDRANT_URL`: Qdrant server URL (default: `http://qdrant:6333`)
- `QDRANT_COLLECTION`: Collection name (default: `mental_health_vi`)

## Có tác dụng gì

1. Tạo embedding từ các file chunk
2. Upsert embeddings + metadata vào Qdrant vector database

