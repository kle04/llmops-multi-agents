# Hướng dẫn xử lý data
## Bước 1: Cài đặt dependencies
```
pip install -r src/01_preprocessing/requirements.txt
pip install -r src/02_embedding/requirements.txt
```


## Bước 2: Chuyển data từ tài liệu pdf thành text

```
python src/01_preprocessing/01_extract_text.py \
  --catalog data/metadata/catalog.csv \
  --raw_dir data/raw \
  --out_dir data/processed/text
```
* Lúc này, text được extract sẽ được lưu ở data/processed/text


## Bước 3: Chunking data
```
python src/preprocessing/02_chunk_text.py \
  --catalog data/metadata/catalog.csv \
  --text_dir data/processed/text \
  --out_dir data/processed/chunks \
  --max_tokens 800 --overlap 120 --min_tokens 120 \
  --auto_toc
```
* Lúc này, data được chunk sẽ lưu ở data/processed/chunks

## Bước 4: Tạo vector embedding
```
python src/02_embedding/03_embedding.py \
  --chunks_dir data/processed/chunks \
  --out_dir data/embeddings \
  --model BAAI/bge-m3 \
  --batch_size 32 \
  --normalize
```
* Lúc này, sẽ tạo các vector ở data/embeddings


## Bước 5: Upsert vector vào Qdrant DB
```
python src/02_embedding/03b_upsert_qdrant.py \
  --emb_dir data/embeddings \
  --chunks_dir data/processed/chunks \
  --qdrant_url http://localhost:6333 \
  --collection mental_health_vi \
  --distance cosine \
  --batch_size 256
```