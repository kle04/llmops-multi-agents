# Hướng dẫn xử lý dữ liệu cho LLMOps Multi-Agent System

Thư mục này chứa pipeline xử lý dữ liệu hoàn chỉnh để chuyển đổi tài liệu PDF thành vector embeddings có thể tìm kiếm. Pipeline này được thiết kế đặc biệt cho tài liệu sức khỏe tâm thần tiếng Việt.

## 📋 Tổng quan Pipeline

Pipeline xử lý dữ liệu bao gồm 3 bước chính:

1. **Trích xuất văn bản** (`01_extract_text.py`) - Chuyển đổi PDF thành văn bản
2. **Phân đoạn văn bản** (`02_chunk_text.py`) - Chia văn bản thành các chunk có kích thước phù hợp
3. **Tạo embeddings** (`03_create_embeddings.py`) - Tạo vector embeddings và lưu vào Qdrant

### Cài đặt dependencies
```bash
# Từ thư mục root của project
pip install -r src/preprocessing/requirements.txt
```

### Cấu trúc dữ liệu đầu vào
Đảm bảo cấu trúc thư mục như sau:
```
data/
├── raw/                          # PDF files gốc
│   ├── document1.pdf
│   └── document2.pdf
├── metadata/
│   └── catalog.csv              # Metadata của documents
└── processed/
    ├── text/                    # Output của bước 1
    ├── chunks/                  # Output của bước 2
    └── embeddings/              # Output của bước 3 (optional)
```

### Cấu trúc file catalog.csv
File `data/metadata/catalog.csv` chứa các cột sau:
```csv
doc_id,filename,title,source,year,language,audience,grade_range,topics,skip_pages
MOET_001,MOET_SoTay_ThucHanh_CTXH_TrongTruongHoc_vi.pdf,Sổ tay thực hành CTXH trong trường học,MOET,2023,vi,teachers,K12,social_work,"1-2,50-52"
```

**Các cột bắt buộc:**
- `doc_id`: ID duy nhất của tài liệu
- `filename`: Tên file PDF trong thư mục `data/raw/`
- `title`: Tiêu đề tài liệu
- `source`: Nguồn tài liệu
- `year`: Năm xuất bản
- `language`: Ngôn ngữ (mặc định: vi)
- `audience`: Đối tượng mục tiêu
- `grade_range`: Cấp học
- `topics`: Chủ đề
- `skip_pages`: Các trang cần bỏ qua (format: "1-3,5,8-9")

## 🚀 Hướng dẫn sử dụng

**⚠️ Lưu ý quan trọng:** Luôn chạy các lệnh từ thư mục root của project.

### Bước 1: Trích xuất văn bản từ PDF

```bash
python src/preprocessing/01_extract_text.py --catalog data/metadata/catalog.csv --raw_dir data/raw --out_dir data/processed/text --log_level INFO
```

**Tham số chính:**
- `--catalog`: Đường dẫn đến file catalog.csv
- `--raw_dir`: Thư mục chứa PDF files
- `--out_dir`: Thư mục output cho văn bản đã trích xuất
- `--max_workers`: Số worker song song (mặc định: 4)
- `--log_level`: Mức độ logging (DEBUG, INFO, WARNING, ERROR)

**Output:**
- `{doc_id}.pages.jsonl`: Văn bản theo từng trang
- `{doc_id}.merged.txt`: Văn bản đã merge và làm sạch

### Bước 2: Phân đoạn văn bản

```bash
python src/preprocessing/02_chunk_text.py --catalog data/metadata/catalog.csv --text_dir data/processed/text --out_dir data/processed/chunks --max_tokens 800 --overlap 120 --min_tokens 120 --auto_toc
```

**Tham số chính:**
- `--text_dir`: Thư mục chứa văn bản đã trích xuất
- `--out_dir`: Thư mục output cho chunks
- `--max_tokens`: Số token tối đa mỗi chunk (mặc định: 800)
- `--overlap`: Số token overlap giữa các chunk (mặc định: 120)
- `--min_tokens`: Số token tối thiểu mỗi chunk (mặc định: 120)
- `--auto_toc`: Tự động bỏ qua trang mục lục

**Output:**
- `{doc_id}.jsonl`: Chunks với metadata đầy đủ

### Bước 3: Tạo vector embeddings

```bash
python src/preprocessing/03_create_embeddings.py --chunks_dir data/processed/chunks --out_dir data/embeddings --model dangvantuan/vietnamese-embedding --batch_size 8 --qdrant_url http://localhost:6333 --collection_name mental_health_vi
```

**Tham số chính:**
- `--chunks_dir`: Thư mục chứa chunks
- `--out_dir`: Thư mục output cho embeddings (optional)
- `--model`: Model embedding (mặc định: dangvantuan/vietnamese-embedding)
- `--batch_size`: Batch size cho encoding (mặc định: 8)
- `--device`: Device sử dụng (auto, cuda, mps, cpu)
- `--qdrant_url`: URL Qdrant server (dùng 'none' để disable)
- `--collection_name`: Tên collection trong Qdrant
- `--recreate_collection`: Tạo lại collection nếu đã tồn tại

**Output:**
- `{doc_id}.npz`: Vector embeddings (nếu --out_dir được chỉ định)
- Dữ liệu được lưu trực tiếp vào Qdrant database

## 🎯 Ví dụ chạy toàn bộ pipeline

```bash
# Bước 1: Trích xuất văn bản
python src/preprocessing/01_extract_text.py

# Bước 2: Phân đoạn văn bản
python src/preprocessing/02_chunk_text.py --auto_toc

# Bước 3: Tạo embeddings (yêu cầu Qdrant đang chạy)
python src/preprocessing/03_create_embeddings.py
```

## 🛠️ Cấu hình nâng cao

### Tối ưu hóa hiệu suất
- **GPU Memory**: Giảm `--batch_size` nếu gặp lỗi out of memory
- **CPU**: Tăng `--max_workers` cho parallel processing
- **Text quality**: Sử dụng `--auto_toc` để tự động loại bỏ mục lục

### Cấu hình embedding model
```bash
# Sử dụng model khác
python src/preprocessing/03_create_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2

# Normalize embeddings
python src/preprocessing/03_create_embeddings.py --normalize

# Chỉ CPU processing
python src/preprocessing/03_create_embeddings.py --device cpu --max_workers 4
```


## 📊 Monitoring và Logging

Tất cả scripts đều hỗ trợ logging chi tiết:
- `--log_level DEBUG`: Thông tin chi tiết nhất
- `--log_level INFO`: Thông tin cơ bản (mặc định)
- `--log_level WARNING`: Chỉ cảnh báo và lỗi
- `--log_level ERROR`: Chỉ lỗi

Logs bao gồm:
- Số lượng documents/chunks được xử lý
- Thời gian xử lý
- Thống kê hiệu suất
- Chi tiết lỗi và cách khắc phục