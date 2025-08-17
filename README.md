LLMOps - Multi Agents - A2A 


* **Folder Structure dự kiến**
project-root/
│
├── data/
│   ├── raw/                     # Dữ liệu gốc (PDF, DOCX, TXT)
│   │   ├── psychology/
│   │   │   ├── hs/              # Học sinh THCS, THPT
│   │   │   └── sv/              # Sinh viên
│   │   └── external/            # Tài liệu tham khảo thêm (tiếng Anh, WHO, UNESCO...)
│   │
│   ├── processed/               # Sau khi làm sạch (text chuẩn hóa, bỏ header/footer)
│   │   ├── chunks/              # Chunked text (JSON/CSV)
│   │   └── metadata/            # Metadata (title, author, nguồn…)
│   │
│   └── embeddings/              # Vector embedding (npy, parquet, hoặc JSON để load vào Qdrant)
│
├── src/                         # Code chính
│   ├── preprocessing/           # Script xử lý PDF -> text, làm sạch, chunking
│   ├── embedding/               # Script sinh embedding và push vào Qdrant
│   ├── agents/                  # Orchestrator + RAG Agent (FastAPI services)
│   ├── monitoring/              # Config Prometheus, Loki
│   └── utils/                   # Helper function, logging, config chung
│
├── configs/                     # File cấu hình
│   ├── qdrant.yaml              # Config vector DB
│   ├── postgres.yaml            # Config PostgreSQL
│   └── app_config.yaml          # Config app (keys, hyperparams)
│
├── notebooks/                   # Jupyter notebooks để thử nghiệm
│
├── deployments/                 # File hạ tầng (Terraform, K8s YAML, Helm chart, ArgoCD manifests)
│   ├── k8s/
│   ├── terraform/
│   └── ci-cd/
│
├── docs/                        # Tài liệu mô tả hệ thống, báo cáo tạm thời
│
└── README.md                    # Giới thiệu project
