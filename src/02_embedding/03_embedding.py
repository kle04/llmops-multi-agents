#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

def load_chunks(chunks_path: Path):
    """Đọc file JSONL của 1 tài liệu -> trả về (ids, texts, payloads)."""
    ids, texts, payloads = [], [], []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj["text"])
            payloads.append(obj)   # giữ lại để sau này upsert (nếu cần)
    return ids, texts, payloads

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", default="data/processed/chunks", help="Thư mục chứa các *.jsonl")
    ap.add_argument("--out_dir",    default="data/embeddings",      help="Nơi lưu .npz")
    ap.add_argument("--model",      default="BAAI/bge-m3",          help="Model embedding")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--normalize",  action="store_true", help="L2-normalize vectors")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo model 1 lần
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        print(f"[WARN] Không thấy file chunk trong {chunks_dir}. Hãy chạy bước 5 trước.")
        return

    for fp in files:
        doc_id = fp.stem  # <doc_id>.jsonl
        print(f"==> Embedding: {doc_id}")

        ids, texts, payloads = load_chunks(fp)
        vectors = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=False,  # mình sẽ normalize thủ công nếu cần
            show_progress_bar=True
        ).astype(np.float32)

        if args.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            vectors = vectors / norms

        # Lưu định dạng gọn nhẹ: ids + vectors
        out_path = out_dir / f"{doc_id}.npz"
        np.savez_compressed(out_path, ids=np.array(ids), vectors=vectors)
        print(f"[OK] Saved: {out_path}  shape={vectors.shape}")

if __name__ == "__main__":
    main()
