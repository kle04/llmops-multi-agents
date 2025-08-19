#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upsert embeddings + payloads vào Qdrant (ID chuỗi UUID v5, batch-friendly).

YÊU CẦU DỮ LIỆU:
- data/embeddings/<doc_id>.npz
    - chứa: ids[] (list[str]), vectors (np.ndarray: N x D)
- data/processed/chunks/<doc_id>.jsonl
    - mỗi dòng là 1 payload JSON, có key "id" khớp với ids[]

VÍ DỤ CHẠY (local):
python src/02_embedding/03b_upsert_qdrant.py \
  --emb_dir data/embeddings \
  --chunks_dir data/processed/chunks \
  --qdrant_url http://localhost:6333 \
  --collection mental_health_vi \
  --distance cosine \
  --batch_size 256

VÍ DỤ CHẠY (docker compose):
cd src
docker compose build upsert
docker compose run --rm upsert
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except Exception:
    print(
        "[ERROR] Thiếu thư viện qdrant-client. Cài đặt bằng:\n"
        "  pip install qdrant-client\n",
        file=sys.stderr,
    )
    raise


# --------------------------
# Helpers
# --------------------------

def to_uuid5_str(s: str) -> str:
    """
    Qdrant nhận int hoặc str. Ta dùng UUID v5 (deterministic) dưới dạng string.
    """
    try:
        # Nếu đã là UUID hợp lệ -> trả nguyên dạng chuỗi chuẩn
        return str(uuid.UUID(s))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def load_embeddings(npz_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Đọc file .npz -> (ids:list[str], vectors:np.ndarray[float32])
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"].tolist()
    vectors = data["vectors"].astype(np.float32)
    return ids, vectors


def load_payloads(chunks_path: Path) -> Dict[str, dict]:
    """
    Đọc file chunks .jsonl -> dict[id] = payload
    """
    out: Dict[str, dict] = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("id")
            if cid is not None:
                out[cid] = obj
    return out


def ensure_collection(
    client: QdrantClient,
    name: str,
    dim: int,
    distance: str = "cosine",
    recreate: bool = False,
):
    """
    Tạo collection nếu chưa có; nếu recreate=True -> xóa & tạo lại.
    Dùng collection_exists + create_collection (tránh API deprecated).
    """
    dist_map = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
        "euclidean": Distance.EUCLID,
        "l2": Distance.EUCLID,
    }
    dist = dist_map.get(distance.lower(), Distance.COSINE)

    if recreate:
        try:
            client.delete_collection(name)
            print(f"[INFO] Đã xóa collection cũ: {name}")
        except Exception:
            # Nếu chưa tồn tại thì thôi
            pass

    # Qdrant client mới có collection_exists; fallback sang get_collection nếu cần
    try:
        exists = client.collection_exists(name)  # type: ignore[attr-defined]
    except AttributeError:
        try:
            client.get_collection(name)
            exists = True
        except Exception:
            exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=dist),
        )
        print(f"[INFO] Đã tạo collection: {name} (dim={dim}, distance={distance})")
    else:
        print(f"[INFO] Collection đã tồn tại: {name}")


def batched(iterable, n: int):
    """Yield list chunks of size <= n."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", default="data/embeddings", help="Thư mục chứa *.npz")
    ap.add_argument("--chunks_dir", default="data/processed/chunks", help="Thư mục chứa *.jsonl")
    ap.add_argument("--qdrant_url", default="http://localhost:6333")
    ap.add_argument("--qdrant_key", default=None)
    ap.add_argument("--collection", default="mental_health_vi")
    ap.add_argument("--distance", default="cosine", help="cosine | dot | euclid")
    ap.add_argument("--recreate", action="store_true", help="Xóa & tạo lại collection")
    ap.add_argument("--batch_size", type=int, default=256, help="Số điểm upsert mỗi batch")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    chunks_dir = Path(args.chunks_dir)

    files = sorted(emb_dir.glob("*.npz"))
    if not files:
        print(f"[WARN] Không tìm thấy *.npz trong {emb_dir}")
        return

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_key)

    collection_ready = False
    expected_dim = None
    total_points = 0

    for npz_path in files:
        doc_id = npz_path.stem
        ids, vectors = load_embeddings(npz_path)

        # Kiểm tra dimension thống nhất
        dim = vectors.shape[1]
        if expected_dim is None:
            expected_dim = dim
        elif expected_dim != dim:
            raise ValueError(
                f"Dimension không nhất quán: {doc_id} có dim={dim} khác expected_dim={expected_dim}"
            )

        # Tạo collection một lần (hoặc recreate nếu yêu cầu)
        if not collection_ready:
            ensure_collection(client, args.collection, dim, args.distance, args.recreate)
            collection_ready = True

        # Tải payloads
        chunks_path = chunks_dir / f"{doc_id}.jsonl"
        payloads = load_payloads(chunks_path)
        if not payloads:
            print(f"[WARN] Không tìm thấy payloads cho {doc_id}: {chunks_path}")

        # Build points
        pts: List[PointStruct] = []
        missing_payload = 0
        for i, cid in enumerate(ids):
            qid = to_uuid5_str(cid)  # <- ép thành chuỗi UUID hợp lệ
            payload = payloads.get(cid)
            if payload is None:
                missing_payload += 1
                payload = {"id": cid, "doc_id": doc_id}
            pts.append(
                PointStruct(
                    id=qid,
                    vector=vectors[i].tolist(),
                    payload=payload,
                )
            )

        if missing_payload:
            print(f"[WARN] {doc_id}: thiếu {missing_payload}/{len(ids)} payload (sẽ upsert với payload tối thiểu)")

        # Upsert theo batch
        sent = 0
        for batch in batched(pts, args.batch_size):
            client.upsert(collection_name=args.collection, points=batch, wait=True)
            sent += len(batch)

        total_points += sent
        print(f"[OK] Upserted {sent} points for {doc_id}")

    print(f"[DONE] Tổng số điểm đã upsert: {total_points} vào collection '{args.collection}'")


if __name__ == "__main__":
    main()
