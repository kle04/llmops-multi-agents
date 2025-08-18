#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def extract_pages(pdf_path: Path):
    """Trả về list[str], mỗi phần tử là text của một trang."""
    # Ưu tiên PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path.as_posix())
        pages = [p.get_text("text") for p in doc]
        doc.close()
        return pages
    except Exception:
        # Fallback: PyPDF2
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path.as_posix())
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages

def normalize_whitespace(t: str) -> str:
    # nối các dòng gãy (không phải khoảng trống đôi)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # thu gọn nhiều khoảng trắng
    t = re.sub(r"[ \t]+", " ", t)
    # thay 3+ dòng trống bằng 2 dòng trống (giữ đoạn)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def strip_headers_footers(pages):
    """
    Loại bỏ:
    - dòng chỉ có số trang,
    - header lặp lại ở đầu trang (nếu xuất hiện >50% số trang).
    """
    cleaned = []
    first_line_count = {}
    for pg in pages:
        lines = [ln.strip() for ln in pg.splitlines() if ln.strip()]
        # bỏ các dòng chỉ có số
        lines = [ln for ln in lines if not re.fullmatch(r"\d{1,4}", ln)]
        if not lines:
            cleaned.append("")
            continue
        first_line_count[lines[0]] = first_line_count.get(lines[0], 0) + 1
        cleaned.append("\n".join(lines))

    # xác định header nếu lặp >50% số trang
    threshold = len(pages) * 0.5
    common_headers = {h for h, c in first_line_count.items() if c > threshold}

    final_pages = []
    for pg in cleaned:
        lines = pg.splitlines()
        if lines and lines[0] in common_headers:
            lines = lines[1:]
        final_pages.append("\n".join(lines))
    return final_pages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/metadata/catalog.csv", help="Đường dẫn catalog CSV")
    ap.add_argument("--raw_dir", default="data/raw", help="Thư mục chứa PDF gốc")
    ap.add_argument("--out_dir", default="data/processed/text", help="Thư mục ghi JSONL theo trang")
    args = ap.parse_args()

    catalog = Path(args.catalog)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(catalog)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        doc_id = str(row["doc_id"])
        filename = str(row["filename"])
        pdf_path = raw_dir / filename
        if not pdf_path.exists():
            print(f"[WARN] Không tìm thấy file: {pdf_path}")
            continue

        # 1) extract
        pages = extract_pages(pdf_path)

        # 2) clean nhẹ
        pages = [normalize_whitespace(p) for p in pages]
        pages = strip_headers_footers(pages)

        # 3) ghi JSONL: data/processed/text/<doc_id>.pages.jsonl
        out_path = out_dir / f"{doc_id}.pages.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for i, txt in enumerate(pages, start=1):
                f.write(json.dumps({"doc_id": doc_id, "page": i, "text": txt}, ensure_ascii=False) + "\n")

        # 4) ghi thêm file gộp (để bạn mở đọc nhanh)
        merged_path = out_dir / f"{doc_id}.merged.txt"
        merged_text = "\n\n".join(pages)
        merged_path.write_text(merged_text, encoding="utf-8")

        print(f"[OK] {doc_id}: {len(pages)} trang → {out_path.name}, {merged_path.name}")

if __name__ == "__main__":
    main()
