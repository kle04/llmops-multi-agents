#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

# ---------- Token counter ----------
_enc = None
def count_tokens(text: str) -> int:
    """Ưu tiên tiktoken; nếu không có, ước lượng 4 ký tự ~ 1 token."""
    global _enc
    try:
        import tiktoken
        if _enc is None:
            _enc = tiktoken.get_encoding("cl100k_base")
        return len(_enc.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))

# ---------- Helpers ----------
def parse_skip_pages(cell: str):
    """Chuyển '1-3,5,8-9' -> set({1,2,3,5,8,9})"""
    if not isinstance(cell, str) or not cell.strip():
        return set()
    out = set()
    for part in cell.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            out.update(range(min(a,b), max(a,b)+1))
        else:
            out.add(int(part))
    return out

def is_toc_page(text: str) -> bool:
    """Heuristic nhận diện 'MỤC LỤC': nhiều dòng kết thúc số, có chữ 'MỤC LỤC'."""
    txt = text.upper()
    if "MỤC LỤC" in txt or "MUC LUC" in txt:
        return True
    # nhiều dòng có '.... 12' hoặc '... 5' => giống mục lục
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    matched = 0
    for ln in lines:
        if re.search(r"\.{2,}\s*\d{1,4}$", ln):
            matched += 1
    return matched >= max(3, len(lines) // 3)

def is_heading(line: str) -> bool:
    """Nhận diện tiêu đề chương/mục đơn giản."""
    l = line.strip()
    if not l:
        return False
    # Các pattern thường gặp
    if re.match(r"^(CHƯƠNG|PHẦN|MỤC)\s+\d+[:\. ]", l, flags=re.IGNORECASE):
        return True
    if re.match(r"^\d+(\.\d+)*\s+", l):  # 1, 1.1, 2.3.4 ...
        return True
    # dòng viết HOA và khá ngắn
    if len(l) <= 70 and l == l.upper() and any(ch.isalpha() for ch in l):
        return True
    return False

def paragraphs_from_pages(pages):
    """Ghép các trang -> đoạn (ngắt bởi dòng trống), giữ tiêu đề."""
    text = "\n\n".join(pages)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras

def chunk_with_overlap(paras, max_toks, overlap, min_toks):
    """Ghép paragraph thành chunk theo số token, có overlap."""
    chunks = []
    buf, tok = [], 0
    for p in paras:
        ptok = count_tokens(p)
        # nếu 1 paragraph quá dài, cắt mạnh theo từ
        if ptok > max_toks * 1.5:
            words = p.split()
            step = max(50, max_toks - overlap)
            for i in range(0, len(words), step):
                seg = " ".join(words[i:i+max_toks])
                if count_tokens(seg) >= min_toks:
                    chunks.append(seg)
            continue

        if tok + ptok <= max_toks:
            buf.append(p); tok += ptok
        else:
            if tok >= min_toks:
                chunks.append("\n\n".join(buf))
                # overlap: lấy đuôi của buf để làm ngữ cảnh
                if overlap > 0 and buf:
                    tail, tail_tok = [], 0
                    for para in reversed(buf):
                        t = count_tokens(para)
                        if tail_tok + t > overlap:
                            break
                        tail.append(para); tail_tok += t
                    buf = list(reversed(tail))
                    tok = sum(count_tokens(x) for x in buf)
            # thêm paragraph hiện tại
            buf.append(p); tok += ptok

    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if c.strip()]

def attach_sections(paras):
    """
    Tìm heading gần nhất và gắn vào paragraph như nhãn 'section'.
    Trả về list[(section, paragraph_text)]
    """
    out = []
    current = ""
    for p in paras:
        lines = [ln for ln in p.splitlines() if ln.strip()]
        if lines and is_heading(lines[0]):
            current = lines[0].strip()
        out.append((current, p))
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/metadata/catalog.csv")
    ap.add_argument("--text_dir", default="data/processed/text")
    ap.add_argument("--out_dir", default="data/processed/chunks")
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--min_tokens", type=int, default=120)
    ap.add_argument("--auto_toc", action="store_true", help="Bật tự động nhận diện trang mục lục")
    args = ap.parse_args()

    catalog = pd.read_csv(args.catalog)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in catalog.iterrows():
        doc_id = str(row["doc_id"])
        text_path = Path(args.text_dir) / f"{doc_id}.pages.jsonl"
        if not text_path.exists():
            print(f"[WARN] Không thấy: {text_path}")
            continue

        # đọc các trang
        pages = []
        with open(text_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                pages.append((int(obj["page"]), obj.get("text", "")))

        # bỏ trang theo catalog (skip_pages)
        skip = parse_skip_pages(str(row.get("skip_pages", "")))
        if skip:
            pages = [(pg, txt) for (pg, txt) in pages if pg not in skip]

        # auto bỏ trang mục lục (tùy chọn)
        if args.auto_toc:
            pages = [(pg, txt) for (pg, txt) in pages if not is_toc_page(txt)]

        # tạo paragraphs và gắn section (heading)
        ordered_texts = [txt for _, txt in sorted(pages, key=lambda x: x[0])]
        paras = paragraphs_from_pages(ordered_texts)
        sec_paras = attach_sections(paras)

        # Nhưng chunker hoạt động theo paragraph text; ta giữ mapping section theo thứ tự
        paras_only = [p for (sec, p) in sec_paras]
        chunks = chunk_with_overlap(paras_only, args.max_tokens, args.overlap, args.min_tokens)

        # build payload cho từng chunk
        rows = []
        for i, ch in enumerate(chunks):
            # section gần nhất cho chunk = section của paragraph đầu tiên trong chunk
            first_para = ch.split("\n\n", 1)[0]
            # tìm lại section
            section = ""
            for (sec, p) in sec_paras:
                if p == first_para:
                    section = sec
                    break

            chunk_id = f"{doc_id}_{i:05d}"
            payload = {
                "id": chunk_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "year": row.get("year", ""),
                "language": row.get("language", "vi"),
                "audience": str(row.get("audience", "")),
                "grade_range": str(row.get("grade_range", "")),
                "topics": str(row.get("topics", "")),
                "section": section,
                "text": ch
            }
            rows.append(payload)

        out_file = out_dir / f"{doc_id}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[OK] {doc_id}: {len(rows)} chunks → {out_file}")

if __name__ == "__main__":
    main()
