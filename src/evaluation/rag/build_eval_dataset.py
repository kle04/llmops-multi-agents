#!/usr/bin/env python
"""
Parse raw Q/A pairs, query Orchestrator Agent, build evaluation dataset JSON.
Usage:
    python src/evaluation/build_eval_dataset.py \
        --raw-file src/evaluation/raw_qa.txt \
        --output-file src/evaluation/eval_dataset.json \
        --orchestrator-url http://localhost:7010
"""

import argparse
import json
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import requests


QA_PATTERN = re.compile(
    r"(?P<label>Q|A)\s*(?P<index>\d+)\s*:\s*(?P<content>.+?)(?=(?:\nQ\d+:|\nA\d+:|\Z))",
    re.DOTALL,
)


def parse_raw_qa(raw_text: str) -> List[Dict[str, str]]:
    """Parse raw text into list of {"question":..., "ground_truth":...}."""
    matches = QA_PATTERN.findall(raw_text)
    buffer = {}
    dataset = []
    for label, idx, content in matches:
        content = content.strip()
        key = (label, idx)
        buffer[key] = content

    indices = sorted(
        {int(idx) for (_, idx) in buffer.keys()},
    )

    for idx in indices:
        question = buffer.get(("Q", str(idx)))
        answer = buffer.get(("A", str(idx)))
        if question and answer:
            dataset.append(
                {
                    "id": idx,
                    "question": question,
                    "ground_truth": answer,
                }
            )
    return dataset


def call_orchestrator(
    question: str,
    base_url: str,
    session_prefix: str = "eval",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Send question to orchestrator /chat endpoint."""
    payload = {
        "message": question,
        "user_id": f"{session_prefix}-user",
        "session_id": f"{session_prefix}-{uuid.uuid4().hex[:8]}",
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def build_dataset(
    qa_pairs: List[Dict[str, str]],
    orchestrator_url: str,
    sleep_secs: float = 0.5,
) -> List[Dict[str, Any]]:
    """Iterate QA list, query orchestrator, return evaluation entries."""
    results = []
    for item in qa_pairs:
        logging.info(f"Processing Q{item['id']}: {item['question'][:50]}...")
        response = call_orchestrator(item["question"], orchestrator_url)
        logging.info(f"Received response for Q{item['id']}")
        answer = response.get("response") or response.get("direct_response") or ""
        contexts = response.get("sources") or []
        results.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": answer.strip(),
                "contexts": contexts if isinstance(contexts, list) else [],
                "metadata": {
                    "selected_agent": response.get("selected_agent"),
                    "raw_response": response,
                },
            }
        )
        time.sleep(sleep_secs)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-file",
        default="src/evaluation/raw_qa.txt",
        help="Path to raw Q/A file (Q1:/A1: format).",
    )
    parser.add_argument(
        "--output-file",
        default="src/evaluation/eval_dataset.json",
        help="Output JSON file for evaluation dataset.",
    )
    parser.add_argument(
        "--orchestrator-url",
        default="http://localhost:7010",
        help="Orchestrator Agent base URL (without /chat).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Delay between requests to avoid throttling.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_file)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw QA file not found: {raw_path}")

    raw_text = raw_path.read_text(encoding="utf-8")
    qa_pairs = parse_raw_qa(raw_text)
    if not qa_pairs:
        raise ValueError("No valid Q/A pairs parsed. Check raw file format.")

    dataset = build_dataset(
        qa_pairs,
        orchestrator_url=args.orchestrator_url,
        sleep_secs=args.sleep,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] Saved evaluation dataset: {out_path} (samples={len(dataset)})")


if __name__ == "__main__":
    main()
