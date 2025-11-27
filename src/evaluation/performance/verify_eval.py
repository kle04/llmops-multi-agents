import requests
import uuid
import time
import json
import re
import os

# Configuration
API_URL = "https://orchestrator.khanklee.id.vn/chat"
RAW_QA_FILE = "src/evaluation/rag/raw_qa.txt"

def parse_raw_qa(file_path: str):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Fixed regex: single backslash for whitespace in raw string
    questions = re.findall(r"Q\d+\s*:\s*(.+)", content)
    return [q.strip() for q in questions]

def evaluate_question(question: str, category: str):
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    
    payload = {
        "message": question,
        "user_id": user_id,
        "session_id": session_id
    }
    
    start_time = time.time()
    try:
        print(f"Sending ({category}): {question[:30]}...")
        response = requests.post(API_URL, json=payload, timeout=30)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"  Success: {latency:.2f}s | Agent: {data.get('selected_agent')}")
            return True
        else:
            print(f"  Error {response.status_code}: {latency:.2f}s")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False

def main():
    # Test 1 RAG question
    questions = parse_raw_qa(RAW_QA_FILE)
    print(f"Found {len(questions)} questions.")
    if questions:
        evaluate_question(questions[0], "RAG")
    else:
        print("No RAG questions found to test.")

    # Test 1 General question
    evaluate_question("1 + 1 bằng mấy?", "General")

if __name__ == "__main__":
    main()
