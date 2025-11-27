import re
from pathlib import Path

file_path = "src/evaluation/performance/raw_qa.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

questions = re.findall(r"Q\d+\s*:\s*(.+)", content)
print(f"Found {len(questions)} questions.")
for i, q in enumerate(questions):
    print(f"{i+1}: {q}")
