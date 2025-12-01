# Performance Evaluation Module

## Abstract

This module provides a performance testing framework for the Mental Health Support System's Orchestrator Agent. It specifically evaluates and compares the end-to-end latency and routing accuracy between **Retrieval-Augmented Generation (RAG)** queries (domain-specific mental health questions) and **Standard Generation** queries (general chitchat/knowledge). By automating the testing process against the live API, this module delivers critical insights into the system's responsiveness and the efficiency of the Agent-to-Agent (A2A) handoff protocol.

## 1. Evaluation Methodology

The evaluation logic is designed to test two distinct pathways in the Orchestrator's decision tree:

### 1.1. Test Scenarios
1.  **RAG Scenario (Mental Health)**:
    *   **Input Source**: `raw_qa.txt` (A curated list of mental health questions).
    *   **Expected Behavior**: The Orchestrator should identify the intent, route the request to the `RAG Agent`, wait for the retrieval and generation process, and return the answer.
    *   **Metric**: Measures the full round-trip time (Orchestrator -> RAG Agent -> Qdrant -> LLM -> Orchestrator).

2.  **No-RAG Scenario (General Chat)**:
    *   **Input Source**: A predefined list of general knowledge and chitchat questions (e.g., "1 + 1 = ?", "Hello").
    *   **Expected Behavior**: The Orchestrator should handle the request directly using its internal LLM without invoking external agents.
    *   **Metric**: Measures the direct response latency.

### 1.2. Metrics Collected
*   **Latency (Seconds)**: End-to-end processing time.
*   **Routing Accuracy**: Verifies if `selected_agent` correctly matches the question type (`"RAG Agent"` for mental health, `null` for general).
*   **Status**: HTTP response codes (200 OK vs errors).

## 2. File Structure

*   **`response_time_evaluation.ipynb`**: The core analysis notebook. It runs the batch tests, collects data, calculates statistics (Mean, Median, P95), and generates visualization plots (Histograms, Boxplots).
*   **`verify_eval.py`**: A lightweight, command-line utility for quick "smoke testing" of the evaluation logic without running the full notebook.
*   **`raw_qa.txt`**: The test dataset containing raw questions formatted as `Q<number>: <question>`.

## 3. Usage Instructions

### 3.1. Running the Full Evaluation
1.  Ensure the Orchestrator API is running (default: `https://orchestrator.khanklee.id.vn/chat` or local equivalent).
2.  Open `response_time_evaluation.ipynb` in Jupyter or VS Code.
3.  Run all cells to execute the test suite.
4.  Review the generated **Latency Statistics** table and **Distribution Plots**.

### 3.2. Running a Quick Verification
To verify that the evaluation script can connect to the API and parse the data correctly:

```bash
uv run python src/evaluation/performance/verify_eval.py
```

## 4. Sample Results

*Typical performance characteristics observed:*

| Category | Mean Latency | Logic Flow |
| :--- | :--- | :--- |
| **No RAG (General)** | ~1.5s - 2.5s | Direct LLM generation. Fast. |
| **RAG (Mental Health)** | ~5.0s - 8.0s | Includes Intent Classification + A2A Network Call + Vector Search + Context Aggregation + LLM Generation. |

## 5. Code Principles
This module adheres to the project's clean code standards:
*   **Separation of Concerns**: Data parsing, API interaction, and Analysis are distinct steps.
*   **Readability**: Clear function names (`evaluate_question`, `parse_raw_qa`).
*   **Robustness**: Includes error handling for network timeouts and API failures.

