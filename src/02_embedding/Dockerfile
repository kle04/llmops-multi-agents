FROM python:3.13.3-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY 03_embedding.py .
COPY 03b_upsert_qdrant.py .
COPY run_pipeline.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "/workspace/src/02_embedding/run_pipeline.py"]