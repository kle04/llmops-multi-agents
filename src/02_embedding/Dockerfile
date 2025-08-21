# Multi-stage build for embedding pipeline
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user --no-warn-script-location \
    -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY 03_embedding.py .
COPY 03b_upsert_qdrant.py .
COPY run_pipeline.py .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "/workspace/src/02_embedding/run_pipeline.py"]