# Deploy Embedding Service

FastAPI service for Vietnamese text embedding generation using CPU-only PyTorch.

## 🚀 Features

- **Model:** AITeamVN/Vietnamese_Embedding_v2
- **CPU-only PyTorch:** Optimized for production deployment
- **FastAPI:** High-performance async API
- **Multi-stage Docker:** Optimized image size
- **Security:** Non-root user, minimal packages

## 📋 API Endpoints

### Health Check
```bash
GET /health
# Response: {"status": "healthy"}
```

### Generate Embedding
```bash
POST /embed
Content-Type: application/json

{
  "text": "Văn bản tiếng Việt cần tạo embedding"
}

# Response:
{
  "embedding": [0.1, -0.2, 0.3, ...]  # 768-dimensional vector
}
```

## 🔧 Build & Run

### Build Image
```bash
chmod +x build.sh
./build.sh
```

### Run Container
```bash
# Basic run
docker run -p 8000:8000 khanhle04/deploy-embedding:latest

# With custom port
docker run -p 5000:8000 -e PORT=8000 khanhle04/deploy-embedding:latest

# Background run
docker run -d -p 8000:8000 --name embedding-api khanhle04/deploy-embedding:latest
```

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Embedding
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Xin chào thế giới"}'
```

### Load Test
```bash
# Install httpie first: pip install httpie
echo '{"text": "Test embedding generation"}' | http POST localhost:8000/embed
```

## ⚙️ Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `HF_HOME`: Hugging Face cache directory
- `PYTHONPATH`: Python module path

### Model Configuration
Edit `app.py` to change model:
```python
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Smaller model
# or
MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"  # Vietnamese model
```

## 🐳 Docker Details

### Image Size Optimization
- Multi-stage build
- CPU-only PyTorch
- Minimal base image
- Stripped binaries
- No dev dependencies

### Security Features
- Non-root user (`appuser`)
- Minimal system packages
- Read-only filesystem ready
- No unnecessary capabilities

## 📊 Performance

### Benchmarks (estimated)
- **Cold start:** ~3-5 seconds (model loading)
- **Inference:** ~50-100ms per text
- **Memory:** ~1-2GB RAM
- **CPU:** Optimized for multi-core

### Scaling
```yaml
# docker-compose.yml
services:
  embedding-api:
    image: khanhle04/deploy-embedding:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
    ports:
      - "8000-8002:8000"
```

## 🔍 Monitoring

### Health Monitoring
```bash
# Container health
docker ps
docker logs embedding-api

# API health
curl -f http://localhost:8000/health || echo "Service down"
```

### Metrics (with Prometheus)
```python
# Add to app.py
from prometheus_client import Counter, Histogram, generate_latest

embedding_requests = Counter('embedding_requests_total', 'Total embedding requests')
embedding_duration = Histogram('embedding_duration_seconds', 'Embedding generation duration')
```

## 🚀 Production Deployment

### Docker Compose
```yaml
version: '3.8'
services:
  embedding-api:
    image: khanhle04/deploy-embedding:latest
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embedding-api
  template:
    metadata:
      labels:
        app: embedding-api
    spec:
      containers:
      - name: embedding-api
        image: khanhle04/deploy-embedding:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```
