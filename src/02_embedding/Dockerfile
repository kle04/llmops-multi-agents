FROM python:3.13.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace

# Cài đặt lib hệ thống tối thiểu (tăng tốc pip, tránh lỗi build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
