FROM python:3.13.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace

# Cài đặt phụ thuộc
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Mặc định vào shell; lệnh thật sẽ ở docker-compose
CMD ["bash"]
