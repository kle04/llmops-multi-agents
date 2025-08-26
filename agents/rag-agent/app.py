"""
RAG Agent
"""

import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Thiết lập biến môi trường
RAG_AGENT_URL = os.getenv("RAG_AGENT_URL", "http://localhost:7005")

# Tạo FastAPI app
app = FastAPI(
    title="RAG Agent",
    description="Agent xử lý query từ user",
    version="1.0.0"
)

# === A2A Protocol Models ===

class MessageType(str, Enum):
    """Loại message trong A2A Protocol"""


# === Endpoints ===
@app.get("/")
async def root():
    """Endpoint gốc"""
    return {
        "service": "RAG Agent",
        "version": "1.0.0",
        "status": "running"
    }

# Endpoint health check
@app.get("/health")
async def health():
    """Kiểm tra sức khỏe service"""
    return {
        "status": "healthy",
        "service": "rag-agent"
    }
