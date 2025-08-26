#!/usr/bin/env python3
"""
Orchestrator Agent
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
    title="Orchestrator Agent",
    description="Agent điều phối các request tới RAG Agent",
    version="1.0.0"
)

# === A2A Protocol Models ===

class MessageType(str, Enum):
    """Loại message trong A2A Protocol"""
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"

class A2AMessage(BaseModel):
    """Standard A2A Protocol Message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    message_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

# === User API Models ===

class UserQuery(BaseModel):
    """Query từ user"""
    query: str

class UserResponse(BaseModel):
    """Response trả về user"""
    query: str
    response: str
    agent_used: str = "rag-agent"
    message_id: Optional[str] = None

# Function gọi RAG Agent với A2A Protocol
async def call_rag_agent_a2a(query: str) -> tuple[str, str]:
    """Gọi tới RAG Agent sử dụng A2A Protocol"""
    
    # Tạo A2A message
    a2a_message = A2AMessage(
        sender="orchestrator-agent",
        receiver="rag-agent",
        message_type=MessageType.REQUEST,
        payload={
            "query": query,
            "context_limit": 5,
            "score_threshold": 0.7
        },
        metadata={
            "user_session": "default"
        }
    )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Gọi tới RAG Agent với A2A Protocol
            response = await client.post(
                f"{RAG_AGENT_URL}/a2a/process",
                json=a2a_message.model_dump(mode='json')
            )
            
            # Kiểm tra response status
            if response.status_code == 200:
                # Parse A2A response
                response_data = response.json()
                response_message = A2AMessage(**response_data)
                
                if response_message.message_type == MessageType.RESPONSE:
                    return (
                        response_message.payload.get("response", "Không có response từ RAG Agent"),
                        response_message.message_id
                    )
                elif response_message.message_type == MessageType.ERROR:
                    return (
                        f"RAG Agent error: {response_message.payload.get('error', 'Unknown error')}",
                        response_message.message_id
                    )
            else:
                return (f"RAG Agent trả về lỗi: {response.status_code}", a2a_message.message_id)
    # Catch lỗi kết nối
    except httpx.ConnectError:
        return ("Không thể kết nối tới RAG Agent", a2a_message.message_id)
    # Catch lỗi timeout
    except httpx.TimeoutException:
        return ("RAG Agent không phản hồi (timeout)", a2a_message.message_id)
    # Catch lỗi khác
    except Exception as e:
        return (f"Lỗi khi gọi RAG Agent: {str(e)}", a2a_message.message_id)

# Function gọi RAG Agent (fallback đơn giản)
async def call_rag_agent_simple(query: str) -> str:
    """Fallback: Gọi RAG Agent với simple protocol"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_AGENT_URL}/process",
                json={"query": query}
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "Không có response từ RAG Agent")
            else:
                return f"RAG Agent trả về lỗi: {response.status_code}"
                
    except Exception as e:
        return f"Lỗi khi gọi RAG Agent: {str(e)}"

# === Các Endpoint FastAPI ===

# Endpoint gốc
@app.get("/")
async def root():
    """Endpoint gốc"""
    return {
        "service": "Orchestrator Agent",
        "version": "1.0.0",
        "status": "running"
    }

# Endpoint health check
@app.get("/health")
async def health():
    """Kiểm tra sức khỏe service"""
    return {
        "status": "healthy",
        "service": "orchestrator-agent"
    }

# Endpoint xử lý query với A2A Protocol
@app.post("/query", response_model=UserResponse)
async def process_query(user_query: UserQuery):
    """Xử lý query từ user sử dụng A2A Protocol"""
    
    # Thử gọi RAG Agent với A2A Protocol trước
    try:
        rag_response, message_id = await call_rag_agent_a2a(user_query.query)
        
        return UserResponse(
            query=user_query.query,
            response=rag_response,
            agent_used="rag-agent-a2a",
            message_id=message_id
        )
    
    except Exception as e:
        # Fallback: Thử với simple protocol
        try:
            rag_response = await call_rag_agent_simple(user_query.query)
            
            return UserResponse(
                query=user_query.query,
                response=rag_response,
                agent_used="rag-agent-simple",
                message_id=None
            )
        
        except Exception as e2:
            # Final fallback: Response từ orchestrator
            return UserResponse(
                query=user_query.query,
                response=f"Không thể kết nối tới RAG Agent. Lỗi: {str(e2)}",
                agent_used="orchestrator-fallback",
                message_id=None
            )

# Chạy server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
