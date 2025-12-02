# Main - FastAPI app cho orchestrator agent

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
from agent import OrchestratorAgent
from config import Config
import uvicorn
from redis_memory import RedisManager, ChatHistoryStore, LangChainHistoryStore
from postgres_memory import PostgresManager, PostgresChatHistoryStore
from langchain_core.messages import BaseMessage

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    selected_agent: Optional[str] = None
    response: str
    sources: Optional[List[str]] = None
    error: Optional[str] = None

agent = OrchestratorAgent()
redis_manager = RedisManager()
postgres_manager = PostgresManager()
chat_store: Optional[ChatHistoryStore] = None
langchain_store: Optional[LangChainHistoryStore] = None
postgres_store: Optional[PostgresChatHistoryStore] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.initialize()
    await redis_manager.initialize()
    await postgres_manager.initialize()
    global chat_store
    chat_store = ChatHistoryStore(redis_manager)
    global langchain_store
    langchain_store = LangChainHistoryStore(redis_manager)
    global postgres_store
    postgres_store = PostgresChatHistoryStore(postgres_manager)
    yield


app = FastAPI(
    title="Orchestrator Agent",
    description="Orchestrator Agent for managing and coordinating tasks.",
    version="1.0.0",
    lifespan=lifespan
)





@app.get("/health")
async def health():
    # include redis health if available
    try:    
        redis_health = await redis_manager.health_check() if redis_manager else {"connected": False}
        postgres_health = await postgres_manager.health_check() if postgres_manager else {"connected": False}
        agent_health = await agent.health_check() if agent else {"status": "unhealthy"}
        return {"status": "healthy", "redis": redis_health, "postgres": postgres_health, "agent": agent_health}
    except Exception as e:
        return {"status": "unhealthy", "redis": {"connected": False}, "agent": {"status": "unhealthy", "error": str(e)}}



@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Ensure minimal identifiers
    user_id = req.user_id or "anonymous"
    session_id = req.session_id or "default"

    # Append user message to chat history
    if chat_store and redis_manager.is_ready():
        await chat_store.append_message(user_id, session_id, role="user", content=req.message)
    
    # Append to Postgres (Long-term)
    if postgres_store:
        await postgres_store.append_message(user_id, session_id, role="user", content=req.message)
    if langchain_store and redis_manager.is_ready():
        await langchain_store.append_turn(user_id, session_id, turn_type="human", content=req.message)
        context = await langchain_store.get_history_context(user_id, session_id)

    result = await agent.process_message(req.message, context)
    # Normalize output shape
    response_text = result.get("response")

    # Append assistant message (with optional sources)
    if chat_store and redis_manager.is_ready():
        sources = result.get("sources") or []
        await chat_store.append_message(
            user_id,
            session_id,
            role="assistant",
            content=response_text or "",
            agent_used=result.get("selected_agent") or "Orchestrator",
            source=sources if isinstance(sources, list) else [],
        )
    
    # Append assistant message to Postgres (Long-term)
    if postgres_store:
        await postgres_store.append_message(
            user_id,
            session_id,
            role="assistant",
            content=response_text or "",
            agent_used=result.get("selected_agent") or "Orchestrator",
            source=sources if isinstance(sources, list) else [],
        )
    if langchain_store and redis_manager.is_ready():
        await langchain_store.append_turn(user_id, session_id, turn_type="ai", content=response_text)

    return {
        "selected_agent": result.get("selected_agent"),
        "response": response_text,
        "sources": result.get("sources"),
        "error": result.get("error")
    }


@app.get("/history/{user_id}/{session_id}")
async def get_postgres_history(user_id: str, session_id: str):
    """Get full conversation history from Postgres (long-term storage)."""
    if not postgres_store or not postgres_manager.is_ready():
        return {"error": "Postgres not available", "messages": [], "created_at": None, "last_updated": None}
    chat = await postgres_store.load_session_history(user_id, session_id)
    return {
        "messages": chat.messages,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "last_updated": chat.last_updated.isoformat() if chat.last_updated else None,
    }

@app.get("/history/redis/{user_id}/{session_id}")
async def get_redis_history(user_id: str, session_id: str, limit: Optional[int] = None):
    """Get conversation history from Redis (short-term storage).
    
    Args:
        limit: Optional limit on number of recent turns to return (default: all)
    """
    if not langchain_store or not redis_manager.is_ready():
        return {"error": "Redis not available", "messages": []}
    
    if limit and limit > 0:
        messages = await langchain_store.get_history_context(user_id, session_id, limit=limit)
    else:
        messages = await langchain_store.get(user_id, session_id)
    
    return {"messages": messages}

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """List all session IDs for a given user."""
    if not postgres_store or not postgres_manager.is_ready():
        return {"error": "Postgres not available", "sessions": []}
    sessions = await postgres_store.list_sessions(user_id)
    return {"sessions": sessions}

def main():
    uvicorn.run(
        app,
        host=Config.ORCHESTRATOR_AGENT_HOST,
        port=Config.ORCHESTRATOR_AGENT_PORT,
    )

if __name__ == "__main__":
    main()
