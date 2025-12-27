# Main - FastAPI app cho orchestrator agent

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
from agent import OrchestratorAgent
from config import Config
import uvicorn
from redis_memory import RedisManager, ChatHistoryStore, LangChainHistoryStore
from postgres_memory import PostgresManager, PostgresChatHistoryStore
from langchain_core.messages import BaseMessage
from fastapi.middleware.cors import CORSMiddleware
# Import ingestion service
import shutil
import tempfile
import os
from pathlib import Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import Depends, status, HTTPException, Response, Cookie
from jose import JWTError, jwt
from auth_utils import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
import uuid
import logging
import asyncio

try:
    from ingestion_service import IngestionService
    HAS_INGESTION = True
except ImportError as e:
    print(f"⚠️  Could not import IngestionService: {e}")
    HAS_INGESTION = False

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    password_hash: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserRegister(BaseModel):
    username: str
    email: Optional[str] = None
    password: str

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
ingestion_service: Optional[IngestionService] = None

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
    
    # Initialize ingestion service
    global ingestion_service
    if HAS_INGESTION:
        try:
            print("🚀 Initializing Ingestion Service...")
            ingestion_service = IngestionService()
        except Exception as e:
            print(f"❌ Failed to initialize ingestion service: {e}")
            ingestion_service = None
            
    yield


app = FastAPI(
    title="Orchestrator Agent",
    description="Orchestrator Agent for managing and coordinating tasks.",
    version="2.0.1",
    lifespan=lifespan
)

# Debug: Log allowed origins
print(f"🌍 CORS Allowed Origins: {Config.CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug Logging Middleware
from fastapi import Request
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log Request
    body = await request.body()
    logger.info(f"➡️  Incoming {request.method} {request.url}")
    # logger.debug(f"    Body: {body.decode('utf-8')[:500]}...") # Uncomment for body logging

    response = await call_next(request)
    
    # Log Response
    process_time = time.time() - start_time
    logger.info(f"⬅️  Response {response.status_code} (took {process_time:.4f}s)")
    
    return response

@app.get("/health")
async def health():
    # include redis health if available
    try:    
        redis_health = await redis_manager.health_check() if redis_manager else {"connected": False}
        postgres_health = await postgres_manager.health_check() if postgres_manager else {"connected": False}
        agent_health = await agent.health_check() if agent else {"status": "unhealthy"}
        ingestion_health = ingestion_service.health_check() if ingestion_service else {"status": "unavailable"}
        return {
            "status": "healthy",
            "redis": redis_health,
            "postgres": postgres_health,
            "agent": agent_health,
            "ingestion": ingestion_health
        }
    except Exception as e:
        return {"status": "unhealthy", "redis": {"connected": False}, "agent": {"status": "unhealthy", "error": str(e)}}



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    access_token: Optional[str] = Cookie(None)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Prioritize cookie, fallback to header
    final_token = access_token or token
    
    if not final_token:
        raise credentials_exception

    try:
        payload = jwt.decode(final_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await postgres_manager.get_user_by_username(token_data.username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/register", response_model=bool)
async def register(user: UserRegister):
    existing_user = await postgres_manager.get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    user_id = str(uuid.uuid4())
    result = await postgres_manager.create_user(user_id, user.username, user.email, hashed_password)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create user")
    return True

@app.post("/token", response_model=Token)
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    user = await postgres_manager.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    
    # Set HttpOnly Cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False, # Set to True in Production (HTTPS)
        samesite="lax",
        max_age=1800 # 30 minutes
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

@app.delete("/users/me")
async def delete_user_me(current_user: dict = Depends(get_current_user)):
    """Permanently delete the current user account and all data."""
    if not postgres_manager.is_ready():
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    success = await postgres_manager.delete_user(current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="User not found or already deleted")
        
    return {"status": "success", "message": f"User {current_user['username']} deleted"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    # Use authenticated user_id
    user_id = current_user["user_id"]
    # Allow session_id from request or default
    session_id = req.session_id or str(uuid.uuid4())
    
    # Verify session ownership if session_id is provided and exists
    is_new_session = True
    if req.session_id and postgres_manager.is_ready():
        existing_session = await postgres_manager.get_session(session_id)
        if existing_session:
            is_new_session = False
            if existing_session["user_id"] != user_id:
                logger.warning(f"Session hijacking attempt: User {user_id} tried to access session {session_id} of user {existing_session['user_id']}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to access this session"
                )
    

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
        
        # Generate Title ONLY if it's a new session
        if is_new_session and postgres_manager.is_ready():
             # Trigger title generation in background (fire and forget)
             asyncio.create_task(generate_and_save_title(session_id, req.message, response_text))

    if langchain_store and redis_manager.is_ready():
        await langchain_store.append_turn(user_id, session_id, turn_type="ai", content=response_text)

    return {
        "selected_agent": result.get("selected_agent"),
        "response": response_text,
        "sources": result.get("sources"),
        "error": result.get("error")
    }

async def generate_and_save_title(session_id: str, user_msg: str, agent_resp: str):
    """Helper to generate and save title."""
    try:
        title = await agent.generate_title(user_msg, agent_resp)
        if title and postgres_manager.is_ready():
            await postgres_manager.update_session_title(session_id, title)
            logger.info(f"Updated title for session {session_id}: {title}")
    except Exception as e:
        logger.error(f"Background title generation failed: {e}")


@app.get("/history/{user_id}/{session_id}")
async def get_postgres_history(user_id: str, session_id: str, current_user: dict = Depends(get_current_user)):
    """Get full conversation history from Postgres (long-term storage)."""
    # Verify user access (optional: strict check if user_id matches current_user)
    if user_id != current_user["user_id"]:
         raise HTTPException(status_code=403, detail="Not authorized to access this history")

    if not postgres_store or not postgres_manager.is_ready():
        return {"error": "Postgres not available", "messages": [], "created_at": None, "last_updated": None}
    chat = await postgres_store.load_session_history(user_id, session_id)
    return {
        "messages": chat.messages,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "last_updated": chat.last_updated.isoformat() if chat.last_updated else None,
    }

@app.get("/history/redis/{user_id}/{session_id}")
async def get_redis_history(user_id: str, session_id: str, limit: Optional[int] = None, current_user: dict = Depends(get_current_user)):
    """Get conversation history from Redis (short-term storage).
    
    Args:
        limit: Optional limit on number of recent turns to return (default: all)
    """
    if user_id != current_user["user_id"]:
         raise HTTPException(status_code=403, detail="Not authorized to access this history")

    if not langchain_store or not redis_manager.is_ready():
        return {"error": "Redis not available", "messages": []}
    
    if limit and limit > 0:
        messages = await langchain_store.get_history_context(user_id, session_id, limit=limit)
    else:
        messages = await langchain_store.get(user_id, session_id)
    
    return {"messages": messages}

@app.get("/sessions/{user_id}")
async def get_user_sessions(user_id: str, current_user: dict = Depends(get_current_user)):
    """List all session IDs for a given user."""
    if user_id != current_user["user_id"]:
         raise HTTPException(status_code=403, detail="Not authorized to access these sessions")

    if not postgres_store or not postgres_manager.is_ready():
        return {"error": "Postgres not available", "sessions": []}
    sessions = await postgres_store.list_sessions(user_id)
    return {"sessions": sessions}

@app.delete("/history/{user_id}/{session_id}")
async def delete_chat_session(user_id: str, session_id: str, current_user: dict = Depends(get_current_user)):
    """Delete all history for a specific session (Redis + Postgres)."""
    if user_id != current_user["user_id"]:
         raise HTTPException(status_code=403, detail="Not authorized to delete this session")
    
    # 1. Clear from Redis (Short-term)
    if chat_store and redis_manager.is_ready():
        await chat_store.clear(user_id, session_id)
    
    if langchain_store and redis_manager.is_ready():
        await langchain_store.clear(user_id, session_id)
        
    # 2. Clear from Postgres (Long-term)
    pg_deleted = False
    if postgres_store and postgres_manager.is_ready():
        pg_deleted = await postgres_store.delete_session(user_id, session_id)
        
    return {
        "status": "success", 
        "message": f"Session {session_id} deleted for user {user_id}",
        "postgres_deleted": pg_deleted
    }

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Ingest a document (PDF, MD, TXT) into the knowledge base."""
    if current_user["username"] != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Ingestion service is not available")
    
    filename = file.filename
    ext = Path(filename).suffix.lower()
    
    if ext not in ['.pdf', '.md', '.txt']:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: .pdf, .md, .txt")
    
    # Create temp file
    tmp_path = None
    try:
        # Create temp file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        print(f"📥 Received file: {filename}, saved to {tmp_path}")
        
        # Process using service
        # ingest_file takes path and original filename (for extension detection context)
        success = ingestion_service.ingest_file(tmp_path, filename)
        
        if success:
            return {"status": "success", "message": f"Successfully ingested {filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest document")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")
    finally:
        # Cleanup temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"⚠️  Failed to delete temp file {tmp_path}: {e}")

def main():
    uvicorn.run(
        app,
        host=Config.ORCHESTRATOR_AGENT_HOST,
        port=Config.ORCHESTRATOR_AGENT_PORT,
    )

if __name__ == "__main__":
    main()
