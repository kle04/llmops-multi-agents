from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mental_health_vi")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:5000")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))

# Global variables
qdrant_client = None
http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    global qdrant_client, http_client
    
    # Startup
    logger.info("Starting Context Retrieval Service...")
    
    # Initialize Qdrant client
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL)
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
        
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            logger.warning(f"Collection '{QDRANT_COLLECTION}' not found. Available collections: {collection_names}")
        else:
            logger.info(f"Collection '{QDRANT_COLLECTION}' is available")
            
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise
    
    # Initialize HTTP client for embedding service
    http_client = httpx.AsyncClient(timeout=30.0)
    
    # Test embedding service connection
    try:
        response = await http_client.get(f"{EMBEDDING_SERVICE_URL}/health")
        if response.status_code == 200:
            logger.info(f"Embedding service is available at {EMBEDDING_SERVICE_URL}")
        else:
            logger.warning(f"Embedding service health check failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to embedding service: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Context Retrieval Service...")
    if http_client:
        await http_client.aclose()
    if qdrant_client:
        qdrant_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="Context Retrieval Service",
    description="Service for retrieving relevant context from Vietnamese mental health documents",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models
class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5
    score_threshold: Optional[float] = 0.7
    filters: Optional[Dict[str, Any]] = None

class DocumentChunk(BaseModel):
    id: str
    chunk_id: str
    doc_id: str
    title: str
    source: str
    year: Optional[float] = None
    language: str
    audience: str
    grade_range: str
    topics: str
    section: str
    text: str
    token_count: int
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentChunk]
    total_found: int
    search_time_ms: float

class HealthResponse(BaseModel):
    status: str
    qdrant_status: str
    embedding_service_status: str
    collection: str

# Helper functions
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text using the embedding service"""
    try:
        response = await http_client.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return result["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding service error: {str(e)}")

def build_qdrant_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    """Build Qdrant filter from search filters"""
    if not filters:
        return None
    
    conditions = []
    
    # Filter by document metadata
    for field, value in filters.items():
        if field in ["doc_id", "source", "language", "audience", "grade_range", "topics"]:
            if isinstance(value, list):
                # Multiple values - use should match any
                for v in value:
                    conditions.append(FieldCondition(key=field, match=MatchValue(value=v)))
            else:
                conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        elif field == "year" and isinstance(value, dict):
            # Year range filter
            if "min" in value:
                conditions.append(FieldCondition(key="year", range={"gte": value["min"]}))
            if "max" in value:
                conditions.append(FieldCondition(key="year", range={"lte": value["max"]}))
    
    return Filter(must=conditions) if conditions else None

def format_search_result(hit, query: str) -> DocumentChunk:
    """Format Qdrant search result into DocumentChunk"""
    payload = hit.payload
    return DocumentChunk(
        id=payload.get("id", ""),
        chunk_id=payload.get("chunk_id", ""),
        doc_id=payload.get("doc_id", ""),
        title=payload.get("title", ""),
        source=payload.get("source", ""),
        year=payload.get("year"),
        language=payload.get("language", ""),
        audience=payload.get("audience", ""),
        grade_range=payload.get("grade_range", ""),
        topics=payload.get("topics", ""),
        section=payload.get("section", ""),
        text=payload.get("text", ""),
        token_count=payload.get("token_count", 0),
        score=hit.score
    )

# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    qdrant_status = "unknown"
    embedding_service_status = "unknown"
    
    # Check Qdrant
    try:
        collections = qdrant_client.get_collections()
        qdrant_status = "healthy"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        qdrant_status = "unhealthy"
    
    # Check embedding service
    try:
        response = await http_client.get(f"{EMBEDDING_SERVICE_URL}/health")
        if response.status_code == 200:
            embedding_service_status = "healthy"
        else:
            embedding_service_status = "unhealthy"
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        embedding_service_status = "unhealthy"
    
    overall_status = "healthy" if qdrant_status == "healthy" and embedding_service_status == "healthy" else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        qdrant_status=qdrant_status,
        embedding_service_status=embedding_service_status,
        collection=QDRANT_COLLECTION
    )

@app.post("/search", response_model=SearchResponse)
async def search_context(search_query: SearchQuery):
    """Search for relevant context based on query"""
    import time
    start_time = time.time()
    
    try:
        # Get embedding for the query
        query_embedding = await get_embedding(search_query.query)
        
        # Build filter
        qdrant_filter = build_qdrant_filter(search_query.filters)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=min(search_query.limit, MAX_RESULTS),
            score_threshold=search_query.score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = [
            format_search_result(hit, search_query.query) 
            for hit in search_results
        ]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=search_query.query,
            results=results,
            total_found=len(results),
            search_time_ms=round(search_time_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/search", response_model=SearchResponse)
async def search_context_get(
    q: str,
    limit: int = 5,
    score_threshold: float = 0.7,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    topics: Optional[str] = None,
    audience: Optional[str] = None,
    grade_range: Optional[str] = None
):
    """Search for relevant context using GET method with query parameters"""
    # Build filters from query parameters
    filters = {}
    if doc_id:
        filters["doc_id"] = doc_id
    if source:
        filters["source"] = source
    if topics:
        filters["topics"] = topics.split(",") if "," in topics else topics
    if audience:
        filters["audience"] = audience.split(",") if "," in audience else audience
    if grade_range:
        filters["grade_range"] = grade_range.split(",") if "," in grade_range else grade_range
    
    search_query = SearchQuery(
        query=q,
        limit=limit,
        score_threshold=score_threshold,
        filters=filters if filters else None
    )
    
    return await search_context(search_query)

@app.get("/collections")
async def get_collections():
    """Get available collections in Qdrant"""
    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "status": col.status,
                    "vectors_count": col.vectors_count,
                    "points_count": col.points_count
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

@app.get("/collection/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get detailed information about a specific collection"""
    try:
        collection_info = qdrant_client.get_collection(collection_name)
        return {
            "name": collection_name,
            "status": collection_info.status,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "config": collection_info.config.dict() if collection_info.config else None
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Context Retrieval Service",
        "version": "1.0.0",
        "description": "Service for retrieving relevant context from Vietnamese mental health documents",
        "endpoints": {
            "health": "/health",
            "search_post": "/search",
            "search_get": "/search?q=your_query",
            "collections": "/collections",
            "collection_info": "/collection/{collection_name}/info"
        },
        "config": {
            "qdrant_url": QDRANT_URL,
            "collection": QDRANT_COLLECTION,
            "embedding_service": EMBEDDING_SERVICE_URL,
            "max_results": MAX_RESULTS
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5005"))
    uvicorn.run(app, host="0.0.0.0", port=port)
