"""
PostgreSQL Memory Utilities

Mục tiêu
--------
- Cung cấp lớp quản lý kết nối PostgreSQL (async) dùng chung cho Orchestrator Agent.
- Lưu trữ lịch sử hội thoại lâu dài (long-term memory).
- Structure tương tự RedisManager nhưng dùng asyncpg.

Thiết kế chính
--------------
- PostgresManager: Quản lý connection pool.
- PostgresChatHistoryStore: Lớp thao tác lịch sử dựa trên PostgresManager.
"""

import logging
import asyncpg
from typing import Any, Dict, List, Optional
from datetime import datetime
from config import Config
from redis_memory import ChatHistory

logger = logging.getLogger(__name__)

class PostgresManager:
    """
    Quản lý kết nối PostgreSQL (async) và tiện ích thao tác dữ liệu.
    """

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None
        self.postgres_config = {
            "host": Config.POSTGRES_HOST,
            "port": Config.POSTGRES_PORT,
            "user": Config.POSTGRES_USER,
            "password": Config.POSTGRES_PASSWORD,
            "database": Config.POSTGRES_DB,
        }

    async def initialize(self) -> None:
        if self.pool is not None:
            return
        try:
            self.pool = await asyncpg.create_pool(**self.postgres_config)
            logger.info("✅ PostgresManager: kết nối PostgreSQL thành công")
            
            # Initialize schema
            await self._init_schema()
            
        except Exception as e:
            logger.error(f"❌ PostgresManager: không thể kết nối PostgreSQL: {e}")
            self.pool = None
            raise

    async def _init_schema(self) -> None:
        """Đọc file schema.sql và khởi tạo tables nếu chưa có."""
        if not self.pool:
            return
        
        # Đọc nội dung file schema
        # Giả sử file nằm cùng thư mục hoặc đường dẫn cố định
        try:
            with open("postgres_schema.sql", "r", encoding="utf-8") as f:
                schema_sql = f.read()
            
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("✅ PostgresManager: đã kiểm tra/khởi tạo schema")
        except FileNotFoundError:
            logger.warning("⚠️ PostgresManager: không tìm thấy file postgres_schema.sql, bỏ qua init schema")
        except Exception as e:
            logger.error(f"❌ PostgresManager: lỗi khi init schema: {e}")

    async def close(self) -> None:
        if self.pool is not None:
            try:
                await self.pool.close()
                logger.info("✅ PostgresManager: đã đóng kết nối")
            finally:
                self.pool = None

    def is_ready(self) -> bool:
        return self.pool is not None

    async def health_check(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "connected": False,
            "host": self.postgres_config["host"],
            "port": self.postgres_config["port"],
            "database": self.postgres_config["database"],
        }
        if not self.pool:
            status["error"] = "Pool not initialized"
            return status
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            status["connected"] = True
            return status
        except Exception as e:
            status["error"] = str(e)
            return status


class PostgresChatHistoryStore:
    """
    Lớp làm việc với PostgreSQL để lưu/đọc lịch sử chat (Long-term memory).
    """

    def __init__(self, postgres_manager: PostgresManager) -> None:
        self.pg = postgres_manager

    async def append_message(
        self,
        user_id: str,
        session_id: str,
        *,
        role: str,
        content: str,
        agent_used: Optional[str] = None,
        source: Optional[List[str]] = None,
    ) -> None:
        """
        Lưu message vào bảng messages.
        Đồng thời cập nhật hoặc tạo mới session trong bảng sessions.
        """
        if not self.pg.is_ready() or not self.pg.pool:
            logger.warning("Postgres chưa sẵn sàng, bỏ qua lưu message")
            return

        try:
            async with self.pg.pool.acquire() as conn:
                async with conn.transaction():
                    # 1. Upsert User
                    await conn.execute("""
                        INSERT INTO users (user_id, last_seen)
                        VALUES ($1, NOW())
                        ON CONFLICT (user_id)
                        DO UPDATE SET last_seen = NOW()
                    """, user_id)

                    # 2. Upsert Session
                    await conn.execute("""
                        INSERT INTO sessions (session_id, user_id, last_updated)
                        VALUES ($1, $2, NOW())
                        ON CONFLICT (session_id) 
                        DO UPDATE SET last_updated = NOW(), user_id = EXCLUDED.user_id
                    """, session_id, user_id)

                    # 3. Insert Message
                    await conn.execute("""
                        INSERT INTO messages (session_id, role, content, agent_used, source, user_id)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, session_id, role, content, agent_used, source or [], user_id)
                    
        except Exception as e:
            logger.error(f"Lỗi khi lưu message vào Postgres: {e}")

    async def load_session_history(self, user_id: str, session_id: str) -> ChatHistory:
        """
        Load toàn bộ lịch sử của session từ Postgres và convert sang ChatHistory object.
        """
        chat = ChatHistory()
        if not self.pg.is_ready() or not self.pg.pool:
            return chat

        try:
            async with self.pg.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT role, content, timestamp, agent_used, source, user_id
                    FROM messages
                    WHERE session_id = $1
                    ORDER BY timestamp ASC
                """, session_id)
                
                for row in rows:
                    # Convert row to dict format expected by ChatHistory
                    # ChatHistory.add_message tự thêm timestamp=now(), nên ta cần set lại timestamp gốc nếu muốn chính xác
                    # Tuy nhiên ChatHistory hiện tại thiết kế in-memory đơn giản, add_message ko nhận timestamp custom.
                    # Ta có thể sửa ChatHistory hoặc manual append.
                    
                    # Manual append to match structure
                    msg = {
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else datetime.now().isoformat(),
                        "agent_used": row["agent_used"],
                        "user_id": row["user_id"],
                        "source": row["source"] or []
                    }
                    chat.messages.append(msg)
                
                # Update metadata if session exists
                session_row = await conn.fetchrow("SELECT created_at, last_updated FROM sessions WHERE session_id = $1", session_id)
                if session_row:
                    chat.created_at = session_row["created_at"]
                    chat.last_updated = session_row["last_updated"]
                    
        except Exception as e:
            logger.error(f"Lỗi khi load history từ Postgres: {e}")
            
        return chat

    async def list_sessions(self, user_id: str) -> List[str]:
        """
        Liệt kê danh sách session_id của user từ Postgres.
        """
        if not self.pg.is_ready() or not self.pg.pool:
            return []

        try:
            async with self.pg.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT session_id
                    FROM sessions
                    WHERE user_id = $1
                    ORDER BY last_updated DESC
                """, user_id)
                return [row["session_id"] for row in rows]
        except Exception as e:
            logger.error(f"Lỗi khi list sessions từ Postgres: {e}")
            return []
            
    async def delete_session(self, user_id: str, session_id: str) -> bool:
        """
        Xoá toàn bộ lịch sử của session (messages + session record).
        
        Args:
            user_id: ID người dùng (để đảm bảo quyền sở hữu)
            session_id: ID session cần xoá
            
        Returns:
            bool: True nếu xoá thành công (hoặc không tồn tại), False nếu lỗi.
        """
        if not self.pg.is_ready() or not self.pg.pool:
            return False
            
        try:
            async with self.pg.pool.acquire() as conn:
                # With ON DELETE CASCADE defined in schema:
                # Constraint fk_session FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                # We only need to delete from sessions table.
                
                result = await conn.execute("""
                    DELETE FROM sessions
                    WHERE session_id = $1 AND user_id = $2
                """, session_id, user_id)
                
                deleted_count = int(result.split(" ")[1])
                if deleted_count > 0:
                    logger.info(f"Đã xoá session {session_id} của user {user_id} (kèm messages)")
                else:
                    logger.info(f"Session {session_id} không tồn tại hoặc không thuộc về {user_id}")
                        
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xoá session {session_id} từ Postgres: {e}")
            return False
