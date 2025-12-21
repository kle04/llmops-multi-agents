#!/usr/bin/env python3
"""
Qdrant Manager cho Mental Health RAG Agent
Quản lý vector database với tối ưu hóa cho domain tâm lý
"""

import os
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionStatus,
    PointStruct, Filter, FieldCondition, Range,
    MatchValue, SearchRequest, ScoredPoint
)
import uuid
from config import Config

class QdrantManager:
    def __init__(self):
        """
        Khởi tạo Qdrant Manager cho mental health domain
        """
        print(f"🗄️  Khởi tạo Qdrant Manager...")
        print(f"   URL: {Config.QDRANT_URL}")
        print(f"   Collection: {Config.COLLECTION_NAME}")
        
        try:
            # Khởi tạo client
            self.client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY if Config.QDRANT_API_KEY else None,
            )
            
            self.collection_name = Config.COLLECTION_NAME
            
            # Test connection
            self._test_connection()
            
            print(f"✅ Kết nối Qdrant thành công")
            
        except Exception as e:
            print(f"❌ Lỗi kết nối Qdrant: {e}")
            raise
    
    def _test_connection(self):
        """Test kết nối với Qdrant"""
        try:
            collections = self.client.get_collections()
            print(f"📊 Tìm thấy {len(collections.collections)} collections")
        except Exception as e:
            raise Exception(f"Không thể kết nối với Qdrant: {e}")
    
    def create_collection(self, vector_size: int, force_recreate: bool = False):
        """
        Tạo collection với cấu hình tối ưu cho mental health documents
        """
        try:
            print(f"📁 Tạo collection: {self.collection_name}")
            print(f"   Vector size: {vector_size}")
            
            # Kiểm tra collection có tồn tại không
            existing_collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in existing_collections)
            
            if collection_exists:
                if force_recreate:
                    print(f"🗑️  Xóa collection cũ...")
                    self.client.delete_collection(self.collection_name)
                else:
                    print(f"✅ Collection đã tồn tại")
                    return True
            
            # Tạo collection mới với cấu hình tối ưu
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,  # Cosine distance tốt cho text embeddings
                    on_disk=True  # Lưu trên disk để tiết kiệm RAM
                ),
                # Optimizers configuration
                optimizers_config={
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 0,
                    "max_segment_size": 20000,
                    "memmap_threshold": 50000,
                    "indexing_threshold": 20000,
                    "flush_interval_sec": 5,
                    "max_optimization_threads": 1
                },
                # HNSW configuration tối ưu cho retrieval accuracy
                hnsw_config={
                    "m": 16,  # Số connections, cân bằng accuracy và memory
                    "ef_construct": 100,  # Build time parameter
                    "full_scan_threshold": 10000,
                    "max_indexing_threads": 0,
                    "on_disk": False,
                    "payload_m": 16
                }
            )
            
            print(f"✅ Đã tạo collection thành công")
            
            # Wait for collection to be ready
            import time
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                try:
                    info = self.client.get_collection(self.collection_name)
                    if info.status == CollectionStatus.GREEN:
                        print(f"✅ Collection sẵn sàng")
                        break
                except:
                    pass
                time.sleep(1)
                wait_time += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tạo collection: {e}")
            return False
    
    def add_documents(self, documents_with_embeddings: List[Dict]):
        """
        Thêm documents vào collection với metadata đơn giản
        """
        if not documents_with_embeddings:
            print("⚠️  Không có documents để thêm")
            return False
        
        print(f"💾 Thêm {len(documents_with_embeddings)} documents vào collection...")
        
        try:
            points = []
            for i, doc in enumerate(documents_with_embeddings):
                if "embedding" not in doc:
                    print(f"⚠️  Document {i} không có embedding, bỏ qua")
                    continue
                
                # Tạo payload với metadata tối giản
                payload = {
                    "content": doc["content"],
                    "source": doc["source"],
                    "chunk_index": doc["chunk_index"],
                    "doc_id": doc["doc_id"],
                    "section": doc["section"]
                }
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=doc["embedding"],
                    payload=payload
                )
                points.append(point)
            
            if not points:
                print("❌ Không có points hợp lệ để thêm")
                return False
            
            # Batch upload với progress tracking
            batch_size = 100
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                print(f"   Batch {batch_num}/{total_batches}: {len(batch)} points")
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"✅ Đã thêm {len(points)} documents thành công")
            
            # Verify upload
            collection_info = self.get_collection_info()
            print(f"📊 Collection hiện có: {collection_info.get('vectors_count', 0)} vectors")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi thêm documents: {e}")
            return False
    
    
    def search_similar_documents(self, query_embedding: List[float], 
                               limit: int = None, 
                               filter_conditions: Dict = None,
                               score_threshold: float = None) -> List[Dict]:
        """
        Tìm kiếm documents tương đồng với query embedding
        """
        if limit is None:
            limit = Config.TOP_K_DOCUMENTS
        if score_threshold is None:
            score_threshold = Config.SIMILARITY_THRESHOLD
        
        try:
            print(f"🔍 Tìm kiếm {limit} documents tương đồng nhất...")
            print(f"   Score threshold: {score_threshold}")
            
            # Tạo filter nếu có
            search_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        for v in value:
                            conditions.append(FieldCondition(key=key, match=MatchValue(value=v)))
                    else:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                
                search_filter = Filter(must=conditions)
            
            # Thực hiện search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False  # Không cần return vectors để tiết kiệm bandwidth
            )
            
            if not search_results:
                print("⚠️  Không tìm thấy documents phù hợp")
                return []
            
            # Chuyển đổi kết quả
            results = []
            for result in search_results:
                doc = {
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload["content"],
                    "source": result.payload["source"],
                    "chunk_index": result.payload["chunk_index"],
                    "doc_id": result.payload["doc_id"],
                    "section": result.payload["section"]
                }
                results.append(doc)
            
            print(f"✅ Tìm thấy {len(results)} documents")
            
            return results
            
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """
        Lấy thông tin collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Convert to dict safely
            if hasattr(collection_info, 'model_dump'):
                info_dict = collection_info.model_dump()
            elif hasattr(collection_info, 'dict'):
                info_dict = collection_info.dict()
            else:
                info_dict = vars(collection_info)
            
            # Extract key information
            result = {
                "collection_name": self.collection_name,
                "status": info_dict.get("status", "unknown"),
                "vectors_count": info_dict.get("vectors_count", 0),
                "indexed_vectors_count": info_dict.get("indexed_vectors_count", 0),
                "points_count": info_dict.get("points_count", 0),
                "config": info_dict.get("config", {})
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Lỗi lấy thông tin collection: {e}",
                "collection_name": self.collection_name
            }
    
    def delete_collection(self):
        """
        Xóa collection
        """
        try:
            print(f"🗑️  Xóa collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            print(f"✅ Đã xóa collection thành công")
            return True
        except Exception as e:
            print(f"❌ Lỗi xóa collection: {e}")
            return False
    
    def search_by_source(self, query_embedding: List[float], 
                        sources: List[str], 
                        limit: int = None) -> List[Dict]:
        """
        Tìm kiếm documents theo source cụ thể
        """
        filter_conditions = {"source": sources}
        return self.search_similar_documents(
            query_embedding=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions
        )
    
    def search_by_section(self, query_embedding: List[float], 
                         sections: List[str], 
                         limit: int = None) -> List[Dict]:
        """
        Tìm kiếm documents theo section cụ thể
        """
        filter_conditions = {"section": sections}
        return self.search_similar_documents(
            query_embedding=query_embedding,
            limit=limit,
            filter_conditions=filter_conditions
        )
    
    def get_collection_stats(self) -> Dict:
        """
        Lấy thống kê chi tiết về collection
        """
        try:
            info = self.get_collection_info()
            
            if "error" in info:
                return info
            
            # Lấy thống kê theo sections và sources
            section_stats = {}
            source_stats = {}
            try:
                # Scroll through all points to get stats (for small collections)
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,  # Adjust based on collection size
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in points:
                    section = point.payload.get("section", "unknown")
                    source = point.payload.get("source", "unknown")
                    section_stats[section] = section_stats.get(section, 0) + 1
                    source_stats[source] = source_stats.get(source, 0) + 1
                    
            except Exception as e:
                print(f"⚠️  Không thể lấy stats: {e}")
            
            stats = {
                **info,
                "section_distribution": section_stats,
                "source_distribution": source_stats,
                "total_sections": len(section_stats),
                "total_sources": len(source_stats)
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Lỗi lấy stats: {e}"}
    
    def health_check(self) -> Dict:
        """
        Kiểm tra sức khỏe của Qdrant connection
        """
        try:
            # Test basic connection
            collections = self.client.get_collections()
            
            # Test collection access
            info = self.get_collection_info()
            
            return {
                "status": "healthy",
                "qdrant_url": Config.QDRANT_URL,
                "collection_name": self.collection_name,
                "collections_available": len(collections.collections),
                "collection_status": info.get("status", "unknown"),
                "vectors_count": info.get("vectors_count", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "qdrant_url": Config.QDRANT_URL,
                "collection_name": self.collection_name
            }
