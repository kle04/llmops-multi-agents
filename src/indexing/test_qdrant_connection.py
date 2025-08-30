#!/usr/bin/env python3
"""
Script test kết nối Qdrant đơn giản
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def test_qdrant_connection():
    """Test kết nối cơ bản với Qdrant"""
    qdrant_url = "http://localhost:6333"
    collection_name = "mental_health_vi"
    
    try:
        # Kết nối đến Qdrant
        print(f"🔗 Connecting to Qdrant at {qdrant_url}...")
        client = QdrantClient(url=qdrant_url)
        
        # Lấy danh sách collections
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"📋 Available collections: {collection_names}")
        
        # Kiểm tra collection cụ thể
        if collection_name in collection_names:
            collection_info = client.get_collection(collection_name)
            print(f"✅ Collection '{collection_name}' found!")
            print(f"   📊 Points: {collection_info.points_count}")
            print(f"   📊 Vector size: {collection_info.config.params.vectors.size}")
            print(f"   📊 Distance: {collection_info.config.params.vectors.distance}")
        else:
            print(f"⚠️ Collection '{collection_name}' not found.")
            print(f"💡 Will be created automatically when running embedding script.")
        
        # Test tạo collection mẫu (nếu chưa có)
        test_collection = "test_connection"
        if test_collection not in collection_names:
            print(f"\n🧪 Creating test collection: {test_collection}")
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"✅ Test collection created successfully!")
            
            # Xóa test collection
            client.delete_collection(test_collection)
            print(f"🗑️ Test collection deleted.")
        
        print(f"\n🎉 Qdrant connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"💡 Make sure Qdrant is running on {qdrant_url}")
        return False

if __name__ == "__main__":
    test_qdrant_connection()
