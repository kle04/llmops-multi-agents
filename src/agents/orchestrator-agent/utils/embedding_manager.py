#!/usr/bin/env python3
"""
Embedding Manager cho Mental Health RAG Agent
Quản lý embeddings với tối ưu hóa cho nội dung tâm lý tiếng Việt
"""

import numpy as np
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import torch
from config import Config

class EmbeddingManager:
    def __init__(self):
        """
        Khởi tạo Embedding Manager với model tối ưu cho tiếng Việt + thuật ngữ tâm lý
        """
        print(f"🧮 Khởi tạo Embedding Manager...")
        print(f"   Model: {Config.EMBEDDING_MODEL}")
        
        # Load model
        try:
            print(f"   Loading {Config.EMBEDDING_MODEL}...")
            # Always use trust_remote_code=True for modern models (like Qwen)
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL, trust_remote_code=True)
            self.model_name = Config.EMBEDDING_MODEL
            
            # Get embedding dimension với text tiếng Việt
            test_texts = ["Xin chào", "test", "sức khỏe tâm lý"]
            sample_embeddings = self.model.encode(test_texts, convert_to_numpy=True)
            
            if len(sample_embeddings.shape) == 1:
                # Single embedding
                self.embedding_dimension = len(sample_embeddings)
            else:
                # Batch embeddings
                self.embedding_dimension = sample_embeddings.shape[1]
            
            print(f"✅ Đã load embedding model thành công")
            print(f"   Dimension: {self.embedding_dimension}")
            print(f"   Device: {self.model.device}")
            print(f"   Test embeddings shape: {sample_embeddings.shape}")
            
        except Exception as e:
            print(f"❌ Lỗi load embedding model: {e}")
            print(f"   Model: {Config.EMBEDDING_MODEL}")
            print(f"   Error type: {type(e).__name__}")
            raise
    
    def preprocess_text_for_embedding(self, text: str) -> str:
        """
        Tiền xử lý text cơ bản cho embedding - không thêm domain-specific keywords
        """
        if not text:
            return ""
        
        # Clean và truncate text để tránh lỗi với Vietnamese model
        cleaned_text = text.strip()
        
        # Remove hoặc replace các ký tự có thể gây vấn đề
        import re
        # Remove control characters và non-printable chars
        cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', ' ', cleaned_text)
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Truncate text nếu quá dài
        # multilingual-e5-base supports ~512 tokens ≈ 2000-2500 chars for Vietnamese
        # Using 2000 as safe limit to avoid truncation of 800-1000 char chunks
        max_chars = 2000  # Safe limit for multilingual-e5-base model
        if len(cleaned_text) > max_chars:
            # Cắt ở boundary của câu để giữ ngữ nghĩa
            sentences = cleaned_text.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= max_chars:
                    truncated += sentence + '. '
                else:
                    break
            cleaned_text = truncated.strip()
            if not cleaned_text.endswith('.'):
                cleaned_text += '.'
        
        # Final validation
        if not cleaned_text or len(cleaned_text.strip()) < 3:
            return ""
            
        return cleaned_text
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Tạo embedding cho query với tiền xử lý đặc biệt
        """
        if not query.strip():
            return np.zeros(self.embedding_dimension)
        
        try:
            # Chỉ tiền xử lý query mà không thêm metadata type
            processed_query = self.preprocess_text_for_embedding(query)
            
            # Tạo embedding thuần túy từ nội dung
            embedding = self.model.encode(
                processed_query,
                convert_to_numpy=True,
                normalize_embeddings=Config.NORMALIZE_EMBEDDINGS
            )
            
            return embedding
            
        except Exception as e:
            print(f"❌ Lỗi tạo embedding cho query: {e}")
            return np.zeros(self.embedding_dimension)
    
    def embed_document(self, document: Dict) -> np.ndarray:
        """
        Tạo embedding cho document thuần túy từ nội dung
        """
        try:
            content = document.get("content", "")
            
            if not content.strip():
                return np.zeros(self.embedding_dimension)
            
            # Chỉ tiền xử lý content mà không thêm metadata type
            processed_content = self.preprocess_text_for_embedding(content)
            
            # Tạo embedding thuần túy từ nội dung
            embedding = self.model.encode(
                processed_content,
                convert_to_numpy=True,
                normalize_embeddings=Config.NORMALIZE_EMBEDDINGS
            )
            
            return embedding
            
        except Exception as e:
            print(f"❌ Lỗi tạo embedding cho document: {e}")
            return np.zeros(self.embedding_dimension)
    
    def embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Tạo embeddings cho danh sách documents với batch processing
        """
        if not documents:
            return []
        
        print(f"🧮 Tạo embeddings cho {len(documents)} documents...")
        
        documents_with_embeddings = []
        batch_size = Config.EMBEDDING_BATCH_SIZE
        
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                print(f"   Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}: {len(batch)} documents")
                
                # Tạo embeddings cho batch
                batch_texts = []
                for doc in batch:
                    content = doc.get("content", "")
                    
                    # Chỉ tiền xử lý content mà không thêm metadata type
                    processed_content = self.preprocess_text_for_embedding(content)
                    batch_texts.append(processed_content)
                
                # Batch encoding với error handling tốt hơn
                try:
                    # Tạo mapping giữa valid texts và original indices
                    valid_items = []
                    for j, text in enumerate(batch_texts):
                        if text.strip():
                            valid_items.append((j, text.strip()))
                    
                    if not valid_items:
                        print(f"   ⚠️  Batch {i//batch_size + 1} không có text hợp lệ")
                        continue
                    
                    # Extract chỉ valid texts
                    valid_texts = [item[1] for item in valid_items]
                    
                    # Debug info
                    print(f"   📊 Batch {i//batch_size + 1}: {len(batch_texts)} total, {len(valid_texts)} valid")
                    
                    # Safe encoding với Vietnamese model
                    try:
                        batch_embeddings = self.model.encode(
                            valid_texts,
                            convert_to_numpy=True,
                            normalize_embeddings=Config.NORMALIZE_EMBEDDINGS,
                            batch_size=min(len(valid_texts), 8),  # Smaller batch for Vietnamese model
                            show_progress_bar=False
                        )
                    except Exception as encode_error:
                        print(f"   ⚠️  Batch encoding failed: {encode_error}")
                        # Try with even smaller batch
                        if len(valid_texts) > 1:
                            print(f"   🔄 Retrying with batch_size=1...")
                            batch_embeddings = self.model.encode(
                                valid_texts,
                                convert_to_numpy=True,
                                normalize_embeddings=Config.NORMALIZE_EMBEDDINGS,
                                batch_size=1,
                                show_progress_bar=False
                            )
                        else:
                            raise encode_error
                    
                    # Đảm bảo shape đúng
                    if len(batch_embeddings.shape) == 1:
                        batch_embeddings = batch_embeddings.reshape(1, -1)
                    
                    print(f"   📐 Embeddings shape: {batch_embeddings.shape}, expected: ({len(valid_texts)}, {self.embedding_dimension})")
                    
                    # Kiểm tra consistency
                    if batch_embeddings.shape[0] != len(valid_texts):
                        print(f"   ❌ Shape mismatch: {batch_embeddings.shape[0]} embeddings vs {len(valid_texts)} texts")
                        continue
                    
                    # Thêm embeddings vào documents với mapping chính xác
                    for embed_idx, (original_idx, _) in enumerate(valid_items):
                        if embed_idx >= len(batch_embeddings):
                            print(f"   ❌ Embedding index {embed_idx} out of range for {len(batch_embeddings)} embeddings")
                            break
                            
                        doc = batch[original_idx]
                        doc_with_embedding = doc.copy()
                        doc_with_embedding["embedding"] = batch_embeddings[embed_idx].tolist()
                        doc_with_embedding["embedding_model"] = self.model_name
                        doc_with_embedding["embedding_dimension"] = self.embedding_dimension
                        documents_with_embeddings.append(doc_with_embedding)
                    
                    print(f"   ✅ Successfully processed {len(valid_items)} documents")
                            
                except Exception as batch_error:
                    print(f"   ❌ Lỗi batch {i//batch_size + 1}: {batch_error}")
                    print(f"   📊 Batch info: {len(batch_texts)} texts, {[len(t) for t in batch_texts[:3]]} chars")
                    
                    # Fallback: Xử lý từng document riêng lẻ
                    print(f"   🔄 Fallback: Xử lý từng document riêng lẻ...")
                    for j, doc in enumerate(batch):
                        try:
                            content = doc.get("content", "").strip()
                            if not content:
                                continue
                                
                            # Validate và preprocess content
                            processed_content = self.preprocess_text_for_embedding(content)
                            if len(processed_content) < 10:  # Skip very short texts
                                print(f"      ⚠️  Skipping too short text: {len(processed_content)} chars")
                                continue
                            
                            # Encode single document với additional safety
                            single_embedding = self.model.encode(
                                processed_content,
                                convert_to_numpy=True,
                                normalize_embeddings=Config.NORMALIZE_EMBEDDINGS,
                                show_progress_bar=False
                            )
                            
                            # Đảm bảo là 1D array
                            if len(single_embedding.shape) > 1:
                                single_embedding = single_embedding.flatten()
                            
                            doc_with_embedding = doc.copy()
                            doc_with_embedding["embedding"] = single_embedding.tolist()
                            doc_with_embedding["embedding_model"] = self.model_name
                            doc_with_embedding["embedding_dimension"] = self.embedding_dimension
                            documents_with_embeddings.append(doc_with_embedding)
                            
                        except Exception as single_error:
                            print(f"      ❌ Lỗi document {j}: {single_error}")
                            continue
                    
                    print(f"   🔄 Fallback completed for batch {i//batch_size + 1}")
                    continue
            
            print(f"✅ Đã tạo embeddings cho tất cả documents")
            return documents_with_embeddings
            
        except Exception as e:
            print(f"❌ Lỗi tạo embeddings: {e}")
            return []
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Tính độ tương đồng cosine giữa 2 embeddings
        """
        try:
            # Ensure embeddings are numpy arrays
            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1)
            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2)
            
            # Normalize if needed
            if Config.NORMALIZE_EMBEDDINGS:
                embedding1 = embedding1 / np.linalg.norm(embedding1)
                embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Lỗi tính similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         document_embeddings: List[np.ndarray], 
                         top_k: int = None) -> List[int]:
        """
        Tìm các documents có embedding tương đồng nhất với query
        """
        if top_k is None:
            top_k = Config.TOP_K_DOCUMENTS
        
        try:
            similarities = []
            for i, doc_embedding in enumerate(document_embeddings):
                similarity = self.compute_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sắp xếp theo độ tương đồng giảm dần
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy top k indices
            top_indices = [idx for idx, _ in similarities[:top_k]]
            return top_indices
            
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm similar embeddings: {e}")
            return []
    
    def get_embedding_stats(self, documents_with_embeddings: List[Dict]) -> Dict:
        """
        Lấy thống kê về embeddings
        """
        if not documents_with_embeddings:
            return {}
        
        embeddings = [doc["embedding"] for doc in documents_with_embeddings if "embedding" in doc]
        
        if not embeddings:
            return {}
        
        embeddings_array = np.array(embeddings)
        
        stats = {
            "total_embeddings": len(embeddings),
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "mean_magnitude": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            "std_magnitude": float(np.std(np.linalg.norm(embeddings_array, axis=1))),
            "normalized": Config.NORMALIZE_EMBEDDINGS
        }
        
        return stats
    
    def test_embedding_quality(self) -> Dict:
        """
        Test chất lượng embedding với các câu mẫu về tâm lý
        """
        print("🧪 Testing embedding quality...")
        
        test_queries = [
            "tôi cảm thấy buồn và mệt mỏi",
            "làm sao để giảm stress học tập",
            "triệu chứng của trầm cảm là gì",
            "tôi gặp khó khăn trong việc ngủ"
        ]
        
        test_documents = [
            {"content": "Trầm cảm là một rối loạn tâm lý phổ biến", "content_type": "symptom_description"},
            {"content": "Các kỹ thuật thư giãn giúp giảm căng thẳng", "content_type": "intervention_guidance"},
            {"content": "Mất ngủ có thể là dấu hiệu của lo âu", "content_type": "symptom_description"},
            {"content": "Học sinh cần có thời gian nghỉ ngơi hợp lý", "content_type": "student_focused"}
        ]
        
        results = {}
        
        try:
            # Test query embeddings
            query_embeddings = [self.embed_query(q) for q in test_queries]
            
            # Test document embeddings
            doc_embeddings = [self.embed_document(d) for d in test_documents]
            
            # Test similarities
            similarities = []
            for i, q_emb in enumerate(query_embeddings):
                for j, d_emb in enumerate(doc_embeddings):
                    sim = self.compute_similarity(q_emb, d_emb)
                    similarities.append({
                        "query": test_queries[i],
                        "document": test_documents[j]["content"][:50] + "...",
                        "similarity": sim
                    })
            
            # Find best matches
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            results = {
                "status": "success",
                "query_embeddings_generated": len(query_embeddings),
                "document_embeddings_generated": len(doc_embeddings),
                "top_similarities": similarities[:3],
                "embedding_dimension": self.embedding_dimension,
                "model": self.model_name
            }
            
            print("✅ Embedding quality test completed")
            
        except Exception as e:
            results = {
                "status": "error",
                "error": str(e)
            }
            print(f"❌ Embedding quality test failed: {e}")
        
        return results
