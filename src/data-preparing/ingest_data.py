#!/usr/bin/env python3
"""
Data Ingestion Pipeline cho Mental Health RAG Agent
Xử lý các tài liệu PDF về tư vấn tâm lý học sinh sinh viên
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
import glob
from datetime import datetime

from utils.pdf_processor import PDFProcessor
from utils.markdown_processor import MarkdownProcessor
from utils.embedding_manager import EmbeddingManager
from utils.qdrant_manager import QdrantManager
from config import Config

class MentalHealthDataIngestion:
    def __init__(self):
        """
        Khởi tạo pipeline nạp dữ liệu cho domain tâm lý
        """
        print("🧠 Khởi tạo Mental Health Data Ingestion Pipeline...")
        print(f"   Collection: {Config.COLLECTION_NAME}")
        
        # Khởi tạo các components
        # Khởi tạo các components
        self.embedding_manager = EmbeddingManager()
        self.pdf_processor = PDFProcessor(embedding_manager=self.embedding_manager)
        self.markdown_processor = MarkdownProcessor(embedding_manager=self.embedding_manager)
        self.qdrant_manager = QdrantManager()
        
        print("✅ Pipeline đã sẵn sàng!")
    
    def check_prerequisites(self):
        """
        Kiểm tra các điều kiện tiên quyết
        """
        print("🔍 Kiểm tra điều kiện tiên quyết...")
        
        # Kiểm tra Qdrant connection
        try:
            qdrant_health = self.qdrant_manager.health_check()
            if qdrant_health["status"] == "healthy":
                print(f"✅ Kết nối Qdrant thành công")
                print(f"   Collections: {qdrant_health['collections_available']}")
            else:
                print(f"❌ Lỗi kết nối Qdrant: {qdrant_health.get('error')}")
                print("💡 Hãy khởi động Qdrant server:")
                print("   docker run -p 6333:6333 qdrant/qdrant")
                return False
        except Exception as e:
            print(f"❌ Lỗi kết nối Qdrant: {e}")
            return False
        
        # Kiểm tra embedding model
        try:
            test_result = self.embedding_manager.test_embedding_quality()
            if test_result["status"] == "success":
                print(f"✅ Embedding model hoạt động (dimension: {test_result['embedding_dimension']})")
            else:
                print(f"❌ Lỗi embedding model: {test_result.get('error')}")
                return False
        except Exception as e:
            print(f"❌ Lỗi embedding model: {e}")
            return False
        
        return True
    
    def find_files(self, paths: List[str]) -> List[str]:
        """
        Tìm tất cả file PDF và Markdown từ paths (có thể là file hoặc folder)
        """
        found_files = []
        
        if not paths:
            # Mặc định tìm trong thư mục data
            paths = ["data"]
        
        for path in paths:
            path = Path(path)
            
            if path.is_file():
                if path.suffix.lower() in ['.pdf', '.md']:
                    found_files.append(str(path))
                    print(f"✅ Tìm thấy file: {path}")
            elif path.is_dir():
                # Tìm tất cả PDF và Markdown trong folder
                for ext in ["*.pdf", "*.md"]:
                    pattern = str(path / "**" / ext)
                    matched = glob.glob(pattern, recursive=True)
                    if matched:
                        found_files.extend(matched)
                        print(f"✅ Tìm thấy {len(matched)} file {ext} trong {path}")
                        for f in matched:
                            print(f"   - {Path(f).name}")
            else:
                print(f"⚠️  Đường dẫn không tồn tại: {path}")
        
        if not found_files:
             print(f"⚠️  Không tìm thấy file nào trong các đường dẫn đã cho.")

        return found_files
    
    def analyze_file_content(self, file_paths: List[str]) -> Dict:
        """
        Phân tích cơ bản nội dung file
        """
        print(f"\n📊 Kiểm tra {len(file_paths)} files...")
        
        analysis = {
            "total_files": len(file_paths),
            "successfully_analyzed": 0,
            "files_analysis": []
        }
        
        for file_path in file_paths:
            try:
                print(f"\n🔍 Kiểm tra: {Path(file_path).name}")
                ext = Path(file_path).suffix.lower()
                text_sample = ""
                
                if ext == '.pdf':
                    # Trích xuất text để kiểm tra khả năng đọc
                    text_sample = self.pdf_processor.extract_text_from_pdf(file_path)
                elif ext == '.md':
                     with open(file_path, 'r', encoding='utf-8') as f:
                        text_sample = f.read()

                if not text_sample:
                    print("   ❌ Không trích xuất được text")
                    continue
                
                file_analysis = {
                    "file": Path(file_path).name,
                    "text_length": len(text_sample),
                    "readable": True
                }
                
                analysis["files_analysis"].append(file_analysis)
                analysis["successfully_analyzed"] += 1
                print(f"   ✅ Readable ({len(text_sample)} chars)")
                
            except Exception as e:
                print(f"   ❌ Lỗi đọc file: {e}")
                continue
        
        print(f"\n📊 Kết quả kiểm tra:")
        print(f"   - Files readable: {analysis['successfully_analyzed']}/{analysis['total_files']}")
        
        return analysis
    
    def process_files(self, file_paths: List[str], force_reprocess: bool = False) -> List[dict]:
        """
        Xử lý danh sách files
        """
        if not file_paths:
            print("❌ Không có file nào để xử lý!")
            return []
        
        print(f"\n📄 Bắt đầu xử lý {len(file_paths)} files...")
        
        all_documents = []
        success_count = 0
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"\n📖 [{i}/{len(file_paths)}] Xử lý: {Path(file_path).name}")
                
                # Kiểm tra file tồn tại
                if not os.path.exists(file_path):
                    print(f"   ❌ File không tồn tại: {file_path}")
                    continue
                
                # Xử lý file dựa trên extension
                ext = Path(file_path).suffix.lower()
                documents = []
                
                if ext == '.pdf':
                    documents = self.pdf_processor.process_pdf(file_path)
                elif ext == '.md':
                    documents = self.markdown_processor.process_markdown(file_path)
                
                if documents:
                    all_documents.extend(documents)
                    success_count += 1
                    print(f"   ✅ Tạo được {len(documents)} chunks")
                    
                    # Thống kê sections
                    sections = {}
                    for doc in documents:
                        section = doc.get("section", "unknown")
                        sections[section] = sections.get(section, 0) + 1
                    
                    print(f"   📊 Sections:")
                    for section, count in sections.items():
                        print(f"      - {section}: {count}")
                else:
                    print(f"   ⚠️  Không tạo được chunk nào")
                    
            except Exception as e:
                print(f"   ❌ Lỗi xử lý {file_path}: {e}")
                continue
        
        print(f"\n📊 Kết quả xử lý:")
        print(f"   - Thành công: {success_count}/{len(file_paths)} files")
        print(f"   - Tổng chunks: {len(all_documents)}")
        
        return all_documents
    
    def create_embeddings(self, documents: List[dict]) -> List[dict]:
        """
        Tạo embeddings cho documents
        """
        if not documents:
            return []
        
        print(f"\n🧮 Tạo embeddings cho {len(documents)} documents...")
        
        try:
            documents_with_embeddings = self.embedding_manager.embed_documents(documents)
            
            if documents_with_embeddings:
                print(f"✅ Đã tạo embeddings thành công")
                
                # Thống kê embedding
                stats = self.embedding_manager.get_embedding_stats(documents_with_embeddings)
                print(f"📊 Embedding stats:")
                print(f"   - Model: {stats.get('model_name')}")
                print(f"   - Dimension: {stats.get('embedding_dimension')}")
                print(f"   - Mean magnitude: {stats.get('mean_magnitude', 0):.3f}")
            
            return documents_with_embeddings
            
        except Exception as e:
            print(f"❌ Lỗi tạo embeddings: {e}")
            return []
    
    def store_in_vector_db(self, documents: List[dict], clear_existing: bool = False) -> bool:
        """
        Lưu documents vào vector database
        """
        if not documents:
            print("❌ Không có documents để lưu!")
            return False
        
        print(f"\n💾 Lưu {len(documents)} documents vào Qdrant...")
        
        try:
            # Xóa collection cũ nếu cần
            if clear_existing:
                print("🗑️  Xóa collection cũ...")
                self.qdrant_manager.delete_collection()
            
            # Tạo collection
            vector_size = self.embedding_manager.embedding_dimension
            self.qdrant_manager.create_collection(vector_size, force_recreate=clear_existing)
            
            # Lưu documents
            success = self.qdrant_manager.add_documents(documents)
            
            if success:
                # Kiểm tra kết quả
                stats = self.qdrant_manager.get_collection_stats()
                print(f"✅ Đã lưu thành công!")
                print(f"   Collection: {stats.get('collection_name')}")
                print(f"   Vectors: {stats.get('vectors_count', 0)}")
                print(f"   Sources: {stats.get('total_sources', 0)}")
                print(f"   Sections: {stats.get('total_sections', 0)}")
                
                # In phân bố sections
                if "section_distribution" in stats:
                    print(f"   📊 Section distribution:")
                    for section, count in stats["section_distribution"].items():
                        print(f"      - {section}: {count}")
            
            return success
            
        except Exception as e:
            print(f"❌ Lỗi lưu vào Qdrant: {e}")
            return False
    
    def run_ingestion(self, paths: List[str] = None, clear_existing: bool = False, 
                     force_reprocess: bool = False, analyze_only: bool = False):
        """
        Chạy toàn bộ pipeline ingestion
        """
        print("🧠 BẮT ĐẦU MENTAL HEALTH DATA INGESTION PIPELINE")
        print("=" * 60)
        print(f"⏰ Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Bước 1: Kiểm tra prerequisites
        if not self.check_prerequisites():
            print("❌ Không đáp ứng điều kiện tiên quyết!")
            return False
        
        # Bước 2: Tìm files
        print(f"\n📁 Tìm files...")
        if paths is None:
            paths = ["data"]  # Mặc định
        
        files = self.find_files(paths)
        
        if not files:
            print("❌ Không tìm thấy file nào!")
            print("💡 Hãy đặt file vào thư mục data/")
            return False
        
        print(f"\n✅ Sẽ xử lý {len(files)} files")
        
        # Bước 3: Phân tích nội dung
        analysis = self.analyze_file_content(files)
        
        if analyze_only:
            print(f"\n📊 KIỂM TRA HOÀN TẤT!")
            return True
        
        if analysis["successfully_analyzed"] == 0:
            print("⚠️  Không đọc được file nào!")
            print("💡 Hãy kiểm tra lại các file")
            return False
        
        # Bước 4: Xử lý files
        documents = self.process_files(files, force_reprocess)
        
        if not documents:
            print("❌ Không có documents để xử lý!")
            return False
        
        # Bước 5: Tạo embeddings
        documents_with_embeddings = self.create_embeddings(documents)
        
        if not documents_with_embeddings:
            print("❌ Không tạo được embeddings!")
            return False
        
        # Bước 6: Lưu vào vector DB
        success = self.store_in_vector_db(documents_with_embeddings, clear_existing)
        
        if success:
            print(f"\n🎉 HOÀN THÀNH DATA INGESTION!")
            print(f"📊 Thống kê cuối:")
            print(f"   - Files: {len(files)}")
            print(f"   - Documents: {len(documents_with_embeddings)}")
            print(f"   - Collection: {Config.COLLECTION_NAME}")
            print(f"   - Embedding model: {Config.EMBEDDING_MODEL}")
            return True
        else:
            print(f"\n❌ DATA INGESTION THẤT BẠI!")
            return False

def main():
    """
    Main function với argument parsing
    """
    parser = argparse.ArgumentParser(
        description="Mental Health Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python ingest_data.py                    # Nạp tất cả PDF từ thư mục 'data'
  python ingest_data.py file.pdf          # Nạp file cụ thể
  python ingest_data.py folder/           # Nạp từ folder cụ thể
  python ingest_data.py --clear           # Xóa collection cũ và nạp từ 'data'
  python ingest_data.py --analyze         # Chỉ phân tích nội dung, không nạp
  python ingest_data.py --check           # Kiểm tra hệ thống
        """
    )
    
    parser.add_argument(
        "paths", 
        nargs='*',  # 0 hoặc nhiều arguments
        help="Đường dẫn đến PDF files hoặc folders (mặc định: thư mục 'data')"
    )
    
    parser.add_argument(
        "--clear", 
        action="store_true", 
        help="Xóa collection cũ trước khi thêm dữ liệu mới"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force reprocess tất cả PDFs"
    )
    
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="Chỉ phân tích nội dung PDF, không nạp vào database"
    )
    
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Chỉ kiểm tra prerequisites và exit"
    )
    
    args = parser.parse_args()
    
    # Khởi tạo pipeline
    try:
        pipeline = MentalHealthDataIngestion()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo pipeline: {e}")
        sys.exit(1)
    
    # Nếu chỉ check
    if args.check:
        success = pipeline.check_prerequisites()
        if success:
            try:
                stats = pipeline.qdrant_manager.get_collection_stats()
                print(f"\n📊 Trạng thái hiện tại:")
                print(f"   Collection: {Config.COLLECTION_NAME}")
                print(f"   Vectors: {stats.get('vectors_count', 0)}")
                print(f"   Sources: {stats.get('total_sources', 0)}")
                print(f"   Sections: {stats.get('total_sections', 0)}")
            except:
                print(f"\n📊 Collection chưa tồn tại")
        sys.exit(0 if success else 1)
    
    # Xử lý paths - nếu không có paths thì dùng thư mục data mặc định
    pdf_paths = args.paths if args.paths else None
    
    # Chạy ingestion
    success = pipeline.run_ingestion(
        paths=pdf_paths,
        clear_existing=args.clear,
        force_reprocess=args.force,
        analyze_only=args.analyze
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
