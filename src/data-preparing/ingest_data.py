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
    
    def find_pdf_files(self, paths: List[str]) -> List[str]:
        """
        Tìm tất cả file PDF từ paths (có thể là file hoặc folder)
        """
        pdf_files = []
        
        if not paths:
            # Mặc định tìm trong thư mục data
            paths = ["data"]
        
        for path in paths:
            path = Path(path)
            
            if path.is_file() and path.suffix.lower() == '.pdf':
                pdf_files.append(str(path))
                print(f"✅ Tìm thấy file: {path}")
            elif path.is_dir():
                # Tìm tất cả PDF trong folder
                pattern = str(path / "**" / "*.pdf")
                found_files = glob.glob(pattern, recursive=True)
                if found_files:
                    pdf_files.extend(found_files)
                    print(f"✅ Tìm thấy {len(found_files)} PDF files trong {path}")
                    for f in found_files:
                        print(f"   - {Path(f).name}")
                else:
                    print(f"⚠️  Không tìm thấy PDF nào trong: {path}")
            else:
                print(f"⚠️  Đường dẫn không tồn tại: {path}")
        
        return pdf_files
    
    def analyze_pdf_content(self, pdf_files: List[str]) -> Dict:
        """
        Phân tích cơ bản nội dung PDF
        """
        print(f"\n📊 Kiểm tra {len(pdf_files)} PDF files...")
        
        analysis = {
            "total_files": len(pdf_files),
            "successfully_analyzed": 0,
            "files_analysis": []
        }
        
        for pdf_file in pdf_files:
            try:
                print(f"\n🔍 Kiểm tra: {Path(pdf_file).name}")
                
                # Trích xuất text để kiểm tra khả năng đọc
                text_sample = self.pdf_processor.extract_text_from_pdf(pdf_file)
                
                if not text_sample:
                    print("   ❌ Không trích xuất được text")
                    continue
                
                file_analysis = {
                    "file": Path(pdf_file).name,
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
    
    def process_pdfs(self, pdf_files: List[str], force_reprocess: bool = False) -> List[dict]:
        """
        Xử lý danh sách PDF files
        """
        if not pdf_files:
            print("❌ Không có file PDF nào để xử lý!")
            return []
        
        print(f"\n📄 Bắt đầu xử lý {len(pdf_files)} PDF files...")
        
        all_documents = []
        success_count = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            try:
                print(f"\n📖 [{i}/{len(pdf_files)}] Xử lý: {Path(pdf_file).name}")
                
                # Kiểm tra file tồn tại
                if not os.path.exists(pdf_file):
                    print(f"   ❌ File không tồn tại: {pdf_file}")
                    continue
                
                # Xử lý PDF
                documents = self.pdf_processor.process_pdf(pdf_file)
                
                if documents:
                    all_documents.extend(documents)
                    success_count += 1
                    print(f"   ✅ Tạo được {len(documents)} chunks")
                    
                    # Thống kê sections
                    sections = {}
                    for doc in documents:
                        section = doc["section"]
                        sections[section] = sections.get(section, 0) + 1
                    
                    print(f"   📊 Sections:")
                    for section, count in sections.items():
                        print(f"      - {section}: {count}")
                else:
                    print(f"   ⚠️  Không tạo được chunk nào")
                    
            except Exception as e:
                print(f"   ❌ Lỗi xử lý {pdf_file}: {e}")
                continue
        
        print(f"\n📊 Kết quả xử lý:")
        print(f"   - Thành công: {success_count}/{len(pdf_files)} files")
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
    
    def run_ingestion(self, pdf_paths: List[str] = None, clear_existing: bool = False, 
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
        
        # Bước 2: Tìm PDF files
        print(f"\n📁 Tìm PDF files...")
        if pdf_paths is None:
            pdf_paths = ["data"]  # Mặc định
        
        pdf_files = self.find_pdf_files(pdf_paths)
        
        if not pdf_files:
            print("❌ Không tìm thấy PDF files nào!")
            print("💡 Hãy đặt PDF files vào thư mục data/")
            return False
        
        print(f"\n✅ Sẽ xử lý {len(pdf_files)} PDF files")
        
        # Bước 3: Phân tích nội dung
        analysis = self.analyze_pdf_content(pdf_files)
        
        if analyze_only:
            print(f"\n📊 KIỂM TRA HOÀN TẤT!")
            return True
        
        if analysis["successfully_analyzed"] == 0:
            print("⚠️  Không đọc được PDF nào!")
            print("💡 Hãy kiểm tra lại các file PDF")
            return False
        
        # Bước 4: Xử lý PDFs
        documents = self.process_pdfs(pdf_files, force_reprocess)
        
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
            print(f"   - PDF files: {len(pdf_files)}")
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
        pdf_paths=pdf_paths,
        clear_existing=args.clear,
        force_reprocess=args.force,
        analyze_only=args.analyze
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
