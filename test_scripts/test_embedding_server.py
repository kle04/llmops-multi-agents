#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for embedding server performance and functionality.
"""

import asyncio
import time
import statistics
import httpx
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import argparse

# Test data - Vietnamese educational/mental health texts
TEST_TEXTS = [
    "Sức khỏe tâm thần của học sinh là vấn đề quan trọng trong giáo dục.",
    "Công tác xã hội trường học giúp hỗ trợ các em học sinh gặp khó khăn.",
    "Bạo lực học đường là hiện tượng cần được ngăn chặn và xử lý kịp thời.",
    "Các hoạt động tư vấn tâm lý trong trường học rất cần thiết.",
    "Phòng ngừa và can thiệp sớm các vấn đề hành vi ở học sinh.",
    "Gia đình và nhà trường cần phối hợp trong việc chăm sóc học sinh.",
    "Kỹ năng sống và kỹ năng xã hội cần được giảng dạy từ sớm.",
    "Hệ thống hỗ trợ tâm lý cho giáo viên cũng rất quan trọng.",
    "Các chương trình giáo dục về sức khỏe tâm thần cần được triển khai.",
    "Môi trường học tập tích cực giúp phát triển nhân cách học sinh.",
    "Việc xây dựng mối quan hệ tốt giữa thầy và trò là nền tảng.",
    "Các biện pháp hỗ trợ học sinh có hoàn cảnh đặc biệt.",
    "Tầm quan trọng của việc lắng nghe và thấu hiểu học sinh.",
    "Phương pháp giáo dục tích cực thay cho hình phạt tiêu cực.",
    "Vai trò của cộng đồng trong việc bảo vệ trẻ em.",
]

class EmbeddingServerTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if embedding server is healthy"""
        try:
            response = await self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {
                "status": "healthy",
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "response_time": None,
                "status_code": None
            }
    
    async def single_embedding_test(self, text: str) -> Dict[str, Any]:
        """Test single text embedding"""
        start_time = time.time()
        try:
            response = await self.session.post(
                f"{self.base_url}/embed",
                json={"text": text}
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            return {
                "success": True,
                "response_time": time.time() - start_time,
                "embedding_dim": len(embedding),
                "text_length": len(text),
                "status_code": response.status_code
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "text_length": len(text)
            }
    
    async def batch_embedding_test(self, texts: List[str], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Test multiple embeddings with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_embedding_test(text: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.single_embedding_test(text)
        
        tasks = [bounded_embedding_test(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def load_test(self, text: str, num_requests: int, max_concurrent: int = 10) -> Dict[str, Any]:
        """Load test with same text repeated"""
        start_time = time.time()
        
        results = await self.batch_embedding_test([text] * num_requests, max_concurrent)
        
        total_time = time.time() - start_time
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        response_times = [r["response_time"] for r in successful]
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_time": total_time,
            "requests_per_second": len(successful) / total_time if total_time > 0 else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "success_rate": len(successful) / num_requests if num_requests > 0 else 0,
            "errors": [r.get("error") for r in failed]
        }

async def run_comprehensive_test(base_url: str, num_load_requests: int = 50, max_concurrent: int = 10):
    """Run comprehensive test suite"""
    print(f"🚀 Testing Embedding Server at {base_url}")
    print("=" * 60)
    
    async with EmbeddingServerTester(base_url) as tester:
        
        # 1. Health Check
        print("\\n1️⃣ Health Check")
        health = await tester.health_check()
        if health["status"] == "healthy":
            print(f"✅ Server is healthy (response time: {health['response_time']:.3f}s)")
        else:
            print(f"❌ Server is unhealthy: {health.get('error', 'Unknown error')}")
            return
        
        # 2. Single Text Test
        print("\\n2️⃣ Single Text Embedding Test")
        single_result = await tester.single_embedding_test(TEST_TEXTS[0])
        if single_result["success"]:
            print(f"✅ Single embedding successful")
            print(f"   📐 Dimension: {single_result['embedding_dim']}")
            print(f"   ⏱️ Response time: {single_result['response_time']:.3f}s")
            print(f"   📝 Text length: {single_result['text_length']} chars")
        else:
            print(f"❌ Single embedding failed: {single_result.get('error')}")
            return
        
        # 3. Multiple Texts Test
        print("\\n3️⃣ Multiple Texts Test")
        batch_results = await tester.batch_embedding_test(TEST_TEXTS[:10], max_concurrent=5)
        successful_batch = [r for r in batch_results if r.get("success", False)]
        print(f"✅ Batch test: {len(successful_batch)}/{len(batch_results)} successful")
        
        if successful_batch:
            avg_time = statistics.mean([r["response_time"] for r in successful_batch])
            dimensions = [r["embedding_dim"] for r in successful_batch]
            print(f"   ⏱️ Average response time: {avg_time:.3f}s")
            print(f"   📐 Consistent dimensions: {len(set(dimensions)) == 1} (all {dimensions[0]}D)")
        
        # 4. Load Test
        print(f"\\n4️⃣ Load Test ({num_load_requests} requests, max {max_concurrent} concurrent)")
        load_result = await tester.load_test(
            "Sức khỏe tâm thần của học sinh là rất quan trọng trong môi trường giáo dục.",
            num_load_requests,
            max_concurrent
        )
        
        print(f"📊 Load Test Results:")
        print(f"   ✅ Successful: {load_result['successful_requests']}/{load_result['total_requests']}")
        print(f"   ❌ Failed: {load_result['failed_requests']}")
        print(f"   📈 Success rate: {load_result['success_rate']:.1%}")
        print(f"   🚀 Requests/second: {load_result['requests_per_second']:.2f}")
        print(f"   ⏱️ Avg response time: {load_result['avg_response_time']:.3f}s")
        print(f"   📊 Response time range: {load_result['min_response_time']:.3f}s - {load_result['max_response_time']:.3f}s")
        print(f"   📊 Median response time: {load_result['median_response_time']:.3f}s")
        
        if load_result['errors']:
            print(f"   ⚠️ Errors encountered: {set(load_result['errors'])}")
        
        # 5. Performance Assessment
        print("\\n5️⃣ Performance Assessment")
        rps = load_result['requests_per_second']
        avg_time = load_result['avg_response_time']
        success_rate = load_result['success_rate']
        
        print(f"🎯 Overall Performance:")
        if rps >= 10 and avg_time <= 1.0 and success_rate >= 0.95:
            print("   🟢 EXCELLENT - High throughput, low latency, high reliability")
        elif rps >= 5 and avg_time <= 2.0 and success_rate >= 0.90:
            print("   🟡 GOOD - Adequate performance for production")
        elif rps >= 1 and avg_time <= 5.0 and success_rate >= 0.80:
            print("   🟠 FAIR - Usable but may need optimization")
        else:
            print("   🔴 POOR - Needs significant optimization")
        
        print(f"\\n💡 Recommendations:")
        if avg_time > 2.0:
            print("   • Consider model optimization or GPU acceleration")
        if rps < 5:
            print("   • Consider scaling horizontally or optimizing batch processing")
        if success_rate < 0.95:
            print("   • Investigate error patterns and improve error handling")

def main():
    parser = argparse.ArgumentParser(description="Test embedding server performance")
    parser.add_argument("--url", default="http://localhost:5000", 
                       help="Embedding server URL")
    parser.add_argument("--requests", type=int, default=50,
                       help="Number of requests for load test")
    parser.add_argument("--concurrent", type=int, default=10,
                       help="Max concurrent requests")
    
    args = parser.parse_args()
    
    asyncio.run(run_comprehensive_test(
        args.url, 
        args.requests, 
        args.concurrent
    ))

if __name__ == "__main__":
    main()
