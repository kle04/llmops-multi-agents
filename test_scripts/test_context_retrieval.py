#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for context retrieval server performance and functionality.
"""

import asyncio
import time
import statistics
import httpx
import json
from typing import List, Dict, Any, Optional
import argparse

# Test queries - Vietnamese educational/mental health questions
TEST_QUERIES = [
    "Làm thế nào để hỗ trợ học sinh có vấn đề về sức khỏe tâm thần?",
    "Cách phòng ngừa bạo lực học đường hiệu quả",
    "Quy trình tư vấn tâm lý cho học sinh trong trường học",
    "Vai trò của gia đình trong việc hỗ trợ học sinh",
    "Các dấu hiệu nhận biết học sinh gặp khó khăn",
    "Phương pháp can thiệp khi học sinh có hành vi bất thường",
    "Cách xây dựng môi trường học tập tích cực",
    "Kỹ năng giao tiếp với học sinh gặp vấn đề",
    "Hướng dẫn hỗ trợ học sinh bị bắt nạt",
    "Cách phối hợp giữa nhà trường và gia đình",
    "Các hoạt động phòng ngừa vấn đề hành vi",
    "Phương pháp đánh giá sức khỏe tâm thần học sinh",
    "Cách xử lý tình huống khủng hoảng tâm lý",
    "Vai trò của giáo viên trong công tác xã hội trường học",
    "Hướng dẫn tổ chức các hoạt động hỗ trợ tâm lý",
]

class ContextRetrievalTester:
    def __init__(self, base_url: str = "http://localhost:5005"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=60.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if context retrieval server is healthy"""
        try:
            response = await self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return {
                "status": "healthy",
                "response_time": response.elapsed.total_seconds(),
                "status_code": response.status_code,
                "data": response.json()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "response_time": None,
                "status_code": None
            }
    
    async def single_search_test(self, query: str, limit: int = 5, 
                               filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Test single search query"""
        start_time = time.time()
        try:
            payload = {
                "query": query,
                "limit": limit
            }
            if filters:
                payload["filters"] = filters
            
            response = await self.session.post(
                f"{self.base_url}/search",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            results = result.get("results", [])
            
            return {
                "success": True,
                "response_time": time.time() - start_time,
                "num_results": len(results),
                "query_length": len(query),
                "status_code": response.status_code,
                "has_scores": all("score" in r for r in results),
                "avg_score": statistics.mean([r.get("score", 0) for r in results]) if results else 0,
                "result_metadata": {
                    "has_text": all("text" in r.get("payload", {}) for r in results),
                    "has_doc_id": all("doc_id" in r.get("payload", {}) for r in results),
                    "has_source": all("source" in r.get("payload", {}) for r in results),
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "query_length": len(query)
            }
    
    async def batch_search_test(self, queries: List[str], max_concurrent: int = 3,
                              limit: int = 5) -> List[Dict[str, Any]]:
        """Test multiple search queries with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_search_test(query: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.single_search_test(query, limit)
        
        tasks = [bounded_search_test(query) for query in queries]
        return await asyncio.gather(*tasks)
    
    async def filter_test(self, query: str) -> Dict[str, Any]:
        """Test search with various filters"""
        filters_to_test = [
            {},  # No filter
            {"doc_id": "MOET_SoTay_ThucHanh_CTXH_TrongTruongHoc_vi"},
            {"language": "vi"},
            {"source": "Bộ GD&ĐT (MOET)"},
        ]
        
        results = []
        for i, filters in enumerate(filters_to_test):
            filter_name = f"filter_{i}" if i > 0 else "no_filter"
            result = await self.single_search_test(query, limit=3, filters=filters)
            result["filter_name"] = filter_name
            result["filter_applied"] = filters
            results.append(result)
        
        return {
            "query": query,
            "filter_tests": results,
            "all_successful": all(r["success"] for r in results)
        }
    
    async def load_test(self, query: str, num_requests: int, max_concurrent: int = 5) -> Dict[str, Any]:
        """Load test with same query repeated"""
        start_time = time.time()
        
        results = await self.batch_search_test([query] * num_requests, max_concurrent)
        
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
            "avg_results_returned": statistics.mean([r["num_results"] for r in successful]) if successful else 0,
            "errors": [r.get("error") for r in failed]
        }

async def run_comprehensive_test(base_url: str, num_load_requests: int = 30, max_concurrent: int = 5):
    """Run comprehensive test suite"""
    print(f"🔍 Testing Context Retrieval Server at {base_url}")
    print("=" * 60)
    
    async with ContextRetrievalTester(base_url) as tester:
        
        # 1. Health Check
        print("\\n1️⃣ Health Check")
        health = await tester.health_check()
        if health["status"] == "healthy":
            print(f"✅ Server is healthy (response time: {health['response_time']:.3f}s)")
            if health.get("data"):
                print(f"   📊 Server info: {health['data']}")
        else:
            print(f"❌ Server is unhealthy: {health.get('error', 'Unknown error')}")
            return
        
        # 2. Single Search Test
        print("\\n2️⃣ Single Search Test")
        single_result = await tester.single_search_test(TEST_QUERIES[0], limit=5)
        if single_result["success"]:
            print(f"✅ Single search successful")
            print(f"   📝 Query length: {single_result['query_length']} chars")
            print(f"   📊 Results returned: {single_result['num_results']}")
            print(f"   ⏱️ Response time: {single_result['response_time']:.3f}s")
            print(f"   🎯 Average score: {single_result['avg_score']:.4f}")
            print(f"   ✅ Has scores: {single_result['has_scores']}")
            metadata = single_result['result_metadata']
            print(f"   📋 Metadata complete: text={metadata['has_text']}, doc_id={metadata['has_doc_id']}, source={metadata['has_source']}")
        else:
            print(f"❌ Single search failed: {single_result.get('error')}")
            return
        
        # 3. Multiple Queries Test
        print("\\n3️⃣ Multiple Queries Test")
        batch_results = await tester.batch_search_test(TEST_QUERIES[:8], max_concurrent=3)
        successful_batch = [r for r in batch_results if r.get("success", False)]
        print(f"✅ Batch test: {len(successful_batch)}/{len(batch_results)} successful")
        
        if successful_batch:
            avg_time = statistics.mean([r["response_time"] for r in successful_batch])
            avg_results = statistics.mean([r["num_results"] for r in successful_batch])
            avg_scores = statistics.mean([r["avg_score"] for r in successful_batch])
            print(f"   ⏱️ Average response time: {avg_time:.3f}s")
            print(f"   📊 Average results per query: {avg_results:.1f}")
            print(f"   🎯 Average relevance score: {avg_scores:.4f}")
        
        # 4. Filter Test
        print("\\n4️⃣ Filter Functionality Test")
        filter_result = await tester.filter_test(TEST_QUERIES[1])
        if filter_result["all_successful"]:
            print("✅ All filter tests successful")
            for test in filter_result["filter_tests"]:
                filter_name = test["filter_name"]
                num_results = test["num_results"]
                response_time = test["response_time"]
                print(f"   📋 {filter_name}: {num_results} results in {response_time:.3f}s")
        else:
            print("❌ Some filter tests failed")
        
        # 5. Load Test
        print(f"\\n5️⃣ Load Test ({num_load_requests} requests, max {max_concurrent} concurrent)")
        load_result = await tester.load_test(
            "Cách hỗ trợ học sinh có vấn đề sức khỏe tâm thần trong trường học",
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
        print(f"   📋 Avg results per query: {load_result['avg_results_returned']:.1f}")
        
        if load_result['errors']:
            print(f"   ⚠️ Errors encountered: {set(load_result['errors'])}")
        
        # 6. Performance Assessment
        print("\\n6️⃣ Performance Assessment")
        rps = load_result['requests_per_second']
        avg_time = load_result['avg_response_time']
        success_rate = load_result['success_rate']
        avg_results = load_result['avg_results_returned']
        
        print(f"🎯 Overall Performance:")
        if rps >= 5 and avg_time <= 2.0 and success_rate >= 0.95 and avg_results >= 3:
            print("   🟢 EXCELLENT - High throughput, low latency, reliable, good recall")
        elif rps >= 2 and avg_time <= 4.0 and success_rate >= 0.90 and avg_results >= 2:
            print("   🟡 GOOD - Adequate performance for production")
        elif rps >= 1 and avg_time <= 8.0 and success_rate >= 0.80 and avg_results >= 1:
            print("   🟠 FAIR - Usable but may need optimization")
        else:
            print("   🔴 POOR - Needs significant optimization")
        
        print(f"\\n💡 Recommendations:")
        if avg_time > 3.0:
            print("   • Consider Qdrant optimization or faster embedding service")
        if rps < 2:
            print("   • Consider scaling or optimizing vector search parameters")
        if success_rate < 0.95:
            print("   • Investigate error patterns and improve error handling")
        if avg_results < 2:
            print("   • Review embedding quality or increase search limit")

def main():
    parser = argparse.ArgumentParser(description="Test context retrieval server performance")
    parser.add_argument("--url", default="http://localhost:5005", 
                       help="Context retrieval server URL")
    parser.add_argument("--requests", type=int, default=30,
                       help="Number of requests for load test")
    parser.add_argument("--concurrent", type=int, default=5,
                       help="Max concurrent requests")
    
    args = parser.parse_args()
    
    asyncio.run(run_comprehensive_test(
        args.url, 
        args.requests, 
        args.concurrent
    ))

if __name__ == "__main__":
    main()
