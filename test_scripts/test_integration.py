#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration test script for the complete embedding + context retrieval pipeline.
Tests end-to-end functionality and performance.
"""

import asyncio
import time
import statistics
import httpx
import json
from typing import List, Dict, Any, Optional
import argparse

# Vietnamese test scenarios for comprehensive integration testing
TEST_SCENARIOS = [
    {
        "category": "mental_health",
        "queries": [
            "Học sinh bị trầm cảm cần được hỗ trợ như thế nào?",
            "Dấu hiệu nhận biết học sinh có vấn đề sức khỏe tâm thần",
            "Cách can thiệp khi học sinh có ý định tự hại"
        ]
    },
    {
        "category": "bullying",
        "queries": [
            "Quy trình xử lý vụ việc bạo lực học đường",
            "Cách hỗ trợ học sinh bị bắt nạt",
            "Phòng ngừa bạo lực trong môi trường học đường"
        ]
    },
    {
        "category": "counseling",
        "queries": [
            "Kỹ năng tư vấn tâm lý cho học sinh",
            "Cách tổ chức buổi tư vấn hiệu quả",
            "Phương pháp lắng nghe tích cực trong tư vấn"
        ]
    },
    {
        "category": "family_cooperation",
        "queries": [
            "Phối hợp giữa gia đình và nhà trường",
            "Cách giao tiếp với phụ huynh khó khăn",
            "Vai trò của gia đình trong giáo dục con em"
        ]
    },
    {
        "category": "prevention",
        "queries": [
            "Các hoạt động phòng ngừa vấn đề hành vi",
            "Xây dựng môi trường học tập tích cực",
            "Giáo dục kỹ năng sống cho học sinh"
        ]
    }
]

class IntegrationTester:
    def __init__(self, embedding_url: str = "http://localhost:5000", 
                 retrieval_url: str = "http://localhost:5005"):
        self.embedding_url = embedding_url.rstrip('/')
        self.retrieval_url = retrieval_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=60.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def check_services_health(self) -> Dict[str, Any]:
        """Check health of both services"""
        print("🏥 Checking service health...")
        
        # Check embedding service
        embedding_health = {"status": "unknown"}
        try:
            response = await self.session.get(f"{self.embedding_url}/health")
            response.raise_for_status()
            embedding_health = {
                "status": "healthy",
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            embedding_health = {"status": "unhealthy", "error": str(e)}
        
        # Check retrieval service
        retrieval_health = {"status": "unknown"}
        try:
            response = await self.session.get(f"{self.retrieval_url}/health")
            response.raise_for_status()
            retrieval_health = {
                "status": "healthy", 
                "response_time": response.elapsed.total_seconds(),
                "data": response.json()
            }
        except Exception as e:
            retrieval_health = {"status": "unhealthy", "error": str(e)}
        
        return {
            "embedding_service": embedding_health,
            "retrieval_service": retrieval_health,
            "both_healthy": (embedding_health["status"] == "healthy" and 
                           retrieval_health["status"] == "healthy")
        }
    
    async def test_embedding_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Test embedding generation quality"""
        results = []
        
        for text in texts:
            start_time = time.time()
            try:
                response = await self.session.post(
                    f"{self.embedding_url}/embed",
                    json={"text": text}
                )
                response.raise_for_status()
                result = response.json()
                
                results.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "success": True,
                    "embedding_dim": len(result.get("embedding", [])),
                    "response_time": time.time() - start_time
                })
            except Exception as e:
                results.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time
                })
        
        successful = [r for r in results if r["success"]]
        return {
            "total_tests": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results),
            "avg_response_time": statistics.mean([r["response_time"] for r in successful]) if successful else 0,
            "consistent_dimensions": len(set(r["embedding_dim"] for r in successful)) <= 1 if successful else False,
            "embedding_dimension": successful[0]["embedding_dim"] if successful else None,
            "results": results
        }
    
    async def test_retrieval_quality(self, queries: List[str]) -> Dict[str, Any]:
        """Test retrieval quality and relevance"""
        results = []
        
        for query in queries:
            start_time = time.time()
            try:
                response = await self.session.post(
                    f"{self.retrieval_url}/search",
                    json={"query": query, "limit": 5}
                )
                response.raise_for_status()
                result = response.json()
                
                search_results = result.get("results", [])
                results.append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "success": True,
                    "num_results": len(search_results),
                    "response_time": time.time() - start_time,
                    "avg_score": statistics.mean([r.get("score", 0) for r in search_results]) if search_results else 0,
                    "min_score": min([r.get("score", 0) for r in search_results]) if search_results else 0,
                    "has_relevant_content": len(search_results) > 0 and search_results[0].get("score", 0) > 0.5
                })
            except Exception as e:
                results.append({
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time
                })
        
        successful = [r for r in results if r["success"]]
        return {
            "total_queries": len(results),
            "successful": len(successful),
            "success_rate": len(successful) / len(results),
            "avg_response_time": statistics.mean([r["response_time"] for r in successful]) if successful else 0,
            "avg_results_per_query": statistics.mean([r["num_results"] for r in successful]) if successful else 0,
            "avg_relevance_score": statistics.mean([r["avg_score"] for r in successful]) if successful else 0,
            "queries_with_relevant_results": sum(1 for r in successful if r["has_relevant_content"]),
            "results": results
        }
    
    async def test_end_to_end_scenarios(self) -> Dict[str, Any]:
        """Test complete scenarios by category"""
        scenario_results = {}
        
        for scenario in TEST_SCENARIOS:
            category = scenario["category"]
            queries = scenario["queries"]
            
            print(f"📋 Testing {category} scenario...")
            
            # Test all queries in this category
            category_result = await self.test_retrieval_quality(queries)
            scenario_results[category] = category_result
            
            # Brief pause between categories
            await asyncio.sleep(0.5)
        
        return scenario_results
    
    async def test_concurrent_load(self, num_requests: int = 20, max_concurrent: int = 5) -> Dict[str, Any]:
        """Test concurrent load on the complete pipeline"""
        print(f"⚡ Running concurrent load test ({num_requests} requests, {max_concurrent} concurrent)...")
        
        # Use a mix of queries from different categories
        all_queries = []
        for scenario in TEST_SCENARIOS:
            all_queries.extend(scenario["queries"])
        
        # Repeat queries to reach desired number of requests
        test_queries = (all_queries * ((num_requests // len(all_queries)) + 1))[:num_requests]
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_search(query: str) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                try:
                    response = await self.session.post(
                        f"{self.retrieval_url}/search",
                        json={"query": query, "limit": 3}
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    return {
                        "success": True,
                        "response_time": time.time() - start_time,
                        "num_results": len(result.get("results", []))
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time
                    }
        
        start_time = time.time()
        results = await asyncio.gather(*[bounded_search(q) for q in test_queries])
        total_time = time.time() - start_time
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful]
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "total_time": total_time,
            "requests_per_second": len(successful) / total_time if total_time > 0 else 0,
            "success_rate": len(successful) / num_requests,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0,
            "errors": [r.get("error") for r in failed]
        }

async def run_integration_tests(embedding_url: str, retrieval_url: str, 
                              load_requests: int = 20, max_concurrent: int = 5):
    """Run complete integration test suite"""
    print("🚀 INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"📊 Embedding Service: {embedding_url}")
    print(f"🔍 Retrieval Service: {retrieval_url}")
    print(f"⚡ Load Test: {load_requests} requests, {max_concurrent} concurrent")
    
    async with IntegrationTester(embedding_url, retrieval_url) as tester:
        
        # 1. Health Check
        print("\\n1️⃣ Service Health Check")
        health = await tester.check_services_health()
        
        print(f"   📊 Embedding Service: {health['embedding_service']['status']}")
        if health['embedding_service']['status'] == 'healthy':
            print(f"      ⏱️ Response time: {health['embedding_service']['response_time']:.3f}s")
        else:
            print(f"      ❌ Error: {health['embedding_service'].get('error', 'Unknown')}")
        
        print(f"   🔍 Retrieval Service: {health['retrieval_service']['status']}")
        if health['retrieval_service']['status'] == 'healthy':
            print(f"      ⏱️ Response time: {health['retrieval_service']['response_time']:.3f}s")
            if health['retrieval_service'].get('data'):
                print(f"      📋 Info: {health['retrieval_service']['data']}")
        else:
            print(f"      ❌ Error: {health['retrieval_service'].get('error', 'Unknown')}")
        
        if not health['both_healthy']:
            print("\\n❌ Cannot proceed - one or more services are unhealthy")
            return
        
        # 2. Embedding Quality Test
        print("\\n2️⃣ Embedding Quality Test")
        sample_texts = [q for scenario in TEST_SCENARIOS[:2] for q in scenario["queries"][:2]]
        embedding_result = await tester.test_embedding_quality(sample_texts)
        
        print(f"   ✅ Success rate: {embedding_result['success_rate']:.1%}")
        print(f"   📐 Embedding dimension: {embedding_result['embedding_dimension']}")
        print(f"   🔄 Consistent dimensions: {embedding_result['consistent_dimensions']}")
        print(f"   ⏱️ Avg response time: {embedding_result['avg_response_time']:.3f}s")
        
        # 3. Retrieval Quality Test
        print("\\n3️⃣ Retrieval Quality Test")
        sample_queries = [q for scenario in TEST_SCENARIOS[:3] for q in scenario["queries"][:1]]
        retrieval_result = await tester.test_retrieval_quality(sample_queries)
        
        print(f"   ✅ Success rate: {retrieval_result['success_rate']:.1%}")
        print(f"   📊 Avg results per query: {retrieval_result['avg_results_per_query']:.1f}")
        print(f"   🎯 Avg relevance score: {retrieval_result['avg_relevance_score']:.4f}")
        print(f"   📋 Relevant results: {retrieval_result['queries_with_relevant_results']}/{retrieval_result['successful']}")
        print(f"   ⏱️ Avg response time: {retrieval_result['avg_response_time']:.3f}s")
        
        # 4. End-to-End Scenario Tests
        print("\\n4️⃣ End-to-End Scenario Tests")
        scenario_results = await tester.test_end_to_end_scenarios()
        
        overall_success = 0
        overall_total = 0
        overall_relevance = []
        
        for category, result in scenario_results.items():
            success_rate = result['success_rate']
            avg_score = result['avg_relevance_score']
            overall_success += result['successful']
            overall_total += result['total_queries']
            if result['successful'] > 0:
                overall_relevance.append(avg_score)
            
            print(f"   📋 {category}: {success_rate:.1%} success, {avg_score:.3f} avg score")
        
        overall_success_rate = overall_success / overall_total if overall_total > 0 else 0
        overall_avg_relevance = statistics.mean(overall_relevance) if overall_relevance else 0
        
        print(f"   📊 Overall: {overall_success_rate:.1%} success, {overall_avg_relevance:.3f} avg relevance")
        
        # 5. Concurrent Load Test
        print("\\n5️⃣ Concurrent Load Test")
        load_result = await tester.test_concurrent_load(load_requests, max_concurrent)
        
        print(f"   ✅ Success rate: {load_result['success_rate']:.1%}")
        print(f"   🚀 Requests/second: {load_result['requests_per_second']:.2f}")
        print(f"   ⏱️ Avg response time: {load_result['avg_response_time']:.3f}s")
        print(f"   📊 Median response time: {load_result['median_response_time']:.3f}s")
        print(f"   📈 P95 response time: {load_result['p95_response_time']:.3f}s")
        
        if load_result['errors']:
            unique_errors = set(str(e) for e in load_result['errors'] if e)
            print(f"   ⚠️ Unique errors: {len(unique_errors)}")
        
        # 6. Overall Assessment
        print("\\n6️⃣ Overall System Assessment")
        
        # Calculate overall score
        scores = {
            "embedding_quality": 1.0 if embedding_result['success_rate'] >= 0.95 and embedding_result['consistent_dimensions'] else 0.5 if embedding_result['success_rate'] >= 0.8 else 0.0,
            "retrieval_quality": 1.0 if retrieval_result['success_rate'] >= 0.95 and retrieval_result['avg_relevance_score'] >= 0.6 else 0.5 if retrieval_result['success_rate'] >= 0.8 else 0.0,
            "scenario_coverage": 1.0 if overall_success_rate >= 0.95 and overall_avg_relevance >= 0.6 else 0.5 if overall_success_rate >= 0.8 else 0.0,
            "load_performance": 1.0 if load_result['success_rate'] >= 0.95 and load_result['requests_per_second'] >= 2 else 0.5 if load_result['success_rate'] >= 0.8 else 0.0
        }
        
        overall_score = statistics.mean(scores.values())
        
        print(f"🎯 System Scores:")
        for component, score in scores.items():
            status = "🟢" if score >= 0.8 else "🟡" if score >= 0.5 else "🔴"
            print(f"   {status} {component.replace('_', ' ').title()}: {score:.1f}/1.0")
        
        print(f"\\n📊 Overall Score: {overall_score:.2f}/1.0")
        
        if overall_score >= 0.8:
            print("🟢 EXCELLENT - System ready for production")
        elif overall_score >= 0.6:
            print("🟡 GOOD - System functional with minor optimization needed")
        elif overall_score >= 0.4:
            print("🟠 FAIR - System needs significant optimization")
        else:
            print("🔴 POOR - System requires major fixes")
        
        # Recommendations
        print("\\n💡 Recommendations:")
        if embedding_result['avg_response_time'] > 1.0:
            print("   • Optimize embedding service response time")
        if retrieval_result['avg_relevance_score'] < 0.6:
            print("   • Review embedding quality or search parameters")
        if load_result['requests_per_second'] < 2:
            print("   • Consider horizontal scaling or performance optimization")
        if load_result['success_rate'] < 0.95:
            print("   • Improve error handling and system reliability")

def main():
    parser = argparse.ArgumentParser(description="Run integration tests for embedding + retrieval pipeline")
    parser.add_argument("--embedding-url", default="http://localhost:5000", 
                       help="Embedding service URL")
    parser.add_argument("--retrieval-url", default="http://localhost:5005",
                       help="Context retrieval service URL") 
    parser.add_argument("--load-requests", type=int, default=20,
                       help="Number of requests for load test")
    parser.add_argument("--max-concurrent", type=int, default=5,
                       help="Max concurrent requests for load test")
    
    args = parser.parse_args()
    
    asyncio.run(run_integration_tests(
        args.embedding_url,
        args.retrieval_url, 
        args.load_requests,
        args.max_concurrent
    ))

if __name__ == "__main__":
    main()
