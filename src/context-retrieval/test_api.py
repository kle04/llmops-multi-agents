#!/usr/bin/env python3
"""
Test script for Context Retrieval Service
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:5005"

async def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health", timeout=10.0)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed: {health_data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

async def test_collections():
    """Test collections endpoint"""
    print("\n🔍 Testing collections endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/collections", timeout=10.0)
            if response.status_code == 200:
                collections_data = response.json()
                print(f"✅ Collections: {json.dumps(collections_data, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"❌ Collections failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Collections error: {e}")
            return False

async def test_search_post(query: str, filters: Dict[str, Any] = None):
    """Test POST search endpoint"""
    print(f"\n🔍 Testing POST search with query: '{query}'...")
    async with httpx.AsyncClient() as client:
        try:
            search_data = {
                "query": query,
                "limit": 5,
                "score_threshold": 0.5,
                "filters": filters
            }
            
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/search", 
                json=search_data,
                timeout=30.0
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                search_results = response.json()
                print(f"✅ Search completed in {response_time:.2f}ms")
                print(f"📊 Found {search_results['total_found']} results")
                
                for i, result in enumerate(search_results['results'][:3], 1):
                    print(f"\n📄 Result {i} (score: {result['score']:.3f}):")
                    print(f"   Title: {result['title']}")
                    print(f"   Source: {result['source']}")
                    print(f"   Topics: {result['topics']}")
                    print(f"   Text preview: {result['text'][:200]}...")
                
                return True
            else:
                print(f"❌ Search failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False

async def test_search_get(query: str, **params):
    """Test GET search endpoint"""
    print(f"\n🔍 Testing GET search with query: '{query}'...")
    async with httpx.AsyncClient() as client:
        try:
            params['q'] = query
            params['limit'] = params.get('limit', 3)
            params['score_threshold'] = params.get('score_threshold', 0.5)
            
            start_time = time.time()
            response = await client.get(f"{BASE_URL}/search", params=params, timeout=30.0)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                search_results = response.json()
                print(f"✅ GET search completed in {response_time:.2f}ms")
                print(f"📊 Found {search_results['total_found']} results")
                
                for i, result in enumerate(search_results['results'], 1):
                    print(f"\n📄 Result {i} (score: {result['score']:.3f}):")
                    print(f"   Title: {result['title']}")
                    print(f"   Source: {result['source']}")
                    print(f"   Text preview: {result['text'][:150]}...")
                
                return True
            else:
                print(f"❌ GET search failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ GET search error: {e}")
            return False

async def test_filtered_search():
    """Test search with filters"""
    print(f"\n🔍 Testing filtered search...")
    
    # Test with topic filter
    filters = {
        "topics": "stress",
        "audience": "giao_vien"
    }
    
    return await test_search_post(
        "Làm thế nào để giúp học sinh giảm stress?", 
        filters=filters
    )

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Context Retrieval Service Tests\n")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health check
    total_tests += 1
    if await test_health_check():
        tests_passed += 1
    
    # Test 2: Collections
    total_tests += 1
    if await test_collections():
        tests_passed += 1
    
    # Test 3: Basic POST search
    total_tests += 1
    if await test_search_post("sức khỏe tâm thần học sinh"):
        tests_passed += 1
    
    # Test 4: GET search
    total_tests += 1
    if await test_search_get("stress học tập", audience="hoc_sinh_pho_thong"):
        tests_passed += 1
    
    # Test 5: Filtered search
    total_tests += 1
    if await test_filtered_search():
        tests_passed += 1
    
    # Test 6: Vietnamese query
    total_tests += 1
    if await test_search_post("tư vấn tâm lý cho học sinh"):
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the service configuration.")
        return False

if __name__ == "__main__":
    print("Context Retrieval Service Test Suite")
    print("=" * 50)
    
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        exit(1)
