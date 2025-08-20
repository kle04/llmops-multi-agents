#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple pipeline: Embedding generation -> Qdrant upsert
"""

import os
import sys
import time
import subprocess

def main():
    print("🚀 Starting Embedding + Qdrant Pipeline")
    
    # Environment variables with defaults
    chunks_dir = os.getenv("CHUNKS_DIR", "/workspace/data/processed/chunks")
    out_dir = os.getenv("OUT_DIR", "/workspace/data/embeddings")
    model = os.getenv("MODEL", "AITeamVN/Vietnamese_Embedding_v2")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "mental_health_vi")
    
    print(f"Chunks dir: {chunks_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Model: {model}")
    print(f"Qdrant: {qdrant_url}")
    print(f"Collection: {collection}")
    
    start_time = time.time()
    
    try:
        # Step 1: Generate embeddings
        print("\n=== Step 1: Generate Embeddings ===")
        embed_cmd = [
            "python", "/workspace/src/02_embedding/03_embedding.py",
            "--chunks_dir", chunks_dir,
            "--out_dir", out_dir,
            "--model", model,
            "--normalize"
        ]
        
        result = subprocess.run(embed_cmd, check=True)
        print("✅ Embedding generation completed")
        
        # Step 2: Upsert to Qdrant
        print("\n=== Step 2: Upsert to Qdrant ===")
        upsert_cmd = [
            "python", "/workspace/src/02_embedding/03b_upsert_qdrant.py",
            "--emb_dir", out_dir,
            "--chunks_dir", chunks_dir,
            "--qdrant_url", qdrant_url,
            "--collection", collection
        ]
        
        result = subprocess.run(upsert_cmd, check=True)
        print("✅ Qdrant upsert completed")
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 Pipeline completed in {total_time:.1f} seconds!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
