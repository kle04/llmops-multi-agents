#!/usr/bin/env python3
"""
Configuration for Embedding Benchmarks
"""

# =============================================================================
# BENCHMARK SETTINGS
# =============================================================================

# Hit@K settings
HIT_AT_K_VALUES = [1, 4, 10]
RETRIEVAL_SAMPLE_SIZE = 500

# STS Correlation settings
STS_SAMPLE_SIZE = 1000

# ViSTS dataset subsets
VISTS_DATASETS = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-Sickr']

# =============================================================================
# MODEL LISTS
# =============================================================================

# All models for benchmarking
HIT_AT_K_MODELS = {
    "intfloat/multilingual-e5-base": {
        "name": "intfloat/multilingual-e5-base"
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "name": "Alibaba-NLP/gte-multilingual-base"
    },
    "keepitreal/vietnamese-sbert": {
        "name": "keepitreal/vietnamese-sbert"
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "name": "Qwen/Qwen3-Embedding-0.6B"
    },
    "BAAI/bge-m3": {
        "name": "BAAI/bge-m3"
    },
    "intfloat/multilingual-e5-large-instruct": {
        "name": "intfloat/multilingual-e5-large-instruct"
    }
}

# Models for STS
STS_CORRELATION_MODELS = HIT_AT_K_MODELS


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("📊 Embedding Benchmark Configuration")
    print("=" * 40)
    print(f"Available Models: {len(HIT_AT_K_MODELS)}")
    print(f"Quick Test Models: {len(QUICK_TEST_MODELS)}")
    print(f"Hit@K Values: {HIT_AT_K_VALUES}")
    print(f"Sample Sizes: Hit@K={RETRIEVAL_SAMPLE_SIZE}, STS={STS_SAMPLE_SIZE}")
    
    print("\n🤖 All Models:")
    for model_path, model_info in HIT_AT_K_MODELS.items():
        print(f"   - {model_info['name']}")
    
    print("\n⚡ Quick Test:")
    for model_path, model_info in QUICK_TEST_MODELS.items():
        print(f"   - {model_info['name']}")