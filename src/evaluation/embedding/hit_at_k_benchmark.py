#!/usr/bin/env python3
"""
Hit@K Benchmark for Embedding Models
Test embedding models using Hit@1, Hit@4 metrics on ViSTS dataset
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    # Try to import local config first
    from config import *
except ImportError:
    # Fallback to parent directory config
    sys.path.append('../../')
    from config import Config

class HitAtKBenchmark:
    def __init__(self):
        """
        Initialize Hit@K Benchmark với ViSTS dataset
        """
        print("🎯 Khởi tạo Hit@K Benchmark...")
        # Use models from config
        self.models_to_test = HIT_AT_K_MODELS
        self.results = {}
        self.dataset = None
        
    def load_vists_dataset(self) -> Dict:
        """
        Load ViSTS dataset từ HuggingFace
        """
        print("📚 Loading ViSTS dataset...")
        
        try:
            # Load all subsets của ViSTS
            try:
                datasets = VISTS_DATASETS
            except NameError:
                datasets = ['STS-B', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS-Sickr']
            all_data = []
            
            for dataset_name in datasets:
                print(f"   Loading {dataset_name}...")
                try:
                    sts_data = load_dataset("anti-ai/ViSTS", dataset_name)["test"]
                    
                    for item in sts_data:
                        all_data.append({
                            'sentence1': item['sentence1'],
                            'sentence2': item['sentence2'], 
                            'score': float(item['score']) / 5.0,  # Normalize 0-1
                            'dataset': dataset_name
                        })
                        
                except Exception as e:
                    print(f"   ⚠️  Lỗi loading {dataset_name}: {e}")
                    continue
            
            print(f"✅ Loaded {len(all_data)} sentence pairs from {len(datasets)} datasets")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Filter high similarity pairs (score >= 0.6) for positive examples
            positive_pairs = df[df['score'] >= 0.6].copy()
            print(f"📊 Positive pairs (score >= 0.6): {len(positive_pairs)}")
            
            self.dataset = {
                'all_data': df,
                'positive_pairs': positive_pairs,
                'total_pairs': len(all_data)
            }
            
            return self.dataset
            
        except Exception as e:
            print(f"❌ Lỗi loading ViSTS dataset: {e}")
            return None
    
    def load_model_safe(self, model_name: str) -> Optional[SentenceTransformer]:
        """
        Load model với error handling
        """
        try:
            print(f"   Loading {model_name}...")
            
            # Try normal loading first
            try:
                model = SentenceTransformer(model_name)
            except ValueError as ve:
                if "trust_remote_code" in str(ve):
                    print(f"   ⚠️  Requires trust_remote_code=True, retrying...")
                    model = SentenceTransformer(model_name, trust_remote_code=True)
                else:
                    raise ve
            
            print(f"   ✅ Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
            return None
    
    def compute_hit_at_k(self, query_embeddings: np.ndarray, 
                        candidate_embeddings: np.ndarray, 
                        true_indices: List[int], 
                        k_values: List[int] = [1, 4]) -> Dict[int, float]:
        """
        Compute Hit@K metrics
        
        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            candidate_embeddings: Candidate embeddings (n_candidates, dim) 
            true_indices: True positive indices for each query
            k_values: K values to compute Hit@K for
            
        Returns:
            Dict with Hit@K scores
        """
        n_queries = len(query_embeddings)
        hit_scores = {k: 0 for k in k_values}
        
        # Compute similarities
        similarities = cosine_similarity(query_embeddings, candidate_embeddings)
        
        for i in range(n_queries):
            # Get similarity scores for query i
            query_similarities = similarities[i]
            
            # Get top-k indices (sorted by similarity descending)
            top_k_indices = np.argsort(query_similarities)[::-1]
            
            # Check if true positive is in top-k for each k
            true_idx = true_indices[i]
            
            for k in k_values:
                if true_idx in top_k_indices[:k]:
                    hit_scores[k] += 1
        
        # Convert to percentages
        for k in k_values:
            hit_scores[k] = (hit_scores[k] / n_queries) * 100
            
        return hit_scores
    
    def prepare_retrieval_task(self, data: pd.DataFrame, sample_size: int = 1000) -> Tuple[List[str], List[str], List[int]]:
        """
        Prepare retrieval task từ ViSTS data
        
        Returns:
            queries: List of query sentences
            candidates: List of candidate sentences  
            true_indices: True positive indices for each query
        """
        print(f"🔄 Preparing retrieval task with {sample_size} samples...")
        
        # Sample data for faster testing
        if len(data) > sample_size:
            sampled_data = data.sample(n=sample_size, random_state=42)
        else:
            sampled_data = data.copy()
        
        queries = []
        candidates = []
        true_indices = []
        
        # Create candidate pool from all sentence2s
        all_sentence2s = sampled_data['sentence2'].tolist()
        
        for idx, row in sampled_data.iterrows():
            query = row['sentence1']
            true_candidate = row['sentence2']
            
            # Find index of true candidate in candidate pool
            try:
                true_idx = all_sentence2s.index(true_candidate)
            except ValueError:
                # If not found, skip this sample
                continue
                
            queries.append(query)
            true_indices.append(true_idx)
        
        candidates = all_sentence2s
        
        print(f"✅ Prepared {len(queries)} queries with {len(candidates)} candidates")
        return queries, candidates, true_indices
    
    def test_model(self, model_name: str, model_info: Dict) -> Dict:
        """
        Test một model với Hit@K metrics
        """
        print(f"\n🧪 Testing {model_info['name']} ({model_name})")
        
        # Load model
        model = self.load_model_safe(model_name)
        if model is None:
            return {
                'model_name': model_name,
                'display_name': model_info['name'],
                'status': 'failed',
                'error': 'Failed to load model'
            }
        
        try:
            # Prepare data
            positive_data = self.dataset['positive_pairs']
            try:
                sample_size = RETRIEVAL_SAMPLE_SIZE
            except NameError:
                sample_size = 500
            queries, candidates, true_indices = self.prepare_retrieval_task(positive_data, sample_size=sample_size)
            
            if len(queries) == 0:
                return {
                    'model_name': model_name,
                    'display_name': model_info['name'], 
                    'status': 'failed',
                    'error': 'No valid query-candidate pairs'
                }
            
            # Encode queries and candidates
            print(f"   Encoding {len(queries)} queries...")
            start_time = time.time()
            query_embeddings = model.encode(queries, convert_to_numpy=True, show_progress_bar=False)
            query_time = time.time() - start_time
            
            print(f"   Encoding {len(candidates)} candidates...")
            start_time = time.time()
            candidate_embeddings = model.encode(candidates, convert_to_numpy=True, show_progress_bar=False)
            candidate_time = time.time() - start_time
            
            # Compute Hit@K
            print(f"   Computing Hit@K metrics...")
            try:
                k_values = HIT_AT_K_VALUES
            except NameError:
                k_values = [1, 4, 10]
            
            hit_scores = self.compute_hit_at_k(
                query_embeddings, 
                candidate_embeddings, 
                true_indices,
                k_values=k_values
            )
            
            total_time = query_time + candidate_time
            
            print(f"   ✅ Results: Hit@1={hit_scores[1]:.1f}%, Hit@4={hit_scores[4]:.1f}%, Hit@10={hit_scores[10]:.1f}%")
            print(f"   ⏱️  Total time: {total_time:.2f}s")
            
            return {
                'model_name': model_name,
                'display_name': model_info['name'],
                'hit_at_1': hit_scores[1],
                'hit_at_4': hit_scores[4], 
                'hit_at_10': hit_scores[10],
                'query_time': query_time,
                'candidate_time': candidate_time,
                'total_time': total_time,
                'n_queries': len(queries),
                'n_candidates': len(candidates),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"   ❌ Error testing model: {e}")
            return {
                'model_name': model_name,
                'display_name': model_info['name'],
                'status': 'failed', 
                'error': str(e)
            }
    
    def run_benchmark(self) -> Dict:
        """
        Chạy benchmark cho tất cả models
        """
        print("🚀 Starting Hit@K Benchmark...")
        
        # Load dataset
        if not self.load_vists_dataset():
            print("❌ Cannot load dataset, aborting benchmark")
            return {}
        
        # Test each model
        for model_name, model_info in self.models_to_test.items():
            result = self.test_model(model_name, model_info)
            self.results[model_name] = result
        
        return self.results
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame
        """
        rows = []
        
        for model_name, result in self.results.items():
            if result['status'] == 'success':
                rows.append({
                    'Model': result['display_name'],
                    'Model Path': model_name,
                    'Hit@1': result['hit_at_1'],
                    'Hit@4': result['hit_at_4'],
                    'Hit@10': result['hit_at_10'],
                    'Total Time (s)': result['total_time'],
                    'Queries': result['n_queries'],
                    'Candidates': result['n_candidates'],
                    'Status': 'Success'
                })
            else:
                rows.append({
                    'Model': result['display_name'],
                    'Model Path': model_name,
                    'Hit@1': 0,
                    'Hit@4': 0, 
                    'Hit@10': 0,
                    'Total Time (s)': 0,
                    'Queries': 0,
                    'Candidates': 0,
                    'Status': f"Failed: {result.get('error', 'Unknown')}"
                })
        
        return pd.DataFrame(rows)
    
    def plot_results(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive visualization với performance và timing charts
        """
        # Filter successful models
        successful_df = df[df['Status'] == 'Success'].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful models to plot")
            return
        
        # Create 2x1 subplot (vertical layout)
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('Hit@K Benchmark Results', fontsize=16, fontweight='bold')
        
        # Sort by Hit@1 for consistent ordering
        performance_sorted = successful_df.sort_values('Hit@1', ascending=True)
        timing_sorted = successful_df.sort_values('Total Time (s)', ascending=True)
        
        # 1. Performance Chart (Hit@1 and Hit@4)
        ax1 = axes[0]
        y_pos1 = np.arange(len(performance_sorted))
        
        bars1 = ax1.barh(y_pos1 - 0.2, performance_sorted['Hit@1'], 0.4, 
                        label='Hit@1', color='#FF8C00', alpha=0.8)
        bars4 = ax1.barh(y_pos1 + 0.2, performance_sorted['Hit@4'], 0.4,
                        label='Hit@4', color='#FF6347', alpha=0.8)
        
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(performance_sorted['Model'])
        ax1.set_xlabel('Score (%)')
        ax1.set_title('Performance Ranking (Hit@1 and Hit@4)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # Add value labels for performance
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
        
        for bar in bars4:
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 2. Timing Chart
        ax2 = axes[1]
        y_pos2 = np.arange(len(timing_sorted))
        
        # Color gradient based on speed (green = fast, red = slow)
        max_time = timing_sorted['Total Time (s)'].max()
        colors = plt.cm.RdYlGn_r(timing_sorted['Total Time (s)'] / max_time)
        
        bars_time = ax2.barh(y_pos2, timing_sorted['Total Time (s)'], 
                            color=colors, alpha=0.8)
        
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(timing_sorted['Model'])
        ax2.set_xlabel('Processing Time (seconds)')
        ax2.set_title('Processing Time Ranking (Lower is Better)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels for timing
        for i, bar in enumerate(bars_time):
            width = bar.get_width()
            # Color label based on speed: green for fast, red for slow
            label_color = 'green' if width < 15 else 'orange' if width < 40 else 'red'
            ax2.text(width + max_time * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}s', ha='left', va='center', fontsize=10, 
                    fontweight='bold', color=label_color)
        
        # Add speed categories as text annotations
        ax2.text(0.02, 0.98, 'Fast: <15s   Medium: 15-40s   Slow: >40s', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Chart saved to: {save_path}")
        
        plt.show()
        return fig
    
    def save_results(self, df: pd.DataFrame, timestamp: str = None):
        """
        Save results to files in results directory
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get current script directory and create results subdirectory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(results_dir, f"hit_at_k_benchmark_results.json")
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Save summary CSV
        csv_file = os.path.join(results_dir, f"hit_at_k_benchmark_summary.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save chart
        chart_file = os.path.join(results_dir, f"hit_at_k_benchmark_chart.png")
        self.plot_results(df, save_path=chart_file)
        
        print(f"💾 Results saved to results/ directory:")
        print(f"   - {os.path.relpath(results_file, script_dir)} (detailed JSON)")
        print(f"   - {os.path.relpath(csv_file, script_dir)} (summary CSV)")
        print(f"   - {os.path.relpath(chart_file, script_dir)} (chart PNG)")

def main():
    """
    Main function to run Hit@K benchmark
    """
    print("🎯 Hit@K Benchmark for Vietnamese Embedding Models")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = HitAtKBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    if not results:
        print("❌ No results to analyze")
        return
    
    # Create results DataFrame
    df = benchmark.create_results_dataframe()
    
    # Display results
    print("\n📊 BENCHMARK RESULTS:")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Show top performers
    successful_df = df[df['Status'] == 'Success']
    if len(successful_df) > 0:
        print(f"\n🏆 TOP PERFORMERS:")
        print("-" * 30)
        
        # Sort by Hit@1
        top_hit1 = successful_df.nlargest(3, 'Hit@1')
        print("📈 Best Hit@1:")
        for i, (_, row) in enumerate(top_hit1.iterrows(), 1):
            print(f"   {i}. {row['Model']}: {row['Hit@1']:.1f}%")
        
        # Sort by Hit@4  
        top_hit4 = successful_df.nlargest(3, 'Hit@4')
        print("\n📈 Best Hit@4:")
        for i, (_, row) in enumerate(top_hit4.iterrows(), 1):
            print(f"   {i}. {row['Model']}: {row['Hit@4']:.1f}%")
    
    # Save results
    benchmark.save_results(df)
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main()