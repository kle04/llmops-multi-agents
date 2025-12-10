#!/usr/bin/env python3
"""
STS Correlation Benchmark for Embedding Models
Test embedding models using Pearson & Spearman correlation on ViSTS dataset
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
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from scipy.stats import pearsonr, spearmanr
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
    try:
        from config import Config
    except ImportError:
        pass  # Will use hardcoded values

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class STSCorrelationBenchmark:
    def __init__(self):
        """
        Initialize STS Correlation Benchmark với ViSTS dataset
        """
        print("📊 Khởi tạo STS Correlation Benchmark...")
        
        # Use models from config
        self.models_to_test = STS_CORRELATION_MODELS
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
                            'score': float(item['score']),  # Keep original 1-5 scale
                            'normalized_score': float(item['score']) / 5.0,  # 0-1 scale
                            'dataset': dataset_name
                        })
                        
                except Exception as e:
                    print(f"   ⚠️  Lỗi loading {dataset_name}: {e}")
                    continue
            
            print(f"✅ Loaded {len(all_data)} sentence pairs from {len(datasets)} datasets")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            self.dataset = {
                'all_data': df,
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
    
    def compute_correlations(self, embeddings1: np.ndarray, embeddings2: np.ndarray, 
                           true_scores: List[float]) -> Dict[str, float]:
        """
        Compute Pearson and Spearman correlations
        """
        # Compute cosine similarities
        cosine_sims = []
        for i in range(len(embeddings1)):
            sim = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
            cosine_sims.append(sim)
        
        cosine_sims = np.array(cosine_sims)
        true_scores = np.array(true_scores)
        
        # Compute correlations
        pearson_corr, _ = pearsonr(cosine_sims, true_scores)
        spearman_corr, _ = spearmanr(cosine_sims, true_scores)
        
        return {
            'cosine_pearson': pearson_corr,
            'cosine_spearman': spearman_corr,
            'cosine_similarities': cosine_sims.tolist(),
            'true_scores': true_scores.tolist()
        }
    
    def test_model(self, model_name: str, model_info: Dict) -> Dict:
        """
        Test một model với STS correlation metrics
        """
        print(f"\n📊 Testing {model_info['name']} ({model_name})")
        
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
            # Sample data for faster testing
            all_data = self.dataset['all_data']
            try:
                sample_size = STS_SAMPLE_SIZE
            except NameError:
                sample_size = 1000  # Sample for faster testing
            
            if len(all_data) > sample_size:
                sampled_data = all_data.sample(n=sample_size, random_state=42)
            else:
                sampled_data = all_data.copy()
            
            print(f"   Using {len(sampled_data)} sentence pairs")
            
            # Extract sentences and scores
            sentences1 = sampled_data['sentence1'].tolist()
            sentences2 = sampled_data['sentence2'].tolist()
            true_scores = sampled_data['normalized_score'].tolist()  # Use 0-1 normalized scores
            
            # Encode sentences
            print(f"   Encoding {len(sentences1)} sentence pairs...")
            start_time = time.time()
            
            embeddings1 = model.encode(sentences1, convert_to_numpy=True, show_progress_bar=False)
            embeddings2 = model.encode(sentences2, convert_to_numpy=True, show_progress_bar=False)
            
            encoding_time = time.time() - start_time
            
            # Compute correlations
            print(f"   Computing correlations...")
            correlations = self.compute_correlations(embeddings1, embeddings2, true_scores)
            
            pearson_score = correlations['cosine_pearson'] * 100  # Convert to percentage
            spearman_score = correlations['cosine_spearman'] * 100
            
            print(f"   ✅ Results: Pearson={pearson_score:.2f}%, Spearman={spearman_score:.2f}%")
            print(f"   ⏱️  Encoding time: {encoding_time:.2f}s")
            
            return {
                'model_name': model_name,
                'display_name': model_info['name'],
                'pearson_correlation': pearson_score,
                'spearman_correlation': spearman_score,
                'encoding_time': encoding_time,
                'n_pairs': len(sampled_data),
                'cosine_similarities': correlations['cosine_similarities'][:100],  # Sample for storage
                'true_scores': correlations['true_scores'][:100],
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
        print("🚀 Starting STS Correlation Benchmark...")
        
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
                    'Pearson (%)': result['pearson_correlation'],
                    'Spearman (%)': result['spearman_correlation'],
                    'Encoding Time (s)': result['encoding_time'],
                    'Pairs Tested': result['n_pairs'],
                    'Status': 'Success'
                })
            else:
                rows.append({
                    'Model': result['display_name'],
                    'Model Path': model_name,
                    'Pearson (%)': 0,
                    'Spearman (%)': 0,
                    'Encoding Time (s)': 0,
                    'Pairs Tested': 0,
                    'Status': f"Failed: {result.get('error', 'Unknown')}"
                })
        
        return pd.DataFrame(rows)
    
    def plot_results(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive visualization với correlation và timing charts
        """
        # Filter successful models
        successful_df = df[df['Status'] == 'Success'].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful models to plot")
            return
        
        # Create 3x1 subplot (vertical layout)
        fig, axes = plt.subplots(3, 1, figsize=(14, 15))
        fig.suptitle('Semantic Textual Similarity Benchmark Results', fontsize=16, fontweight='bold')
        
        # Sort data for different charts
        pearson_sorted = successful_df.sort_values('Pearson (%)', ascending=True)
        spearman_sorted = successful_df.sort_values('Spearman (%)', ascending=True)
        timing_sorted = successful_df.sort_values('Encoding Time (s)', ascending=True)
        
        # 1. Pearson Correlation (Horizontal Bar Chart)
        ax1 = axes[0]
        y_pos1 = np.arange(len(pearson_sorted))
        
        bars1 = ax1.barh(y_pos1, pearson_sorted['Pearson (%)'], 
                        color='#FF6B6B', alpha=0.8)
        
        ax1.set_yticks(y_pos1)
        ax1.set_yticklabels(pearson_sorted['Model'])
        ax1.set_xlabel('Pearson Correlation (%)')
        ax1.set_title('Pearson Correlation Ranking')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # Add value labels for Pearson
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 2. Spearman Correlation (Horizontal Bar Chart)
        ax2 = axes[1]
        y_pos2 = np.arange(len(spearman_sorted))
        
        bars2 = ax2.barh(y_pos2, spearman_sorted['Spearman (%)'], 
                        color='#4ECDC4', alpha=0.8)
        
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(spearman_sorted['Model'])
        ax2.set_xlabel('Spearman Correlation (%)')
        ax2.set_title('Spearman Correlation Ranking')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        
        # Add value labels for Spearman
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 3. Encoding Time Chart
        ax3 = axes[2]
        y_pos3 = np.arange(len(timing_sorted))
        
        # Color gradient based on speed (green = fast, red = slow)
        max_time = timing_sorted['Encoding Time (s)'].max()
        colors = plt.cm.RdYlGn_r(timing_sorted['Encoding Time (s)'] / max_time)
        
        bars_time = ax3.barh(y_pos3, timing_sorted['Encoding Time (s)'], 
                            color=colors, alpha=0.8)
        
        ax3.set_yticks(y_pos3)
        ax3.set_yticklabels(timing_sorted['Model'])
        ax3.set_xlabel('Encoding Time (seconds)')
        ax3.set_title('Encoding Time Ranking (Lower is Better)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels for timing
        for i, bar in enumerate(bars_time):
            width = bar.get_width()
            # Color label based on speed: green for fast, red for slow
            label_color = 'green' if width < 25 else 'orange' if width < 70 else 'red'
            ax3.text(width + max_time * 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}s', ha='left', va='center', fontsize=10, 
                    fontweight='bold', color=label_color)
        
        # Add speed categories as text annotations
        ax3.text(0.02, 0.98, ' Fast: <25s   Medium: 25-70s   Slow: >70s', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
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
        results_file = os.path.join(results_dir, f"sts_correlation_benchmark_results.json")
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Save summary CSV
        csv_file = os.path.join(results_dir, f"sts_correlation_benchmark_summary.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save chart
        chart_file = os.path.join(results_dir, f"sts_correlation_benchmark_chart.png")
        self.plot_results(df, save_path=chart_file)
        
        print(f"💾 Results saved to results/ directory:")
        print(f"   - {os.path.relpath(results_file, script_dir)} (detailed JSON)")
        print(f"   - {os.path.relpath(csv_file, script_dir)} (summary CSV)")
        print(f"   - {os.path.relpath(chart_file, script_dir)} (chart PNG)")

def main():
    """
    Main function to run STS Correlation benchmark
    """
    print("📊 STS Correlation Benchmark for Vietnamese Embedding Models")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = STSCorrelationBenchmark()
    
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
        
        # Sort by Pearson
        top_pearson = successful_df.nlargest(3, 'Pearson (%)')
        print("📈 Best Pearson Correlation:")
        for i, (_, row) in enumerate(top_pearson.iterrows(), 1):
            print(f"   {i}. {row['Model']}: {row['Pearson (%)']:.2f}%")
        
        # Sort by Spearman  
        top_spearman = successful_df.nlargest(3, 'Spearman (%)')
        print("\n📈 Best Spearman Correlation:")
        for i, (_, row) in enumerate(top_spearman.iterrows(), 1):
            print(f"   {i}. {row['Model']}: {row['Spearman (%)']:.2f}%")
    
    # Save results
    benchmark.save_results(df)
    
    print("\n✅ Benchmark completed!")

if __name__ == "__main__":
    main()
