#!/usr/bin/env python3
"""
Evaluate retrieval performance using Hit Rate@K, Precision@K, and NDCG@K metrics.
This script evaluates how well the retrieval system finds relevant documents.
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

try:
    import ranx
    HAS_RANX = True
except ImportError:
    HAS_RANX = False
    logger.warning("⚠️  ranx not found. Installing is recommended: pip install ranx")


class RetrievalEvaluator:
    """Evaluates retrieval performance using ranx or manual calculation."""

    def __init__(self, dataset_path: str, output_dir: str, k: int = 5):
        """
        Initialize the evaluator.

        Args:
            dataset_path: Path to the evaluation dataset JSON.
            output_dir: Directory to save results and visualizations.
            k: The 'K' in Hit Rate@K, Precision@K, etc.
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.k = k
        self.metrics_keys = [f"hit_rate@{k}", f"precision@{k}", f"ndcg@{k}"]
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, limit: Optional[int] = None) -> List[Dict]:
        """Load and validate dataset."""
        logger.info("📖 Loading dataset from %s...", self.dataset_path)
        if not self.dataset_path.exists():
            logger.error("❌ Dataset file not found: %s", self.dataset_path)
            sys.exit(1)

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if limit and limit > 0:
            dataset = dataset[:limit]
            logger.info("✅ Loaded %d entries (limited)", len(dataset))
        else:
            logger.info("✅ Loaded %d entries", len(dataset))
            
        return dataset

    def prepare_data(self, dataset: List[Dict]) -> Tuple[Dict, Dict]:
        """Convert dataset to qrels and run dictionaries."""
        qrels = {}
        run = {}

        for item in dataset:
            q_id = str(item.get("question_id", ""))
            if not q_id:
                # Generate stable ID if missing
                q_id = str(hash(item.get("question", "")))
            
            contexts = item.get("contexts", [])
            relevance = item.get("relevance", [])

            if not contexts:
                continue
            
            # Handle missing relevance
            if not relevance or len(relevance) != len(contexts):
                logger.debug("Missing relevance for %s. Assuming all relevant.", q_id)
                relevance = [1] * len(contexts) # 1 for boolean relevant

            qrels[q_id] = {}
            run[q_id] = {}

            for i, context in enumerate(contexts):
                doc_id = f"doc_{q_id}_{i}"
                
                # Qrels: 1 if relevant, 0 otherwise
                is_rel = 1 if relevance[i] else 0
                qrels[q_id][doc_id] = is_rel
                
                # Run: Score based on position (higher is better)
                # 1.0 for rank 1, 0.5 for rank 2, etc.
                score = 1.0 / (i + 1)
                run[q_id][doc_id] = score

        return qrels, run

    def evaluate(self, qrels_dict: Dict, run_dict: Dict) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Run evaluation. Uses ranx if available and safe, otherwise manual fallback.
        Returns: (overall_metrics, per_query_df)
        """
        logger.info("📊 Evaluating retrieval performance (k=%d)...", self.k)
        
        if HAS_RANX:
            return self._evaluate_with_ranx(qrels_dict, run_dict)
        else:
            return self._evaluate_manual(qrels_dict, run_dict)

    def _evaluate_with_ranx(self, qrels_dict: Dict, run_dict: Dict) -> Tuple[Dict, pd.DataFrame]:
        """Use ranx library for evaluation."""
        try:
            qrels = ranx.Qrels(qrels_dict)
            run = ranx.Run(run_dict)

            # ranx.evaluate returns a dict for overall metrics
            overall = ranx.evaluate(qrels, run, metrics=self.metrics_keys)
            
            per_query_df = self._calculate_per_query_manual(qrels_dict, run_dict)
            return overall, per_query_df

        except Exception as e:
            logger.error("❌ ranx evaluation failed: %s. Falling back to manual.", e)
            return self._evaluate_manual(qrels_dict, run_dict)

    def _evaluate_manual(self, qrels_dict: Dict, run_dict: Dict) -> Tuple[Dict, pd.DataFrame]:
        """Manual calculation fallback."""
        df = self._calculate_per_query_manual(qrels_dict, run_dict)
        
        # Average
        overall = {
            f"hit_rate@{self.k}": df[f"hit_rate@{self.k}"].mean(),
            f"precision@{self.k}": df[f"precision@{self.k}"].mean(),
            f"ndcg@{self.k}": df[f"ndcg@{self.k}"].mean()
        }
        return overall, df

    def _calculate_per_query_manual(self, qrels_dict: Dict, run_dict: Dict) -> pd.DataFrame:
        """Calculate metrics per query manually."""
        results = []
        k = self.k

        for q_id, q_doc_scores in run_dict.items():
            if q_id not in qrels_dict:
                continue

            # Ground truth: doc_id -> relevance (0 or 1)
            truth = qrels_dict[q_id]
            # Retrieved: list of (doc_id, score), sorted descending
            retrieved = sorted(q_doc_scores.items(), key=lambda x: x[1], reverse=True)
            top_k_retrieved = retrieved[:k]
            top_k_ids = [d for d, s in top_k_retrieved]

            # 1. Hit Rate @ K
            # definition: 1 if any relevant doc is in top k
            relevant_docs = {d for d, r in truth.items() if r > 0}
            hits = [1 for d in top_k_ids if d in relevant_docs]
            hit_rate = 1.0 if hits else 0.0

            # 2. Precision @ K
            # definition: proportion of retrieved docs that are relevant
            precision = sum(hits) / len(top_k_ids) if top_k_ids else 0.0

            # 3. NDCG @ K
            # DCG = sum( rel_i / log2(i + 2) )
            dcg = 0.0
            for i, doc_id in enumerate(top_k_ids):
                rel = 1.0 if doc_id in relevant_docs else 0.0
                dcg += rel / np.log2(i + 2)

            # IDCG = Ideal DCG (best possible ordering of relevant docs)
            # We take all true relevant docs, assume they are at top positions
            ideal_rels = sorted([1.0] * len(relevant_docs), reverse=True)
            # Only consider top k slots for IDCG
            ideal_rels = ideal_rels[:k]
            
            idcg = 0.0
            for i, rel in enumerate(ideal_rels):
                idcg += rel / np.log2(i + 2)

            ndcg = dcg / idcg if idcg > 0 else 0.0

            results.append({
                "query_id": q_id,
                f"hit_rate@{k}": hit_rate,
                f"precision@{k}": precision,
                f"ndcg@{k}": ndcg
            })

        if not results:
             return pd.DataFrame(columns=["query_id", f"hit_rate@{k}", f"precision@{k}", f"ndcg@{k}"])

        return pd.DataFrame(results)

    def visualize(self, overall: Dict, df: pd.DataFrame):
        """Generate thesis-quality plots."""
        logger.info("📊 Generating visualizations in %s...", self.output_dir)
        
        # Set larger font sizes for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300
        })

        # 1. Bar Chart (Overall Performance)
        # ----------------------------------
        plt.figure(figsize=(8, 6))
        values = [overall.get(k, 0.0) for k in self.metrics_keys]
        # Professional colors: muted blue, muted green, muted orange
        colors = ['#4e79a7', '#59a14f', '#f28e2b']
        
        bars = plt.bar(self.metrics_keys, values, color=colors, edgecolor='black', alpha=0.8)
        plt.title(f'Retrieval Performance (Top-{self.k})', fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add values on top
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "retrieval_metrics_bar.png", bbox_inches='tight')
        plt.close()

        if df.empty:
            return

        # 2. Box Plot (Distribution/Variance - Critical for Thesis)
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))
        data_to_plot = []
        labels = []
        
        for key in self.metrics_keys:
            if key in df.columns:
                data_to_plot.append(df[key].dropna())
                labels.append(key)
        
        # Create boxplot with patch_artist=True to fill with color
        bplot = plt.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, 
                            medianprops=dict(color="black", linewidth=1.5),
                            boxprops=dict(linewidth=1.2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))
        
        # Color the boxes
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        plt.title(f'Metric Distribution (Box Plot)', fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(-0.05, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "retrieval_metrics_boxplot.png", bbox_inches='tight')
        plt.close()

        # 3. Histograms (Detailed Distribution)
        # -------------------------------------
        # Create a single figure with subplots
        n_metrics = len(data_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1: axes = [axes]
        
        for idx, (data, label) in enumerate(zip(data_to_plot, labels)):
            ax = axes[idx]
            ax.hist(data, bins=10, color=colors[idx], alpha=0.7, edgecolor='black')
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('Score')
            ax.set_xlim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "retrieval_metrics_histograms.png", bbox_inches='tight')
        plt.close()

    def save_results(self, overall: Dict, df: pd.DataFrame, filename: str = "retrieval_results.json"):
        """Save results to disk."""
        # Save JSON
        json_path = self.output_dir / filename
        
        # Add statistics to JSON
        stats = {}
        for key in self.metrics_keys:
            if key in df.columns:
                d = df[key]
                stats[key] = {
                    "mean": float(d.mean()),
                    "std": float(d.std()),
                    "min": float(d.min()),
                    "max": float(d.max())
                }

        output_data = {
            "parameters": {
                "k": self.k,
                "dataset_path": str(self.dataset_path)
            },
            "overall_metrics": overall,
            "statistics": stats
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        # Save CSV
        csv_path = self.output_dir / filename.replace(".json", ".csv")
        df.to_csv(csv_path, index=False)
        
        logger.info("💾 Saved results to %s and %s", json_path, csv_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate retrieval system using ranx.")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--k", type=int, default=Config.TOP_K_DOCUMENTS, help=f"Top K documents (default: {Config.TOP_K_DOCUMENTS})")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of entries")

    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        dataset_path=args.dataset, 
        output_dir=args.output_dir,
        k=args.k
    )

    dataset = evaluator.load_dataset(limit=args.limit)
    qrels, run = evaluator.prepare_data(dataset)
    
    overall, df = evaluator.evaluate(qrels, run)
    
    # Print summary
    print("\n" + "="*30)
    print("   RETRIEVAL EVALUATION REPORT   ")
    print("="*30)
    for k, v in overall.items():
        print(f"{k:<15}: {v:.4f}")
    print("="*30 + "\n")

    evaluator.save_results(overall, df)
    evaluator.visualize(overall, df)

if __name__ == "__main__":
    main()
