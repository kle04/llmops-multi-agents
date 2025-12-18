#!/usr/bin/env python3
"""
Evaluate retrieval performance using Hit Rate@K, Precision@K, and NDCG@K metrics.
This script evaluates how well the retrieval system finds relevant documents.
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

try:
    import ranx
    HAS_RANX = True
except ImportError:
    HAS_RANX = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config import Config



def load_dataset(dataset_file: str, limit: int = None) -> List[Dict]:
    """
    Load the evaluation dataset.
    
    Args:
        dataset_file: Path to the dataset JSON file
        limit: Optional limit on number of entries to load
        
    Returns:
        List of dataset entries
    """
    logger.info("📖 Loading dataset from %s...", dataset_file)
    dataset_path = Path(dataset_file)
    if not dataset_path.exists():
        logger.error("❌ Dataset file not found: %s", dataset_path)
        sys.exit(1)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_entries = len(dataset)
    
    if limit is not None and limit > 0:
        dataset = dataset[:limit]
        logger.info("✅ Loaded %d entries (limited from %d total)", len(dataset), total_entries)
    else:
        logger.info("✅ Loaded %d entries", total_entries)
    
    return dataset


def prepare_ranx_data(dataset: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Prepare data in ranx format for evaluation.
    
    Reads relevance judgments from the dataset (pre-judged).
    
    Args:
        dataset: List of dataset entries with 'relevance' field
        
    Returns:
        Tuple of (qrels_dict, run_dict)
    """
    logger.info("🔄 Preparing data for ranx evaluation...")
    
    qrels_dict = {}
    run_dict = {}
    
    for item in dataset:
        question = item.get("question", "")
        question_id = str(item.get("question_id", len(qrels_dict) + 1))
        contexts = item.get("contexts", [])
        relevance = item.get("relevance", [])
        
        if not question or not contexts:
            continue
        
        # Check if relevance judgments exist
        if not relevance or len(relevance) != len(contexts):
            logger.warning("⚠️  Missing relevance judgments for question_id %s. Assuming all contexts are relevant.", question_id)
            relevance = [True] * len(contexts)
        
        relevant_count = sum(relevance)
        logger.debug("   Question %s: %d/%d contexts are relevant", question_id, relevant_count, len(contexts))
        
        # Build qrels (ground truth relevance)
        # Format: {query_id: {doc_id: relevance_score}}
        qrels_dict[question_id] = {}
        for i, (context, is_relevant) in enumerate(zip(contexts, relevance)):
            doc_id = f"doc_{question_id}_{i}"
            qrels_dict[question_id][doc_id] = 1 if is_relevant else 0
        
        # Build run (retrieved documents with scores)
        # Format: {query_id: {doc_id: score}}
        # We use position-based scores (higher = better rank)
        run_dict[question_id] = {}
        for i, context in enumerate(contexts):
            doc_id = f"doc_{question_id}_{i}"
            # Score decreases with position (first = highest score)
            score = 1.0 / (i + 1)
            run_dict[question_id][doc_id] = score
    
    logger.info("✅ Prepared %d queries for evaluation", len(qrels_dict))
    
    return qrels_dict, run_dict


def evaluate_retrieval(qrels_dict: Dict, run_dict: Dict) -> Dict:
    """
    Evaluate retrieval performance using ranx.
    
    Args:
        qrels_dict: Ground truth relevance judgments as dict
        run_dict: Retrieved documents with scores as dict
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("📊 Evaluating retrieval performance...")
    
    if not HAS_RANX:
        logger.error("❌ ranx is not installed. Install it with: pip install ranx")
        sys.exit(1)
    
    try:
        # Convert to ranx objects
        qrels = ranx.Qrels(qrels_dict)
        run = ranx.Run(run_dict)
        
        # Calculate metrics - use metrics that don't require numba JIT
        # Calculate manually to avoid numba issues
        metrics = calculate_metrics_manual(qrels_dict, run_dict)
        
        logger.info("✅ Evaluation completed")
        logger.info("   Hit Rate@5:  %.4f", metrics["hit_rate_at_5"])
        logger.info("   Precision@5: %.4f", metrics["precision_at_5"])
        logger.info("   NDCG@5:      %.4f", metrics["ndcg_at_5"])
        
        return metrics
        
    except Exception as e:
        logger.error("❌ Error during evaluation: %s", e)
        # Fallback to manual calculation
        logger.info("   Falling back to manual calculation...")
        return calculate_metrics_manual(qrels_dict, run_dict)


def calculate_metrics_manual(qrels_dict: Dict, run_dict: Dict) -> Dict:
    """
    Manually calculate Hit Rate@K, Precision@K, and NDCG@K to avoid numba issues.
    
    Args:
        qrels_dict: Ground truth relevance judgments
        run_dict: Retrieved documents with scores
        
    Returns:
        Dictionary with evaluation metrics
    """
    hit_rate_5_scores = []
    precision_5_scores = []
    ndcg_5_scores = []
    
    for query_id in qrels_dict.keys():
        if query_id not in run_dict:
            continue
        
        qrels = qrels_dict[query_id]
        run = run_dict[query_id]
        
        # Get relevant doc IDs and their relevance scores
        relevant_docs = {doc_id: rel for doc_id, rel in qrels.items() if rel > 0}
        
        if not relevant_docs:
            # No relevant docs: Hit Rate@5 = 0 (no relevant chunks = no hits possible)
            logger.debug("   Query %s: No relevant documents, Hit Rate@5 = 0.0", query_id)
            hit_rate_5_scores.append(0.0)
            precision_5_scores.append(0.0)
            ndcg_5_scores.append(0.0)
            continue
        
        # Sort retrieved docs by score (descending)
        sorted_docs = sorted(run.items(), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc_id for doc_id, _ in sorted_docs]
        
        # Hit Rate@5: Binary metric per query
        # For each query: 1 if at least 1 relevant chunk is in top 5, else 0
        # Overall: Average of all query scores
        # Formula: Hit Rate@5 = (1/N) * Σ [1 if at least 1 relevant in top 5 else 0] for each query
        top_5 = set(retrieved_docs[:5])
        relevant_in_top_5_set = top_5 & set(relevant_docs.keys())
        hit_rate_5 = 1.0 if len(relevant_in_top_5_set) > 0 else 0.0
        hit_rate_5_scores.append(hit_rate_5)
        
        # Precision@5: Fraction of top 5 retrieved chunks that are relevant
        # Formula: Precision@5 = |relevant chunks in top 5| / 5
        # For each query: Precision@5_i = relevant_in_top_5 / 5
        # Overall: Precision@5 = (1/N) * Σ [Precision@5_i] for each query
        top_5_list = retrieved_docs[:5]
        relevant_in_top_5 = sum(1 for doc_id in top_5_list if doc_id in relevant_docs)
        precision_5 = relevant_in_top_5 / len(top_5_list) if top_5_list else 0.0
        precision_5_scores.append(precision_5)
        
        # NDCG@5: Normalized Discounted Cumulative Gain
        # DCG@K = Σ_{i=1}^{K} (rel_i / log₂(i+1))
        # rel_i is the relevance score (binary or graded)
        dcg_5 = 0.0
        for i, doc_id in enumerate(top_5_list[:5]):
            if doc_id in relevant_docs:
                rel_score = relevant_docs[doc_id]
                dcg_5 += rel_score / np.log2(i + 2)  # i+2 because log2(1) = 0, we want log2(2) for position 1
        
        # IDCG@5: Ideal DCG for top 5 positions
        # Standard implementation: Sort ALL relevant documents by descending relevance score,
        # then calculate DCG only on the top K positions of that sorted list.
        # If there are more than K relevant documents, IDCG@K still only uses top K positions.
        all_relevant_sorted = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
        # Calculate DCG only on top 5 positions of the ideal ranking
        idcg_5 = 0.0
        for i, (doc_id, rel_score) in enumerate(all_relevant_sorted[:5]):
            idcg_5 += rel_score / np.log2(i + 2)
        
        # NDCG@5 = DCG@5 / IDCG@5 (avoid division by zero)
        ndcg_5 = dcg_5 / idcg_5 if idcg_5 > 0 else 0.0
        ndcg_5_scores.append(ndcg_5)
    
    # Calculate average metrics across all queries
    # Hit Rate@5 = (1/N) * Σ [1 if at least 1 relevant in top 5 else 0] for each query
    # Precision@5 = (1/N) * Σ [|relevant in top 5| / 5] for each query
    # NDCG@5 = (1/N) * Σ [NDCG@5_i] for each query
    metrics = {
        "hit_rate_at_5": np.mean(hit_rate_5_scores) if hit_rate_5_scores else 0.0,
        "precision_at_5": np.mean(precision_5_scores) if precision_5_scores else 0.0,
        "ndcg_at_5": np.mean(ndcg_5_scores) if ndcg_5_scores else 0.0,
    }
    
    return metrics


def calculate_per_query_metrics(qrels_dict: Dict, run_dict: Dict) -> pd.DataFrame:
    """
    Calculate metrics for each query individually.
    
    Args:
        qrels_dict: Ground truth relevance judgments as dict
        run_dict: Retrieved documents with scores as dict
        
    Returns:
        DataFrame with per-query metrics
    """
    logger.info("📊 Calculating per-query metrics...")
    
    results = []
    
    for query_id in qrels_dict.keys():
        if query_id not in run_dict:
            continue
        
        qrels = qrels_dict[query_id]
        run = run_dict[query_id]
        
        # Get relevant doc IDs and their relevance scores
        relevant_docs = {doc_id: rel for doc_id, rel in qrels.items() if rel > 0}
        
        if not relevant_docs:
            # No relevant docs, set all metrics to 0 (correct: no relevant docs = no hits)
            logger.debug("   Query %s: No relevant documents, setting all metrics to 0.0", query_id)
            results.append({
                "query_id": query_id,
                "hit_rate_at_5": 0.0,
                "precision_at_5": 0.0,
                "ndcg_at_5": 0.0,
            })
            continue
        
        # Sort retrieved docs by score (descending)
        sorted_docs = sorted(run.items(), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc_id for doc_id, _ in sorted_docs]
        
        # Hit Rate@5: 1 if at least 1 relevant chunk in top 5, else 0
        top_5 = set(retrieved_docs[:5])
        relevant_in_top_5_set = top_5 & set(relevant_docs.keys())
        hr5 = 1.0 if len(relevant_in_top_5_set) > 0 else 0.0
        
        # Precision@5: |relevant chunks in top 5| / 5
        top_5_list = retrieved_docs[:5]
        relevant_in_top_5 = sum(1 for doc_id in top_5_list if doc_id in relevant_docs)
        p5 = relevant_in_top_5 / len(top_5_list) if top_5_list else 0.0
        
        # NDCG@5: Normalized Discounted Cumulative Gain
        # DCG@K = Σ_{i=1}^{K} (rel_i / log₂(i+1))
        dcg_5 = 0.0
        for i, doc_id in enumerate(top_5_list[:5]):
            if doc_id in relevant_docs:
                rel_score = relevant_docs[doc_id]
                dcg_5 += rel_score / np.log2(i + 2)
        
        # IDCG@5: Sort ALL relevant documents by descending relevance,
        # then calculate DCG only on top 5 positions of that sorted list
        all_relevant_sorted = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
        idcg_5 = 0.0
        for i, (doc_id, rel_score) in enumerate(all_relevant_sorted[:5]):
            idcg_5 += rel_score / np.log2(i + 2)
        
        ndcg5 = dcg_5 / idcg_5 if idcg_5 > 0 else 0.0
        
        results.append({
            "query_id": query_id,
            "hit_rate_at_5": hr5,
            "precision_at_5": p5,
            "ndcg_at_5": ndcg5,
        })
    
    df = pd.DataFrame(results)
    logger.info("✅ Calculated metrics for %d queries", len(df))
    
    return df


def visualize_results(metrics: Dict, per_query_df: pd.DataFrame, output_dir: str) -> None:
    """
    Visualize retrieval evaluation results.
    
    Args:
        metrics: Dictionary with overall metrics
        per_query_df: DataFrame with per-query metrics
        output_dir: Directory to save visualizations
    """
    logger.info("📊 Generating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar chart of overall metrics
    plt.figure(figsize=(12, 6))
    metric_names = ["Hit Rate@5", "Precision@5", "NDCG@5"]
    metric_values = [
        metrics["hit_rate_at_5"],
        metrics["precision_at_5"],
        metrics["ndcg_at_5"]
    ]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    bars = plt.bar(metric_names, metric_values, color=colors)
    plt.ylabel('Score', fontsize=12)
    plt.title('Retrieval Evaluation Metrics', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_metrics_barchart.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved bar chart to %s", output_path / "retrieval_metrics_barchart.png")
    
    # 2. Histograms for per-query metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ("hit_rate_at_5", "Hit Rate@5"),
        ("precision_at_5", "Precision@5"),
        ("ndcg_at_5", "NDCG@5")
    ]
    
    plot_colors = ['#3498db', '#e74c3c', '#f39c12']
    
    for idx, (col, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = per_query_df[col].dropna()
        ax.hist(values, bins=10, edgecolor='black', alpha=0.7, color=plot_colors[idx])
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_val = values.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_metrics_histograms.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved histograms to %s", output_path / "retrieval_metrics_histograms.png")
    
    # 3. Box plots
    plt.figure(figsize=(12, 6))
    data_to_plot = [
        per_query_df["hit_rate_at_5"].dropna(),
        per_query_df["precision_at_5"].dropna(),
        per_query_df["ndcg_at_5"].dropna()
    ]
    labels = ["Hit Rate@5", "Precision@5", "NDCG@5"]
    box_colors = ['#3498db', '#e74c3c', '#f39c12']
    
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Distribution of Retrieval Metrics (Per Query)', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "retrieval_metrics_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Saved box plot to %s", output_path / "retrieval_metrics_boxplot.png")
    
    logger.info("📊 Visualizations saved to %s", output_path)


def save_results(metrics: Dict, per_query_df: pd.DataFrame, output_file: str) -> None:
    """
    Save evaluation results to JSON and CSV files.
    
    Args:
        metrics: Dictionary with overall metrics
        per_query_df: DataFrame with per-query metrics
        output_file: Path to output JSON file
    """
    logger.info("💾 Saving results...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics as JSON
    results = {
        "overall_metrics": metrics,
        "statistics": {
            "hit_rate_at_5": {
                "mean": float(per_query_df["hit_rate_at_5"].mean()),
                "std": float(per_query_df["hit_rate_at_5"].std()),
                "min": float(per_query_df["hit_rate_at_5"].min()),
                "max": float(per_query_df["hit_rate_at_5"].max()),
            },
            "precision_at_5": {
                "mean": float(per_query_df["precision_at_5"].mean()),
                "std": float(per_query_df["precision_at_5"].std()),
                "min": float(per_query_df["precision_at_5"].min()),
                "max": float(per_query_df["precision_at_5"].max()),
            },
            "ndcg_at_5": {
                "mean": float(per_query_df["ndcg_at_5"].mean()),
                "std": float(per_query_df["ndcg_at_5"].std()),
                "min": float(per_query_df["ndcg_at_5"].min()),
                "max": float(per_query_df["ndcg_at_5"].max()),
            },
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ Results saved to %s", output_path)
    
    # Save per-query metrics as CSV
    csv_path = output_path.with_suffix('.csv')
    per_query_df.to_csv(csv_path, index=False)
    logger.info("✅ Per-query metrics saved to %s", csv_path)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval performance using Hit Rate@K, Precision@K, and NDCG@K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings
  python evaluate_retrieval.py --dataset ranx_evaluation_dataset.json
  
  # Test with limited entries
  python evaluate_retrieval.py --dataset ranx_evaluation_dataset.json --limit 5
  
  # Custom output directory
  python evaluate_retrieval.py --dataset ranx_evaluation_dataset.json --output-dir results/
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ranx_evaluation_dataset.json",
        help="Input dataset file (default: ranx_evaluation_dataset.json)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results and visualizations (default: results)"
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        default="retrieval_evaluation_results.json",
        help="Output JSON file for results (default: retrieval_evaluation_results.json)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to evaluate (useful for testing)"
    )
    
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    dataset_file = script_dir / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    results_file = output_dir / args.results_file
    
    if not HAS_RANX:
        logger.error("❌ ranx is not installed.")
        logger.error("   Install it with: pip install ranx")
        sys.exit(1)
    
    # Load dataset
    dataset = load_dataset(str(dataset_file), limit=args.limit)
    
    # Check if relevance judgments exist in dataset
    has_relevance = all("relevance" in item for item in dataset)
    if not has_relevance:
        logger.warning("⚠️  Dataset missing 'relevance' field. Please add relevance judgments to the dataset.")
        logger.warning("   Relevance should be a list of booleans, one for each context.")
    
    # Prepare ranx data (reads relevance from dataset)
    qrels_dict, run_dict = prepare_ranx_data(dataset)
    
    # Evaluate
    metrics = evaluate_retrieval(qrels_dict, run_dict)
    
    # Calculate per-query metrics
    per_query_df = calculate_per_query_metrics(qrels_dict, run_dict)
    
    # Save results
    save_results(metrics, per_query_df, str(results_file))
    
    # Visualize
    visualize_results(metrics, per_query_df, str(output_dir))
    
    logger.info("\n✅ Retrieval evaluation complete!")
    logger.info("   Results saved to: %s", results_file)
    logger.info("   Visualizations saved to: %s", output_dir)


if __name__ == "__main__":
    main()

