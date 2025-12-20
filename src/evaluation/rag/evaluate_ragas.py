#!/usr/bin/env python3
"""
Evaluate RAG system using RAGAs metrics.
This script evaluates Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper
    HAS_LLM_SUPPORT = True
except ImportError:
    HAS_LLM_SUPPORT = False
    logger.warning("langchain-google-genai not available. LLM configuration will be skipped.")

try:
    # Use HuggingfaceEmbeddings (lowercase) which has embed_query/embed_documents
    # HuggingFaceEmbeddings (capital) only has embed_text/embed_texts
    from ragas.embeddings import HuggingfaceEmbeddings
    HAS_RAGAS_EMBEDDING_SUPPORT = True
except ImportError:
    HAS_RAGAS_EMBEDDING_SUPPORT = False
    logger.warning("RAGAs HuggingfaceEmbeddings not available. Will try alternative embedding configuration.")


def load_dataset(dataset_file: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load the evaluation dataset with answers.
    
    Args:
        dataset_file: Path to the dataset JSON file
        limit: Optional limit on number of entries to load (for testing)
        
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
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        dataset = dataset[:limit]
        logger.info("✅ Loaded %d entries (limited from %d total)", len(dataset), total_entries)
    else:
        logger.info("✅ Loaded %d entries", total_entries)
    
    return dataset


def prepare_ragas_dataset(dataset: List[Dict]) -> Dataset:
    """
    Convert dataset to RAGAs format.
    
    Args:
        dataset: List of dataset entries
        
    Returns:
        Dataset in RAGAs format
    """
    logger.info("🔄 Preparing dataset for RAGAs evaluation...")
    
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    
    valid_count = 0
    skipped_count = 0
    
    for item in dataset:
        question = item.get("question", "")
        answer = item.get("answer", "")
        ground_truth = item.get("ground_truth", "")
        contexts = item.get("contexts", [])
        answer_status = item.get("answer_status", "")
        
        # Skip entries without answers or with error status
        if not answer or answer_status not in ["success", "Success"]:
            skipped_count += 1
            logger.debug("Skipping entry %d: no answer or error status", item.get("question_id", 0))
            continue
        
        # Ensure contexts is a list of strings
        if not isinstance(contexts, list):
            contexts = []
        
        # RAGAs expects contexts as list of strings
        ragas_data["question"].append(question)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(ground_truth)
        valid_count += 1
    
    logger.info("✅ Prepared %d valid entries for evaluation", valid_count)
    logger.info("⚠️  Skipped %d entries (no answer or error status)", skipped_count)
    
    if valid_count == 0:
        logger.error("❌ No valid entries found for evaluation")
        sys.exit(1)
    
    return Dataset.from_dict(ragas_data)


def setup_ragas_llm():
    """
    Setup LLM for RAGAs evaluation.
    
    Returns:
        LangchainLLMWrapper instance or None if setup fails
    """
    if not HAS_LLM_SUPPORT:
        logger.warning("⚠️  LLM support not available. RAGAs will use default LLM if available.")
        return None
    
    logger.info("🤖 Setting up LLM for RAGAs evaluation...")
    
    if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == "":
        logger.warning("⚠️  GOOGLE_API_KEY not set. RAGAs will use default LLM if available.")
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=Config.RAGAS_LLM_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.1,
            # Note: Gemini may not support n parameter for multiple generations
            # RAGAs will handle this gracefully with warnings
        )
        ragas_llm = LangchainLLMWrapper(llm)
        logger.info("✅ LLM configured: %s", Config.RAGAS_LLM_MODEL)
        logger.info("   Note: Gemini may return 1 generation instead of requested 3 (this is normal)")
        return ragas_llm
    except Exception as e:
        logger.warning("⚠️  Failed to setup LLM: %s. RAGAs will use default LLM if available.", e)
        return None


def setup_ragas_embeddings():
    """
    Setup embeddings for RAGAs evaluation using RAGAs' native HuggingFace embeddings.
    
    Note: HuggingFaceEmbeddings (capital) has embed_text/embed_texts but metrics need
    embed_query/embed_documents. We create a wrapper to add these methods.
    
    Returns:
        Wrapped HuggingFaceEmbeddings instance or None if setup fails
    """
    logger.info("🔤 Setting up embeddings for RAGAs evaluation...")
    
    # Use the same embedding model as the rest of the project
    embedding_model_name = Config.EMBEDDING_MODEL
    logger.info("   Loading embedding model: %s", embedding_model_name)
    
    try:
        # Import RAGAs' HuggingFaceEmbeddings (capital) which works
        from ragas.embeddings import HuggingFaceEmbeddings as RAGAsHuggingFaceEmbeddings
        
        # Create the embeddings instance
        base_embeddings = RAGAsHuggingFaceEmbeddings(model=embedding_model_name)
        
        # Create a wrapper class that adds embed_query and embed_documents methods
        # These are just aliases to embed_text since HuggingFace models don't distinguish
        class EmbeddingsWrapper:
            def __init__(self, base_emb):
                self._base = base_emb
                # Copy all attributes from base
                for attr in dir(base_emb):
                    if not attr.startswith('_') and not hasattr(self, attr):
                        try:
                            setattr(self, attr, getattr(base_emb, attr))
                        except:
                            pass
            
            def embed_query(self, text: str):
                """Embed a query text (alias for embed_text)."""
                return self._base.embed_text(text)
            
            def embed_documents(self, texts):
                """Embed documents (alias for embed_texts)."""
                if isinstance(texts, str):
                    texts = [texts]
                return self._base.embed_texts(texts)
            
            async def aembed_query(self, text: str):
                """Async embed a query text (alias for aembed_text)."""
                return await self._base.aembed_text(text)
            
            async def aembed_documents(self, texts):
                """Async embed documents (alias for aembed_texts)."""
                if isinstance(texts, str):
                    texts = [texts]
                return await self._base.aembed_texts(texts)
        
        ragas_embeddings = EmbeddingsWrapper(base_embeddings)
        logger.info("✅ Embeddings configured with query/document methods: %s", embedding_model_name)
        return ragas_embeddings
        
    except ImportError as e:
        logger.error("❌ Cannot import RAGAs HuggingFaceEmbeddings: %s", e)
        logger.error("   Make sure you have ragas installed: pip install ragas")
        return None
    except Exception as e:
        logger.error("❌ Failed to setup embeddings: %s", e)
        logger.error("   This is required for metrics like context_precision and context_recall.")
        return None


def evaluate_ragas(dataset: Dataset, ragas_llm=None, ragas_embeddings=None) -> Dict:
    """
    Evaluate dataset using RAGAs metrics.
    
    Args:
        dataset: Dataset in RAGAs format
        ragas_llm: Optional LLM wrapper for RAGAs
        ragas_embeddings: Optional embeddings wrapper for RAGAs
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("🚀 Starting RAGAs evaluation...")
    logger.info("   Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    
    try:
        # Create metric instances with embeddings and LLM configured
        metrics_list = []
        
        # Configure faithfulness (doesn't need embeddings, but may need LLM)
        faithfulness_metric = faithfulness
        if ragas_llm:
            try:
                faithfulness_metric.llm = ragas_llm
            except Exception as e:
                logger.warning("   ⚠️  Could not set LLM for faithfulness: %s", e)
        metrics_list.append(faithfulness_metric)
        
        # Configure answer_relevancy (needs embeddings)
        if ragas_embeddings:
            try:
                # Create new instance with embeddings
                answer_relevancy_metric = answer_relevancy(embeddings=ragas_embeddings)
                if ragas_llm:
                    answer_relevancy_metric.llm = ragas_llm
                logger.info("   ✅ Configured embeddings for answer_relevancy")
            except (TypeError, AttributeError):
                # Fallback: try setting as attribute if constructor doesn't accept it
                try:
                    answer_relevancy_metric = answer_relevancy
                    answer_relevancy_metric.embeddings = ragas_embeddings
                    if ragas_llm:
                        answer_relevancy_metric.llm = ragas_llm
                    logger.info("   ✅ Configured embeddings for answer_relevancy (fallback)")
                except Exception as e:
                    logger.warning("   ⚠️  Could not set embeddings for answer_relevancy: %s", e)
                    answer_relevancy_metric = answer_relevancy
        else:
            answer_relevancy_metric = answer_relevancy
        metrics_list.append(answer_relevancy_metric)
        
        # Configure context_precision (needs embeddings)
        if ragas_embeddings:
            try:
                context_precision_metric = context_precision(embeddings=ragas_embeddings)
                logger.info("   ✅ Configured embeddings for context_precision")
            except (TypeError, AttributeError):
                # Fallback: try setting as attribute
                try:
                    context_precision_metric = context_precision
                    context_precision_metric.embeddings = ragas_embeddings
                    logger.info("   ✅ Configured embeddings for context_precision (fallback)")
                except Exception as e:
                    logger.warning("   ⚠️  Could not set embeddings for context_precision: %s", e)
                    context_precision_metric = context_precision
        else:
            context_precision_metric = context_precision
        metrics_list.append(context_precision_metric)
        
        # Configure context_recall (needs embeddings)
        if ragas_embeddings:
            try:
                context_recall_metric = context_recall(embeddings=ragas_embeddings)
                logger.info("   ✅ Configured embeddings for context_recall")
            except (TypeError, AttributeError):
                # Fallback: try setting as attribute
                try:
                    context_recall_metric = context_recall
                    context_recall_metric.embeddings = ragas_embeddings
                    logger.info("   ✅ Configured embeddings for context_recall (fallback)")
                except Exception as e:
                    logger.warning("   ⚠️  Could not set embeddings for context_recall: %s", e)
                    context_recall_metric = context_recall
        else:
            context_recall_metric = context_recall
        metrics_list.append(context_recall_metric)
        
        # Configure LLM for metrics that need it (if not already set)
        if ragas_llm:
            logger.info("   Configuring LLM for metrics...")
            for metric in metrics_list:
                try:
                    if hasattr(metric, 'llm') and not hasattr(metric, '_llm_set'):
                        metric.llm = ragas_llm
                        metric._llm_set = True
                except Exception as e:
                    logger.warning("   ⚠️  Could not set LLM for metric: %s", e)
        
        if not ragas_embeddings:
            logger.error("❌ No embeddings configured. RAGAs will try to use OpenAI embeddings.")
            logger.error("   This will fail without OPENAI_API_KEY.")
            raise ValueError("Embeddings must be configured for RAGAs evaluation")
        
        # Ensure embeddings are set on all metrics that need them
        # This prevents RAGAs from trying to create its own embeddings
        logger.info("   Verifying embeddings are set on all metrics...")
        for metric in metrics_list:
            if hasattr(metric, 'embeddings'):
                if metric.embeddings is None or not hasattr(metric.embeddings, 'embed_text'):
                    logger.warning("   ⚠️  Metric %s has no valid embeddings, setting...", metric.__class__.__name__)
                    try:
                        metric.embeddings = ragas_embeddings
                    except Exception as e:
                        logger.warning("   ⚠️  Could not set embeddings: %s", e)
        
        # Try to pass embeddings to evaluate function if supported
        eval_kwargs = {}
        if ragas_embeddings:
            # Some versions of RAGAs support passing embeddings directly
            try:
                eval_kwargs['embeddings'] = ragas_embeddings
            except TypeError:
                # If not supported, we rely on setting it on metrics
                pass
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics_list,
                **eval_kwargs
            )
            
            logger.info("✅ Evaluation completed")
            
            # Check for partial results (some metrics might have failed)
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
                for metric in metrics:
                    if metric in df.columns:
                        valid_count = df[metric].notna().sum()
                        total_count = len(df)
                        if valid_count < total_count:
                            logger.warning("⚠️  Metric '%s' has only %d/%d valid results (some may have failed)", 
                                         metric, valid_count, total_count)
            
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                logger.error("❌ API rate limit exceeded (429 error)")
                logger.error("   Some metrics may have partial results.")
                logger.error("   You can:")
                logger.error("   1. Wait and retry later")
                logger.error("   2. Check if partial results were saved")
                logger.error("   3. Reduce the dataset size for testing")
            raise
        
    except Exception as e:
        logger.error("❌ Error during evaluation: %s", e)
        raise


def save_results(result, output_file: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        result: Evaluation results from RAGAs (Dataset or dict)
        output_file: Path to output JSON file
    """
    logger.info("💾 Saving results to %s...", output_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RAGAs result to a serializable format
    try:
        # Try to convert to pandas DataFrame first, then to dict
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            # Convert DataFrame to dict with records orientation
            result_dict = df.to_dict(orient='records')
        elif hasattr(result, 'to_dict'):
            # Try to_dict method
            result_dict = result.to_dict()
        elif isinstance(result, pd.DataFrame):
            result_dict = result.to_dict(orient='records')
        elif isinstance(result, dict):
            result_dict = result
        else:
            # Try to iterate and convert manually
            try:
                result_dict = []
                for i, row in enumerate(result):
                    row_dict = {}
                    if hasattr(row, '__dict__'):
                        row_dict = row.__dict__
                    elif isinstance(row, dict):
                        row_dict = row
                    else:
                        # Try to convert row to dict
                        row_dict = dict(row) if hasattr(row, 'items') else {str(i): row}
                    result_dict.append(row_dict)
            except (TypeError, AttributeError) as e:
                logger.warning("⚠️  Could not convert result to dict directly: %s", e)
                # Last resort: convert to string representation
                result_dict = {"error": "Could not serialize result", "type": str(type(result)), "repr": str(result)}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ Results saved (%d records)", len(result_dict) if isinstance(result_dict, list) else 1)
        
    except Exception as e:
        logger.error("❌ Failed to save results: %s", e)
        logger.error("   Result type: %s", type(result))
        # Try to save as CSV as fallback
        try:
            csv_path = output_path.with_suffix('.csv')
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
            elif isinstance(result, pd.DataFrame):
                df = result
            else:
                raise ValueError("Cannot convert to DataFrame")
            
            df.to_csv(csv_path, index=False)
            logger.info("✅ Results saved as CSV instead: %s", csv_path)
        except Exception as e2:
            logger.error("❌ Failed to save as CSV as well: %s", e2)
            raise


def calculate_statistics(result: Dict) -> Dict:
    """
    Calculate statistics for each metric.
    
    Args:
        result: Evaluation results from RAGAs
        
    Returns:
        Dictionary with statistics for each metric
    """
    logger.info("📊 Calculating statistics...")
    
    # Convert to DataFrame for easier analysis
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)
    
    stats = {}
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    
    for metric in metrics:
        # Skip if metric column doesn't exist
        if metric not in df.columns:
            logger.warning("⚠️  Metric '%s' not found in results", metric)
            stats[metric] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
            continue
        
        values = df[metric].dropna()
        if len(values) > 0:
            stats[metric] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": int(len(values)),
            }
        else:
            logger.warning("⚠️  No valid values for metric: %s", metric)
            stats[metric] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }
    
    return stats


def visualize_metrics(result: Dict, output_dir: str) -> None:
    """
    Create thesis-quality visualizations for RAGAs metrics.
    
    Args:
        result: Evaluation results from RAGAs
        output_dir: Directory to save visualization files
    """
    logger.info("📈 Creating visualizations...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set professional plot style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300
    })
    
    # Convert to DataFrame
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Ans. Relevancy",
        "context_precision": "Ctx. Precision",
        "context_recall": "Ctx. Recall",
    }
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        logger.warning("⚠️  No metrics found in results for visualization")
        return
    
    # Common thesis colors (muted/professional)
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    
    # 1. Box plot for all metrics (Distribution)
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [df[m].dropna() for m in available_metrics]
    labels = [metric_labels.get(m, m) for m in available_metrics]
    
    # Create boxplot with modern styling
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                   medianprops=dict(color="black", linewidth=1.5),
                   boxprops=dict(linewidth=1.2),
                   whiskerprops=dict(linewidth=1.2),
                   capprops=dict(linewidth=1.2))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Score')
    ax.set_title('Metric Distribution (Box Plot)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ragas_metrics_boxplot.png', bbox_inches='tight')
    logger.info("✅ Saved boxplot: ragas_metrics_boxplot.png")
    plt.close()
    
    # 2. Bar chart with Mean & Std Dev
    # --------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    means = [df[m].mean() for m in available_metrics]
    stds = [df[m].std() for m in available_metrics]
    labels = [metric_labels.get(m, m) for m in available_metrics]
    
    # Bars with error bars (capsize adds the little horizontal lines)
    bars = ax.bar(labels, means, yerr=stds, capsize=5, 
                  color=colors[:len(available_metrics)], 
                  edgecolor='black', alpha=0.8,
                  error_kw=dict(lw=1.5, capthick=1.5))
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        # Place text slightly above max(height, mean+error) to avoid overlap usually, 
        # but simple height + offset is often cleaner if error bars aren't huge.
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Mean Score')
    ax.set_title('Average Performance', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ragas_metrics_barchart.png', bbox_inches='tight')
    logger.info("✅ Saved bar chart: ragas_metrics_barchart.png")
    plt.close()
    
    # 3. Correlation heatmap (if > 1 metric)
    # --------------------------------------
    if len(available_metrics) > 1:
        fig, ax = plt.subplots(figsize=(7, 6))
        correlation_matrix = df[available_metrics].corr()
        
        # 'coolwarm' is good standard for correlation (-1 to 1)
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Ticks
        ax.set_xticks(range(len(available_metrics)))
        ax.set_yticks(range(len(available_metrics)))
        ax.set_xticklabels([metric_labels.get(m, m) for m in available_metrics], rotation=45, ha='right')
        ax.set_yticklabels([metric_labels.get(m, m) for m in available_metrics])
        
        # Annotations
        for i in range(len(available_metrics)):
            for j in range(len(available_metrics)):
                val = correlation_matrix.iloc[i, j]
                # White text for dark colors, black for light
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f'{val:.2f}',
                       ha="center", va="center", color=text_color, fontweight='bold')
        
        ax.set_title('Metric Correlation Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        plt.savefig(output_path / 'ragas_metrics_correlation.png', bbox_inches='tight')
        logger.info("✅ Saved correlation matrix: ragas_metrics_correlation.png")
        plt.close()


def print_summary(stats: Dict) -> None:
    """
    Print summary statistics to console.
    
    Args:
        stats: Dictionary with statistics for each metric
    """
    logger.info("\n" + "="*80)
    logger.info("📊 RAGAs Evaluation Summary")
    logger.info("="*80)
    
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    
    for metric, label in metric_labels.items():
        if metric in stats:
            s = stats[metric]
            logger.info("\n%s:", label)
            logger.info("  Mean:   %.4f", s["mean"])
            logger.info("  Median: %.4f", s["median"])
            logger.info("  Std:    %.4f", s["std"])
            logger.info("  Min:    %.4f", s["min"])
            logger.info("  Max:    %.4f", s["max"])
            logger.info("  Count:  %d", s["count"])
    
    logger.info("\n" + "="*80)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system using RAGAs metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings (full dataset)
  python evaluate_ragas.py --dataset ragas_evaluation_dataset_with_answers.json
  
  # Test with limited entries (e.g., 5 entries)
  python evaluate_ragas.py --dataset ragas_evaluation_dataset_with_answers.json --limit 5
  
  # Specify custom output directory
  python evaluate_ragas.py --dataset ragas_evaluation_dataset_with_answers.json --output-dir results/
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="ragas_evaluation_dataset_with_answers.json",
        help="Input dataset file with answers (default: ragas_evaluation_dataset_with_answers.json)"
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
        default="ragas_evaluation_results.json",
        help="Output JSON file for results (default: ragas_evaluation_results.json)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to evaluate (useful for testing, e.g., --limit 5)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    dataset_file = script_dir / args.dataset if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_dir = script_dir / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    results_file = output_dir / args.results_file
    
    # Load dataset
    dataset = load_dataset(str(dataset_file), limit=args.limit)
    
    # Prepare for RAGAs
    ragas_dataset = prepare_ragas_dataset(dataset)
    
    # Setup LLM for RAGAs
    ragas_llm = setup_ragas_llm()
    
    # Setup embeddings for RAGAs
    ragas_embeddings = setup_ragas_embeddings()
    
    # Ensure OPENAI_API_KEY is not set to prevent RAGAs from using OpenAI embeddings
    # (We want to use our local HuggingFace embeddings instead)
    import os
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    if original_openai_key:
        logger.info("⚠️  OPENAI_API_KEY is set. Temporarily unsetting to use local embeddings.")
        os.environ.pop("OPENAI_API_KEY", None)
    
    if ragas_embeddings:
        logger.info("   Using local HuggingFace embeddings: %s", Config.EMBEDDING_MODEL)
    else:
        logger.warning("⚠️  No embeddings configured. RAGAs may try to use OpenAI embeddings.")
        logger.warning("   This will cause an error if OPENAI_API_KEY is not set.")
    
    try:
        # Evaluate
        result = evaluate_ragas(ragas_dataset, ragas_llm, ragas_embeddings)
    except Exception as e:
        error_msg = str(e).lower()
        if "openai_api_key" in error_msg or "openai" in error_msg:
            logger.error("❌ RAGAs is trying to use OpenAI embeddings.")
            logger.error("   This usually means embeddings were not properly configured.")
            logger.error("   Solutions:")
            logger.error("   1. Make sure langchain-community is installed: pip install langchain-community")
            logger.error("   2. Check that the embedding model '%s' can be loaded", Config.EMBEDDING_MODEL)
            logger.error("   3. If the error persists, RAGAs may need embeddings configured differently")
            logger.error("   Error details: %s", e)
        raise
    finally:
        # Restore OPENAI_API_KEY if it was set
        if original_openai_key:
            os.environ["OPENAI_API_KEY"] = original_openai_key
    
    # Calculate statistics
    stats = calculate_statistics(result)
    
    # Print summary
    print_summary(stats)
    
    # Save results
    save_results(result, str(results_file))
    
    # Create visualizations
    visualize_metrics(result, str(output_dir))
    
    logger.info("\n✅ Evaluation complete!")
    logger.info("   Results saved to: %s", results_file)
    logger.info("   Visualizations saved to: %s", output_dir)


if __name__ == "__main__":
    main()

