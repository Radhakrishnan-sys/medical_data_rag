import csv
from datetime import datetime
from typing import Dict, Any
 
# ====== CONFIG ======
EXPERIMENT_LOG = "rag_experiments.csv"
 
EXPERIMENT_COLUMNS = [
    "timestamp", "experiment_name", "embedding_model", "vectorstore", "index_type",
    "chunk_size", "chunk_overlap", "k", "retrieval_strategy", "rag_strategy",
    "reranker", "llm_model", "chain_type", "metrics", "latency", "notes"
]
 
 
def init_experiment_log():
    """Create CSV log with headers if not already present."""
    try:
        with open(EXPERIMENT_LOG, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(EXPERIMENT_COLUMNS)
    except FileExistsError:
        pass  # File already exists, do nothing
 
 
def log_experiment(config: Dict[str, Any], metrics: Dict[str, Any], notes: str = ""):
    """
    Append a single experiment record to the CSV log.
 
    Args:
        config: dict of experiment config variables (use the keys in EXPERIMENT_COLUMNS)
        metrics: dict of metrics (accuracy, recall, latency, cost, etc)
        notes: optional free-text notes
    """
    row = [
        datetime.now().isoformat(),
        config.get("experiment_name", ""),
        config.get("embedding_model", ""),
        config.get("vectorstore", ""),
        config.get("index_type", ""),
        config.get("chunk_size", ""),
        config.get("chunk_overlap", ""),
        config.get("k", ""),
        config.get("retrieval_strategy", ""),
        config.get("rag_strategy", ""),
        config.get("reranker", ""),
        config.get("llm_model", ""),
        config.get("chain_type", ""),
        str(metrics),  # Store as stringified dict for flexibility
        metrics.get("latency", ""),
        notes
    ]
    with open(EXPERIMENT_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
 
 
# =========================
# Example usage in pipeline
# =========================
 
if __name__ == "__main__":
    # 1. Initialize log (call once at the start of your workflow)
    init_experiment_log()
 
   
 