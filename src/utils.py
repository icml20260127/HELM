import os
import pickle
import random
import json
import logging


class Pool:
    def __init__(self, elements=None):
        self.item_list = []
        self.item_index = {}
        if elements:
            self.add(*elements)

    def sample(self):
        return random.choice(self.item_list) if self.item_list else None

    def remove(self, *items):
        for item in items:
            if item in self.item_index:
                idx = self.item_index[item]
                last_item = self.item_list[-1]
                self.item_list[idx] = last_item
                self.item_index[last_item] = idx
                self.item_list.pop()
                del self.item_index[item]

    def add(self, *items):
        for item in items:
            if item not in self.item_index:
                self.item_index[item] = len(self.item_list)
                self.item_list.append(item)


class UnionFind:
    def __init__(self, nodes):
        self.parent = {x: x for x in nodes}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        self.parent[yr] = xr
        return True


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_graph(graph, file_path):
    # Ensure parent directory exists before writing
    dirpath = os.path.dirname(file_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(graph, f)


def setup_logger(
    out_dir: str, name: str = "optuna_tree_search", level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a module logger writing to ``out_dir/logs/{name}.log``
    and to stdout. If the logger already has handlers, it is returned unchanged.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, f"{name}.log"))
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    # Prevent propagation to the root logger to avoid duplicate messages
    logger.propagate = False
    return logger


def setup_worker_logger(
    worker_id: int,
    logs_dir: str,
    name: str = "optuna_tree_search",
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to ``logs_dir/worker_{id}.log``.
    This is intended for per-worker process logging.
    """
    logger_name = f"{name}.worker_{worker_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    os.makedirs(logs_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(logs_dir, f"worker_{worker_id}.log"))
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Prevent propagation to the root logger so worker logs don't duplicate
    logger.propagate = False
    return logger


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Return human-friendly duration (e.g. 1h2m3s or 4.56s)."""
    try:
        s = float(seconds)
    except Exception:
        return "0s"
    if s < 1:
        return f"{s*1000:.0f}ms"
    secs = int(s)
    hours, rem = divmod(secs, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    if s < 60 and s != int(s):
        parts[-1] = f"{s:.2f}s"
    return "".join(parts)


def check_gpu_availability():
    """Check if GPU is available and can be used.

    Returns:
        (bool, str) - availability and message
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return (
                False,
                "CUDA is not available. Please ensure NVIDIA GPU drivers are installed.",
            )
        return True, f"GPU available: {torch.cuda.get_device_name(0)}"
    except ImportError:
        return (
            False,
            "PyTorch is required for GPU support. Install with: pip install torch",
        )
    except Exception as e:
        return False, f"Error checking GPU availability: {str(e)}"


def validate_gpu_device(device_id):
    """Validate that the specified GPU device exists.

    This checks the device count according to the current PyTorch view
    (i.e., after any CUDA_VISIBLE_DEVICES remapping).
    """
    try:
        import torch

        if device_id >= torch.cuda.device_count():
            return (
                False,
                f"GPU device {device_id} not found. Available devices: {torch.cuda.device_count()}",
            )
        return True, f"GPU device {device_id} is available"
    except ImportError:
        return (
            False,
            "PyTorch is required for GPU support. Install with: pip install torch",
        )
    except Exception as e:
        return False, f"Error validating GPU device: {str(e)}"


def safe_set_gpu_device(device_id):
    """Set GPU device by assigning `CUDA_VISIBLE_DEVICES` only.

    This intentionally avoids importing or calling into PyTorch here — the
    environment variable is sufficient to restrict visible GPUs before any
    CUDA initialization occurs.

    Returns:
        (bool, str) - success and message
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        return True, f"CUDA_VISIBLE_DEVICES set to {device_id}"
    except Exception as e:
        return False, f"Error setting CUDA_VISIBLE_DEVICES: {str(e)}"


def load_edge_scores(file_path):
    """Load edge scores from CSV with columns: source, target, score."""
    import pandas as pd

    df = pd.read_csv(file_path)
    return {(row["source"], row["target"]): row["score"] for _, row in df.iterrows()}


def load_degree_distribution(path):
    """Load target degree distribution from json/csv/txt.

    Expected formats:
    - JSON: array of numbers
    - CSV: first numeric column (header ignored)
    - TXT: one number per line
    """
    import json as _json
    import pandas as pd

    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            data = _json.load(f)
        if not isinstance(data, (list, tuple)):
            raise ValueError("Degree distribution JSON must be a list")
        return list(map(float, data))

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("Degree distribution CSV is empty")
        return list(map(float, df[df.columns[0]].to_list()))

    # fallback: txt with one value per line
    with open(path, "r") as f:
        return list(map(float, [line.strip() for line in f if line.strip()]))


def compute_tree_metrics(true_tree, pred_tree):
    """Compute standard tree comparison metrics (TPR, FPR, precision, recall, F1, TP/FP/FN).

    Args:
        true_tree: Ground truth tree (networkx Graph)
        pred_tree: Predicted tree (networkx Graph)

    Returns:
        Dict with keys: tpr, fpr, precision, recall, f1, tp, fp, fn
    """
    true_edges = {frozenset((u, v)) for u, v in true_tree.edges()}
    pred_edges = {frozenset((u, v)) for u, v in pred_tree.edges()}

    TP = len(true_edges & pred_edges)
    FP = len(pred_edges - true_edges)
    FN = len(true_edges - pred_edges)

    metrics = compute_metrics_from_confusion(TP, FP, FN)
    metrics.update({"tp": TP, "fp": FP, "fn": FN})
    return metrics


def compute_metrics_from_confusion(tp, fp, fn, total_pos=None, tn=None):
    """Compute classification metrics from confusion matrix counts.

    Args:
        tp, fp, fn: True positives, false positives, false negatives
        total_pos: Total positive examples (defaults to tp + fn)
        tn: True negatives (required for correct FPR; if None, FPR is set to 0.0)

    Returns:
        Dict with keys: tpr, fpr, precision, recall, f1
    """
    if total_pos is None:
        total_pos = tp + fn

    tpr = tp / total_pos if total_pos > 0 else 0.0
    # Correct FPR definition: fp / (fp + tn). If tn unavailable, set to 0.0.
    if tn is not None and (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_confusion_from_trees(true_edges, pred_tree_edges, all_graph_edges):
    """Compute TP/FP/FN/TN from edge sets.

    Args:
        true_edges: Set of frozensets representing positive (true) edges
        pred_tree_edges: Edges from predicted tree (list of tuples or frozensets)
        all_graph_edges: Set of all possible edges in graph (frozensets)

    Returns:
        Dict with keys: tp, fp, fn, tn, total_true
    """
    pred_edges = {
        frozenset((u, v)) if isinstance((u, v), tuple) else (u, v)
        for u, v in pred_tree_edges
    }

    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    tn = len(all_graph_edges - true_edges - pred_edges)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_true": len(true_edges),
    }
