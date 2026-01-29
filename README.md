# Hierarchy Core Components

Core algorithms for **hierarchy prediction from knowledge graphs** using machine learning and optimization.

This package provides end-to-end solutions for:

- **Feature engineering** from graph structures
- **Edge scoring** with XGBoost/LightGBM or Graph Neural Networks
- **Hierarchy reconstruction** using Simulated Annealing
- **Root selection** for optimal tree directionality
- **Hyperparameter optimization** with Optuna

## Directory Structure

```
core_components/
├── README.md                          # This file
├── src/                               # Core Python modules (6 modules, 2000+ lines)
│   ├── __init__.py
│   ├── utils.py                       # Utilities: graph I/O, metrics, logging
│   ├── algorithms/                    # Core algorithms
│   │   ├── edge_features.py          # Feature extraction (40+ graph measures)
│   │   ├── edge_scores.py            # XGBoost/LightGBM edge scoring
│   │   ├── edge_scores_gnn.py        # GNN-based edge scoring (GINEConv)
│   │   ├── optimal_root.py           # Optimal root finding (235x faster)
│   │   └── tree_search.py            # Simulated annealing tree search
│   ├── data_extractors/               # Data extraction from external sources
│   │   ├── memetracker_extractor.py  # MemeTracker data processing
│   │   └── wiki_extractor.py         # Wikipedia category extraction
│   └── scripts/
│       └── optuna_tree_search.py     # Hyperparameter tuning framework
├── data/                              # Sample graphs (60 graph pairs, 3 datasets)
│   ├── wiki/                          # Wikipedia categories (20 graphs)
│   │   ├── Algorithms/, AI/, ML/, ... 
│   │   └── Each: entity_graph.pkl + hierarchy_tree.pkl
│   ├── microbiome/                    # Taxonomic data (20 graphs)
│   │   ├── Collections: C0001, C0005, ...
│   │   └── Patients: P0004, P0005, ...
│   └── memetracker/                   # Cascades (20 graphs)
│       └── cascades/: cascade_00154, ...
└── manifests/                         # Dataset manifests (6 files)
    ├── manifest_10_wiki_{train,test}.json
    ├── manifest_10_microbiome_{train,test}.json
    └── manifest_10_memetracker_{train,test}.json
```

## Dependencies

### Core Dependencies (Required)

These are required for all core algorithms:

```bash
pip install networkx numpy pandas scikit-learn
```

| Package | Version | Used By | Purpose |
|---------|---------|---------|----------|
| `networkx` | ≥2.6 | All modules | Graph data structures and algorithms |
| `numpy` | ≥1.19 | All modules | Numerical computations |
| `pandas` | ≥1.1 | edge_scores, utils | Tabular data and metrics |
| `scikit-learn` | ≥0.24 | edge_features, tree_search | Feature scaling, metrics |

### Tree Search & Feature Extraction

Required for edge scoring and hierarchy reconstruction:

```bash
pip install xgboost lightgbm optuna
pip install graphMeasures
pip install --upgrade networkx
```

**⚠️ Important:** After installing `graphMeasures`, you **must** upgrade `networkx` to ensure compatibility.

| Package | Version | Module | Purpose |
|---------|---------|--------|----------|
| `xgboost` | ≥1.4 | edge_scores | XGBoost classifier for edge predictions |
| `lightgbm` | ≥3.1 | edge_scores | LightGBM classifier alternative to XGBoost |
| `optuna` | ≥2.8 | optuna_tree_search, edge_scores | Hyperparameter optimization |
| `graphMeasures` | latest | edge_features | 40+ structural graph features (centrality, clustering, etc.) |

### Graph Neural Networks (Optional)

Only required if using GNN-based edge scoring (`edge_scores_gnn.py`):

```bash
pip install torch pytorch-lightning torch-geometric
```

| Package | Version | Module | Purpose |
|---------|---------|--------|----------|
| `torch` | ≥1.10 | edge_scores_gnn | PyTorch deep learning framework |
| `pytorch-lightning` | ≥1.5 | edge_scores_gnn | Lightning training abstraction |
| `torch-geometric` | ≥2.0 | edge_scores_gnn | Graph neural network layers (GINEConv) |

### Wikipedia Extraction (Optional)

Only required if extracting new Wikipedia data (`wiki_extractor.py`):

```bash
pip install wikipediaapi
```

| Package | Version | Module | Purpose |
|---------|---------|--------|----------|
| `wikipediaapi` | ≥0.5 | wiki_extractor | Wikipedia API access with rate limiting |

### Full Installation (All Features)

```bash
# Core + tree search + GNN + Wikipedia
pip install networkx numpy pandas scikit-learn xgboost lightgbm optuna graphMeasures
pip install --upgrade networkx
pip install torch pytorch-lightning torch-geometric
pip install wikipediaapi
```

## Quick Start (5 Minutes)

### 1. Load and Explore Data

```python
import json
from src.utils import load_graph

# Load manifest (maps graph IDs to file paths)
with open("manifests/manifest_10_wiki_test.json") as f:
    manifest = json.load(f)

# Get first graph
entry = manifest[0]
print(f"Graph ID: {entry['graph_id']}")

# Load entity graph (G) and ground-truth hierarchy (T)
G = load_graph(entry["G_path"])   # Full knowledge graph
T = load_graph(entry["T_path"])   # Target hierarchy (subset of G)

print(f"Entity graph G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Hierarchy T: {T.number_of_nodes()} nodes, {T.number_of_edges()} edges")
```

### 2. Quick Feature Extraction

```bash
# Compute features for a single graph
python -m src.algorithms.edge_features \
  --gid "Algorithms" \
  --collection wiki \
  --output-base .
```

Outputs:

- `outputs/wiki/Algorithms/features/node_features.csv`
- `outputs/wiki/Algorithms/features/edge_features.csv`

### 3. Train Edge Scoring Model

```bash
# Train XGBoost model on 10 graphs (multi-graph training)
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --model xgb \
  --n-trials 20
```

Outputs:

- `outputs/wiki/model/best_model.pkl` (trained classifier)
- `outputs/wiki/model/best_params.json` (optimal hyperparameters)
- Per-graph: `outputs/wiki/*/scores/edge_scores.csv` (edge predictions)

### 4. Hierarchy Reconstruction (Simulated Annealing)

```bash
# Run tree search on a single graph
python -m src.algorithms.tree_search \
  --gid "Algorithms" \
  --graph data/wiki/Algorithms/entity_graph.pkl \
  --tree data/wiki/Algorithms/hierarchy_tree.pkl \
  --collection wiki \
  --output-dir .
```

Outputs:

- `outputs/wiki/Algorithms/search/tree.pkl` (reconstructed tree)
- `outputs/wiki/Algorithms/search/metrics.json` (evaluation metrics)

### 5. Find Optimal Root and Evaluate

```bash
# Automatically select best root and direct tree to arborescence
python -m src.algorithms.optimal_root \
  --gid "Algorithms" \
  --collection wiki \
  --output-dir . \
  --mode train
```

Outputs:

- `outputs/wiki/Algorithms/search/optimal_root/tree_directed.pkl` (rooted tree)
- `outputs/wiki/Algorithms/search/optimal_root/optimal_root.json` (root info)

## Full Workflow Pipeline

For end-to-end hierarchy prediction:

```bash
# Step 1: Feature extraction (10 graphs)
python -m src.algorithms.edge_features \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-base . \
  --workers 4

# Step 2: Train scoring model (multi-graph, 10 graphs)
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --model xgb \
  --n-trials 50

# Step 3: Hyperparameter tuning for tree search
python -m src.scripts.optuna_tree_search \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --n-workers 4 \
  --trials-per-worker 25 \
  --graph-workers 10

# Step 4: Reconstruct trees on test set (using tuned hyperparameters)
python -m src.algorithms.tree_search \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --config outputs/wiki/optuna/best_hyperparameters.json

# Step 5: Find optimal roots
python -m src.algorithms.optimal_root \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --mode eval

# Step 6: Evaluate results
python -c "
import json
import pandas as pd
from src.utils import load_graph, compute_tree_metrics

with open('manifests/manifest_10_wiki_test.json') as f:
    manifest = json.load(f)

metrics = []
for entry in manifest:
    gid = entry['graph_id']
    coll = entry['collection']
    
    # Load predicted and ground-truth trees
    S = load_graph(f'outputs/{coll}/{gid}/search/optimal_root/tree_directed.pkl')
    T = load_graph(entry['T_path'])
    
    m = compute_tree_metrics(T, S)
    m['graph_id'] = gid
    metrics.append(m)

df = pd.DataFrame(metrics)
print(df[['graph_id', 'tpr', 'precision', 'recall', 'f1']])
print(f'Mean TPR: {df[\"tpr\"].mean():.4f}')
"
```

## Core Modules

### `src/algorithms/edge_features.py`

Extracts 40+ structural features from graphs:

**Features computed:**

- Node features (centrality, clustering, k-core, Fiedler vector, etc.)
- Edge features (betweenness, connectivity, shortest path)
- Combined node-pair features (avg and diff vectors)

**Parallelization:** ProcessPoolExecutor for 4x speedup

**Usage:**

```bash
# Single graph
python -m src.algorithms.edge_features --gid ID --collection NAME --output-base .

# Batch (manifest mode)
python -m src.algorithms.edge_features --manifest manifests/manifest.json --collection wiki
```

### `src/algorithms/edge_scores.py`

Edge scoring with GBDT models:

**Models:** XGBoost, LightGBM  
**Optimization:** Optuna with TPESampler  
**Early stopping:** MedianPruner (trial-level), PlateauCallback (study-level)  
**Multi-graph training:** Pool features across graphs, train single model  
**Evaluation:** AUC, F1, precision, recall on train/val/test splits

**Usage:**

```bash
# Train on 10 graphs
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --model xgb \
  --n-trials 50 \
  --plateau-patience 20

# Score test graphs
python -m src.algorithms.edge_scores \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --score-only \
  --model-path outputs/wiki/model
```

### `src/algorithms/edge_scores_gnn.py`

Graph Neural Network edge scoring:

**Architecture:** GINEConv with 4 layers, BatchNorm  
**Features:** Node embeddings (from node_features.csv) + edge features  
**Training:** PyTorch Lightning with early stopping  
**Optimization:** Optuna hyperparameter search  
**Evaluation:** ROC curves, per-graph and aggregate metrics

**Key features:**

- Scalable node feature encoding
- Edge dropout for regularization (20-60%)
- Aggressive hyperparameter tuning (hidden_dim, dropout_rate, learning_rate)
- Multi-graph training with train/val/test splits
- GPU acceleration support

**Usage:**

```bash
# Train GNN with hyperparameter search
python -m src.scripts.optuna_tree_search \
  --mode train \
  --train_manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output_dir . \
  --max_trials 30 \
  --gpu_id 0
```

### `src/algorithms/tree_search.py`

Simulated Annealing for hierarchy reconstruction:

**Features:**

- 5 loss components: community, diversity, degree, shortcut, edge score
- 3 move types: NNI, SPR, TBR
- Adaptive boldness (TBR/SPR ratio increases during stagnation)
- Early stopping (2-tier: stagnation + bold stagnation)
- Optimized adjacency dict (235x faster than NetworkX)

**Loss functions:**

- **Community:** Penalizes edges between different communities (Leiden)
- **Diversity:** Maximizes neighbor diversity within communities
- **Degree:** MSE between predicted and target degree distribution
- **Shortcut:** Penalizes edges not in original graph
- **Score:** Minimizes edge scores (prefers high-confidence edges)

**Configuration:** 12 hyperparameters (optimizable via Optuna)

**Usage:**

```bash
# Optimize hyperparameters (Optuna)
python -m src.scripts.optuna_tree_search \
  --manifest manifests/manifest_10_wiki_train.json \
  --collection wiki \
  --output-dir . \
  --n-workers 4 \
  --trials-per-worker 25

# Use best parameters for test set
python -m src.algorithms.tree_search \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki \
  --output-dir . \
  --config outputs/wiki/optuna/.../best_hyperparameters.json
```

### `src/algorithms/optimal_root.py`

Optimal root selection for tree directionality:

**Algorithm:** BFS-based, tests all candidate roots (O(n²) with optimizations)  
**Criterion:** Minimizes MSE between predicted and ground-truth depth distributions  
**Speedup:** 235x faster than NetworkX via adjacency dict  

**Outputs per graph:**

- Optimal root node ID
- Depth distribution vectors
- Directed arborescence (tree rooted at optimal_root)
- Directed recall metrics (if ground-truth available)

**Usage:**

```bash
# Find optimal root (single graph)
python -m src.algorithms.optimal_root \
  --gid ID --collection wiki --output-dir . --mode train

# Batch processing
python -m src.algorithms.optimal_root \
  --manifest manifests/manifest_10_wiki_test.json \
  --collection wiki --output-dir . --mode eval
```

### `src/utils.py`

Utility functions:

| Function | Purpose |
|----------|---------|
| `load_graph(path)` | Load NetworkX graph from pickle |
| `save_graph(graph, path)` | Save NetworkX graph to pickle |
| `compute_tree_metrics(true_tree, pred_tree)` | Compute TPR, FPR, F1, etc. |
| `compute_confusion_from_trees(...)` | Get TP/FP/FN/TN counts |
| `setup_logger(out_dir, level)` | Configure logging |
| `Pool`, `UnionFind` | Efficient data structures |

## Data Format Reference

### Manifest File Format

```json
[
  {
    "graph_id": "Algorithms",
    "collection": "wiki",
    "G_path": "data/wiki/Algorithms/entity_graph.pkl",
    "T_path": "data/wiki/Algorithms/hierarchy_tree.pkl",
    "node_features_path": "outputs/wiki/Algorithms/features/node_features.csv",
    "edge_features_path": "outputs/wiki/Algorithms/features/edge_features.csv",
    "score_path": "outputs/wiki/Algorithms/scores/edge_scores.csv",
    "positive_edges_path": "outputs/wiki/Algorithms/positive_edges/positive.pkl"
  }
]
```

### Graph File Formats

- **`.pkl` files:** NetworkX Graph objects (pickle format)
  - Can load with: `G = pickle.load(open("graph.pkl", "rb"))`
  - Or use: `from src.utils import load_graph`

- **`.csv` files:** Features or scores in tabular format
  - Features: rows=nodes/edges, columns=feature_values
  - Scores: columns=[source, target, score]

## Datasets Overview

| Dataset | # Graphs | # Nodes (avg) | # Edges (avg) | Type |
|---------|----------|--------------|---------------|------|
| **Wiki** | 20 | ~500 | ~1000 | Category hierarchies |
| **Microbiome** | 20 | ~300 | ~600 | Taxonomic trees |
| **MemeTracker** | 20 | ~150 | ~400 | Cascade trees |

### Graph Statistics

```python
import json
from src.utils import load_graph

with open("manifests/manifest_10_wiki_test.json") as f:
    manifest = json.load(f)

for entry in manifest[:3]:
    G = load_graph(entry["G_path"])
    T = load_graph(entry["T_path"])
    print(f"{entry['graph_id']:20} | G: {G.number_of_nodes():4} nodes | T: {T.number_of_edges():4} edges")
```

## Performance & Scaling

| Operation | Time (10 graphs) | Parallelization |
|-----------|-----------------|-----------------|
| Feature extraction | ~2 min | 4 workers (ProcessPoolExecutor) |
| Model training (XGB) | ~1 min | Single process + Optuna |
| Hyperparameter search (50 trials) | ~15 min | 4 workers × 25 trials each |
| Tree search (per graph) | ~10-60s | Parallelizable across graphs |
| Optimal root selection | ~0.5s | Very fast (BFS-based) |

## Troubleshooting

**Issue:** `FileNotFoundError: data/wiki/...`

- **Solution:** Ensure manifest paths are relative to where you run the scripts

**Issue:** Out of memory during feature extraction

- **Solution:** Reduce `--workers` (default 4) or use `--feature-workers 1`

**Issue:** Slow tree search

- **Solution:** Reduce `--max-iter` (default 5M) or increase `--workers` for parallelization

**Issue:** GPU not detected

- **Solution:** Check `CUDA_VISIBLE_DEVICES` or use `--gpu_id -1` for CPU-only mode
