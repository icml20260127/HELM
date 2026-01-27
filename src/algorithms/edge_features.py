import argparse
import os
import sys

# BLAS thread limiting (BEFORE importing numpy/scipy)
# Check if user explicitly disabled it via --no-blas-limit flag
# We need to check sys.argv early since --no-blas-limit should prevent threading limits
_blas_limit_disabled = "--no-blas-limit" in sys.argv or os.environ.get(
    "HIERARCHY_NO_BLAS_LIMIT", ""
).lower() in ("1", "true", "yes")

if not _blas_limit_disabled:
    # CRITICAL: Set BLAS/OMP threads to 1 BEFORE importing numpy/scipy
    # This prevents thread oversubscription in multi-process context
    # See THREADING_ANALYSIS.md for detailed explanation
    # User can disable with: --no-blas-limit or HIERARCHY_NO_BLAS_LIMIT=1
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import tempfile
import random
import re
import numpy as np
import pandas as pd
import networkx as nx
import concurrent.futures
import time
import multiprocessing
import logging

from graphMeasures import FeatureCalculator as NodeFeatureCalculator
from src.utils import load_graph, format_duration

# Worker-local cached graph (set by _worker_init in ProcessPoolExecutor initializer)
_WORKER_GRAPH = None


def _worker_init(graph_path):
    """Initializer for per-graph ProcessPoolExecutor workers: load graph once.

    This is passed as `initializer` to ProcessPoolExecutor in `process_graph`.
    """
    global _WORKER_GRAPH
    try:
        _WORKER_GRAPH = load_graph(graph_path)
        logger.debug(f"Worker initialized with graph {graph_path}")
    except Exception:
        logger.exception(
            f"Failed to initialize worker graph from {graph_path}", exc_info=True
        )


# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # avoid duplicate stdout from parent loggers


# format_duration moved to src.utils.format_duration


FEATURE_MODE = {
    "average_neighbor_degree": "either",
    "degree": "either",
    "in_degree": "directed",
    "out_degree": "directed",
    "louvain": "undirected",
    # "hierarchy_energy": "either", # ! NOT IMPLEMENTED
    "motif3": "either",  # has gpu support in graphMeasures
    "motif4": "either",  # has gpu support in graphMeasures
    "k_core": "either",  # has gpu support in graphMeasures
    # "attractor_basin": "directed",  # * disabled: often all-NaN; revisit if GPU/robust impl available
    "page_rank": "either",  # has gpu support in graphMeasures
    "fiedler_vector": "undirected",
    "closeness_centrality": "either",
    "eccentricity": "either",
    "load_centrality": "either",
    "bfs_moments": "either",
    "flow": "directed",  # has gpu support in graphMeasures
    "betweenness_centrality": "either",
    # "communicability_betweenness_centrality": "undirected",  # * disabled: very long runtime; consider GPU in future
    "eigenvector_centrality": "either",
    "clustering_coefficient": "either",
    "square_clustering_coefficient": "either",
    # "generalized_degree": "undirected", # ! BUG
    # "all_pairs_shortest_path_length": "either",  # * disabled: O(n^2) output; too large for wiki graphs
}


def split_features_for_graph(graph):
    is_directed = graph.is_directed()
    directed_feats, undirected_feats = [], []

    for feat, mode in FEATURE_MODE.items():
        if mode == "directed" and is_directed:
            directed_feats.append(feat)
        elif mode == "undirected":
            undirected_feats.append(feat)
        elif mode == "either":
            (
                directed_feats.append(feat)
                if is_directed
                else undirected_feats.append(feat)
            )
    return directed_feats, undirected_feats


def calculate_node_features(graph, features, output_dir, directed):
    fc = NodeFeatureCalculator(
        graph,
        features,
        dir_path=output_dir,
        acc=False,
        directed=directed,
        should_zscore=False,
    )
    fc.calculate_features(should_dump=False)
    df = fc.get_features()
    return df


def _check_existing_tmp(output_dir, feature_type, feature_name):
    """Check for existing tmp files and return newest if found."""
    if not os.environ.get("HIERARCHY_SKIP_ON_TMP"):
        return None
    try:
        import glob
        pattern = os.path.join(output_dir, f"*.{feature_type}.{feature_name}.tmp.csv")
        matches = glob.glob(pattern)
        if matches:
            return max(matches, key=os.path.getmtime)
    except Exception:
        pass
    return None


def _compute_feature_worker_node(args_tuple):
    """Compute a single node feature and return a temporary CSV path (index=node).

    args_tuple: (graph_path, feature_name, output_dir, directed)
    """
    graph_path, feature_name, output_dir, directed = args_tuple
    try:
        # Check for existing tmp file
        existing_tmp = _check_existing_tmp(output_dir, "node", feature_name)
        if existing_tmp:
            return {
                "tmp_path": existing_tmp,
                "feature_name": feature_name,
                "count": 0,
                "duration": 0.0,
                "dropped": [],
                "start_ts": None,
                "end_ts": None,
            }

        start = time.time()
        G = globals().get("_WORKER_GRAPH") or load_graph(graph_path)
        graph_for_calc = G if (directed or not G.is_directed()) else G.to_undirected()
        df = calculate_node_features(graph_for_calc, [feature_name], output_dir, directed)
        
        os.makedirs(output_dir, exist_ok=True)
        if df is None or df.shape[1] == 0:
            fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=f".node.{feature_name}.tmp.csv")
            os.close(fd)
            logger.info(f"Wrote empty marker for node feature {feature_name} at {tmp_path}")
            return None
        
        # Drop columns containing NaN
        cols_before = list(df.columns)
        df.dropna(axis=1, inplace=True)
        dropped = [c for c in cols_before if c not in df.columns]
        
        fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=f".node.{feature_name}.tmp.csv")
        try:
            os.close(fd)
            df.to_csv(tmp_path, index_label="node")
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise
        
        return {
            "tmp_path": tmp_path,
            "feature_name": feature_name,
            "count": df.shape[0],
            "duration": time.time() - start,
            "dropped": dropped,
            "start_ts": start,
            "end_ts": time.time(),
        }
    except Exception:
        logger.exception(
            f"Exception computing node feature {feature_name}", exc_info=True
        )
        return None


def _compute_feature_worker_edge(args_tuple):
    """Compute a single edge feature and return a temporary CSV path.

    args_tuple: (graph_path, feature_key, output_dir)
    Supported feature_key: 'edge_betweenness','edge_current_flow','min_edge_cut'
    """
    graph_path, feature_key, output_dir = args_tuple
    try:
        # Check for existing tmp file
        existing_tmp = _check_existing_tmp(output_dir, "edge", feature_key)
        if existing_tmp:
            return {
                "tmp_path": existing_tmp,
                "feature_key": feature_key,
                "count": 0,
                "duration": 0.0,
            }

        start = time.time()
        G = globals().get("_WORKER_GRAPH") or load_graph(graph_path)
        
        # Compute feature-specific values
        if feature_key == "edge_betweenness":
            vals = nx.edge_betweenness_centrality(G)
            records = [{"source": u, "target": v, feature_key: vals.get((u, v), 0)} for u, v in G.edges()]
        elif feature_key == "edge_current_flow":
            vals = nx.edge_current_flow_betweenness_centrality(G.to_undirected())
            records = [{"source": u, "target": v, feature_key: vals.get((u, v), 0)} for u, v in G.edges()]
        elif feature_key == "min_edge_cut":
            records = [{"source": u, "target": v, feature_key: nx.edge_connectivity(G, u, v)} for u, v in G.edges()]
        else:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        if not records:
            fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=f".edge.{feature_key}.tmp.csv")
            os.close(fd)
            logger.info(f"Wrote empty marker for edge feature {feature_key} at {tmp_path}")
            return None
        
        df = pd.DataFrame(records)
        # Drop columns with NaN (but keep source/target)
        cols_before = list(df.columns)
        cols_to_keep = [c for c in cols_before if c in ("source", "target") or not df[c].isna().any()]
        dropped = [c for c in cols_before if c not in cols_to_keep]
        df = df[cols_to_keep]
        
        fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=f".edge.{feature_key}.tmp.csv")
        try:
            os.close(fd)
            df.to_csv(tmp_path, index=False)
        except Exception:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            raise
        
        return {
            "tmp_path": tmp_path,
            "feature_key": feature_key,
            "count": df.shape[0],
            "duration": time.time() - start,
            "dropped": dropped,
            "start_ts": start,
            "end_ts": time.time(),
        }
    except Exception:
        logger.exception(
            f"Exception computing edge feature {feature_key}", exc_info=True
        )
        return None


# Helper functions for feature processing
def _extract_feature_name(tmp_path, feature_type="node"):
    """Extract feature name from tmp CSV path."""
    pattern = rf"\.{feature_type}\.([^.]+)\.tmp\.csv$"
    match = re.search(pattern, tmp_path)
    return match.group(1) if match else "unknown"


def _write_feature_metadata(csv_path, successful, failed, expected):
    """Write metadata JSON for feature CSV."""
    is_node = "node" in csv_path
    try:
        df = pd.read_csv(csv_path, nrows=1 if not is_node else None, 
                        index_col="node" if is_node else None)
        shape = [len(df), len(df.columns)] if is_node else [None, len(df.columns)]
        columns = list(df.columns)
    except Exception:
        shape, columns = [0, 0], []
    
    metadata = {
        "successful_features": successful,
        "failed_features": failed,
        "expected_features": expected,
        "shape": shape,
        "columns": columns,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(csv_path.replace('.csv', '.metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        logger.warning(f"Failed to write metadata: {csv_path.replace('.csv', '.metadata.json')}")


def _cleanup_feature_tmps(tmp_paths, failed_features, feature_type="node"):
    """Remove successful tmp files, keep failed ones."""
    for tp in tmp_paths:
        feature_name = _extract_feature_name(tp, feature_type)
        if feature_name not in failed_features:
            try:
                os.remove(tp)
            except Exception:
                pass


def _read_tmp_features(tmp_paths, feature_type="node"):
    """Read tmp files and categorize as successful or failed.
    
    Returns: (dataframes_list, successful_features, failed_features)
    """
    dfs = []
    successful = []
    failed = []
    
    for tp in tmp_paths:
        feature_name = _extract_feature_name(tp, feature_type)
        try:
            if feature_type == "node":
                df = pd.read_csv(tp, index_col="node")
            else:
                df = pd.read_csv(tp)
            
            if df.empty or len(df.columns) == 0:
                logger.error(f"{feature_type.capitalize()} feature '{feature_name}' tmp file is empty: {tp}")
                failed.append(feature_name)
            else:
                dfs.append(df)
                successful.append(feature_name)
        except Exception as e:
            logger.error(f"Failed reading {feature_type} feature '{feature_name}' from {tp}: {e}")
            logger.exception(f"Failed reading {feature_type} tmp {tp}", exc_info=True)
            failed.append(feature_name)
    
    return dfs, successful, failed


def _merge_feature_dataframes(dfs, existing_df=None, drop_cols=None, feature_type="node"):
    """Merge feature dataframes (node or edge type)."""
    if existing_df is not None:
        try:
            existing_df = existing_df.drop(columns=(drop_cols or []), errors="ignore")
            dfs.insert(0, existing_df)
        except Exception:
            logger.exception(f"Failed to prepare existing {feature_type} df for merge")
    
    if not dfs:
        return None
    
    if feature_type == "node":
        merged = pd.concat(dfs, axis=1)
        cols_sorted = sorted(list(merged.columns))
        return merged[cols_sorted]
    else:  # edge
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=["source", "target"], how="outer")
        other_cols = sorted([c for c in merged.columns if c not in ("source", "target")])
        return merged[["source", "target"] + other_cols]


def _write_merged_csv(csv_path, df, gid, index_label=None):
    """Write merged dataframe to CSV atomically."""
    out_dir = os.path.dirname(csv_path)
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".csv.tmp")
    try:
        os.close(fd)
        if index_label:
            df.to_csv(tmp_path, index_label=index_label)
        else:
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)
        logger.info(f"Wrote {'node' if index_label else 'edge'} features for {gid} -> {csv_path} (shape={df.shape})")
        return True
    except Exception as e:
        logger.error(f"Failed writing features for {gid}: {e}")
        logger.exception(e, exc_info=True)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return False


def process_graph(graph_path, node_out_path, edge_out_path):
    """Compute node and edge features for a graph with partial recompute support."""
    G = load_graph(graph_path)
    force_flag = bool(getattr(process_graph, "force", False))
    recalc_missing = bool(getattr(process_graph, "recalc_missing", False))
    recalc_edge_features = set(getattr(process_graph, "recalc_edge_features", []) or [])
    
    directed_feats, undirected_feats = split_features_for_graph(G)
    out_dir = os.path.dirname(node_out_path)
    os.makedirs(out_dir, exist_ok=True)
    gid = os.path.basename(os.path.dirname(os.path.dirname(node_out_path)))
    
    node_exists = os.path.exists(node_out_path)
    edge_exists = os.path.exists(edge_out_path)
    
    # Plan node feature tasks
    node_feature_tasks = []
    existing_node_df = None
    drop_node_cols = []
    
    if not node_exists:
        for f in directed_feats:
            node_feature_tasks.append((graph_path, f, out_dir, True))
        for f in undirected_feats:
            node_feature_tasks.append((graph_path, f, out_dir, False))
    elif recalc_missing:
        meta_path = node_out_path.replace(".csv", ".metadata.json")
        missing_nodes = []
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                expected = set(meta.get("expected_features", []) or [])
                successful = set(meta.get("successful_features", []) or [])
                missing_nodes = sorted(expected - successful)
        except Exception:
            logger.exception(f"Failed to read node metadata for {gid}")
        
        if missing_nodes:
            for f in missing_nodes:
                node_feature_tasks.append((graph_path, f, out_dir, f in directed_feats))
            try:
                existing_node_df = pd.read_csv(node_out_path, index_col="node")
                drop_node_cols = [c for c in existing_node_df.columns 
                                 if any(c.startswith(feat) for feat in missing_nodes)]
            except Exception:
                logger.exception(f"Failed to load existing node CSV for {gid}")
        else:
            logger.info(f"Node features for {gid} already exist and complete; skipping")
    
    # Plan edge feature tasks
    edge_feature_keys = ["edge_betweenness", "edge_current_flow", "min_edge_cut"]
    edge_feature_tasks = []
    existing_edge_df = None
    drop_edge_cols = []
    
    if not edge_exists or force_flag:
        for k in edge_feature_keys:
            edge_feature_tasks.append((graph_path, k, os.path.dirname(edge_out_path)))
    else:
        features_to_compute = recalc_edge_features & set(edge_feature_keys)
        
        if recalc_missing:
            try:
                existing_edge_df = pd.read_csv(edge_out_path)
                existing_cols = set(existing_edge_df.columns)
                for k in edge_feature_keys:
                    if k not in existing_cols:
                        features_to_compute.add(k)
            except Exception:
                logger.exception(f"Failed to read existing edge CSV for {gid}")
        
        if features_to_compute:
            if existing_edge_df is None:
                try:
                    existing_edge_df = pd.read_csv(edge_out_path)
                except Exception:
                    pass
            drop_edge_cols = list(features_to_compute)
            for k in features_to_compute:
                edge_feature_tasks.append((graph_path, k, os.path.dirname(edge_out_path)))
        else:
            logger.info(f"Edge features for {gid} already exist and complete; skipping")
    
    # Run tasks
    all_tasks = [("node", t) for t in node_feature_tasks] + [("edge", t) for t in edge_feature_tasks]
    if not all_tasks:
        return
    
    try:
        cpu = multiprocessing.cpu_count() or 1
        fw_hint = getattr(process_graph, "feature_workers_hint", 1)
        feature_workers = min(cpu, max(1, int(fw_hint)))
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=feature_workers, initializer=_worker_init, initargs=(graph_path,)
        ) as ex:
            futures = {}
            for kind, task in all_tasks:
                if kind == "node":
                    futures[ex.submit(_compute_feature_worker_node, task)] = (kind, task)
                else:
                    feature_key = task[1]
                    logger.info(f"Starting edge feature {feature_key} for {gid} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    futures[ex.submit(_compute_feature_worker_edge, task)] = (kind, task)
            
            node_temp_paths = []
            edge_temp_paths = []
            for fut in concurrent.futures.as_completed(futures):
                kind, task = futures[fut]
                try:
                    r = fut.result()
                    if not r:
                        continue
                    if isinstance(r, str):
                        r = {"tmp_path": r}
                    
                    tp = r.get("tmp_path")
                    fk = r.get("feature_name") or r.get("feature_key")
                    dropped = r.get("dropped") or []
                    drop_msg = f", dropped={dropped}" if dropped else ""
                    duration_str = format_duration(r.get("duration", 0))
                    
                    if kind == "node":
                        node_temp_paths.append(tp)
                        if fk:
                            logger.info(f"Computed node feature {fk} for {gid}: {r.get('count','?')} entries in {duration_str}{drop_msg}")
                    else:
                        edge_temp_paths.append(tp)
                        s_ts, e_ts = r.get("start_ts"), r.get("end_ts")
                        if s_ts and e_ts:
                            logger.info(f"Edge feature {fk} for {gid}: started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(s_ts))}, "
                                      f"finished {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e_ts))}, "
                                      f"duration={duration_str}, rows={r.get('count','?')}{drop_msg}")
                        else:
                            logger.info(f"Computed edge feature {fk} for {gid}: {r.get('count','?')} entries in {duration_str}{drop_msg}")
                except Exception:
                    logger.exception(f"Error handling future for {gid} kind={kind} task={task}", exc_info=True)
        
        # Process node features
        if node_temp_paths:
            tmp_dfs, successful_node, failed_node = _read_tmp_features(node_temp_paths, "node")
            merged_node = _merge_feature_dataframes(tmp_dfs, existing_node_df, drop_node_cols, "node")
            
            if merged_node is not None:
                if _write_merged_csv(node_out_path, merged_node, gid, index_label="node"):
                    expected_node = sorted(list(set(directed_feats + undirected_feats)))
                    _write_feature_metadata(node_out_path, successful_node, failed_node, expected_node)
                    if failed_node:
                        logger.warning(f"⚠️  INCOMPLETE: Node features for {gid} missing {len(failed_node)} feature(s): {', '.join(failed_node)}")
                    _cleanup_feature_tmps(node_temp_paths, failed_node, "node")
            else:
                logger.warning(f"No node features produced for {gid}")
        
        # Process edge features
        if edge_temp_paths:
            tmp_dfs, successful_edge, failed_edge = _read_tmp_features(edge_temp_paths, "edge")
            merged_edge = _merge_feature_dataframes(tmp_dfs, existing_edge_df, drop_edge_cols, "edge")
            
            if merged_edge is not None:
                if _write_merged_csv(edge_out_path, merged_edge, gid):
                    _write_feature_metadata(edge_out_path, successful_edge, failed_edge, edge_feature_keys)
                    if failed_edge:
                        logger.warning(f"⚠️  INCOMPLETE: Edge features for {gid} missing {len(failed_edge)} feature(s): {', '.join(failed_edge)}")
                    _cleanup_feature_tmps(edge_temp_paths, failed_edge, "edge")
            elif failed_edge:
                logger.error(f"❌ FAILED: All edge features failed for {gid}. Failed features: {', '.join(failed_edge)}")
    
    except Exception:
        logger.exception(f"Exception while computing features for {gid}", exc_info=True)


def get_graph_feature_paths(collection, gid, base_dir="outputs"):
    graph_dir = os.path.join(base_dir, collection, gid, "features")
    node_path = os.path.join(graph_dir, "node_features.csv")
    edge_path = os.path.join(graph_dir, "edge_features.csv")
    return node_path, edge_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute node and edge features for a graph or for all entries in a manifest."
    )
    parser.add_argument("--graph_path", type=str, help="Path to graph pickle")
    parser.add_argument(
        "--gid",
        type=str,
        help="Graph ID for single-graph mode (derives outputs/[gid]/features/)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Optional collection name for output path (outputs/<collection>/<gid>/features).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="outputs",
        help="Base directory for feature outputs (default: outputs)",
    )
    parser.add_argument(
        "--manifest", type=str, help="Path to manifest JSON to process multiple graphs"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, force recalculation even when output CSVs exist (default is to skip existing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility (sets Python and NumPy RNG seeds). "
            "Note: the seed is set in the main process and (on Unix) is inherited by "
            "worker processes when using the default fork start method; if your "
            "environment uses the spawn start method (e.g. macOS or explicit "
            "`set_start_method('spawn')`), child processes will not inherit the "
            "seed and you should use explicit worker reseeding for portability."
        ),
    )
    parser.add_argument(
        "--recalc-missing",
        action="store_true",
        help="Recompute missing features (edge or node) detected via metadata/column checks without deleting existing CSVs.",
    )
    parser.add_argument(
        "--recalc-edge-features",
        nargs="+",
        help="Recompute specific edge features even if edge_features.csv exists (e.g., --recalc-edge-features edge_betweenness)",
        default=[],
    )
    parser.add_argument(
        "--no-skip-on-tmp",
        action="store_true",
        help="Disable skipping when per-feature tmp CSVs exist (default is to skip based on tmp files).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel graph workers (ProcessPoolExecutor). Default: min(cpu_count, number of graphs)",
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=1,
        help="Max parallel feature workers per graph. Default: 1",
    )
    parser.add_argument(
        "--no-blas-limit",
        action="store_true",
        help="Disable BLAS thread limiting (OPENBLAS_NUM_THREADS=1). "
        "By default, BLAS threads are limited to 1 to prevent thread oversubscription "
        "when using process-level parallelism. See THREADING_ANALYSIS.md for details. "
        "Use this flag if you want to use multi-threaded BLAS (not recommended for most cases).",
    )
    args = parser.parse_args()

    # propagate knobs to process_graph
    process_graph.recalc_missing = args.recalc_missing
    process_graph.recalc_edge_features = args.recalc_edge_features
    process_graph.force = args.force

    # Log BLAS threading configuration
    if _blas_limit_disabled:
        logger.warning(
            "BLAS thread limiting DISABLED (--no-blas-limit flag used or HIERARCHY_NO_BLAS_LIMIT set)"
        )
        logger.warning(
            "  WARNING: This may cause thread oversubscription and performance degradation!"
        )
    else:
        logger.info("BLAS thread limiting ENABLED (OPENBLAS_NUM_THREADS=1)")
        logger.info(
            "  This prevents thread oversubscription. Override with: --no-blas-limit"
        )

    # Apply random seed if provided (simple: seed main RNGs only).
    # Note: on Unix the fork-based ProcessPoolExecutor children inherit the
    # parent's RNG state at fork time; with spawn start method children do not
    # inherit the seeded state. See --seed help for details.

    try:
        random.seed(int(args.seed))
    except Exception:
        logger.warning("Failed to set Python random seed")
    try:
        np.random.seed(int(args.seed))
    except Exception:
        logger.warning("Failed to set NumPy random seed")

    # Convert single-graph mode to a temporary manifest-of-1 (consistent with edge_scores)
    if not args.manifest:
        if not args.graph_path or not args.gid:
            raise SystemExit("Provide either --manifest or both --graph_path and --gid")

        coll = args.collection or ""

        manifest_entry = {
            "graph_id": args.gid,
            "collection": coll,
            "G_path": args.graph_path,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([manifest_entry], f, indent=2)
            temp_manifest = f.name

        args.manifest = temp_manifest

    # At this point args.manifest is always set; process manifest entries
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
    changed = False
    # Build list of work items
    work_items = []
    for entry in manifest:
        coll = entry.get("collection") or ""
        gid = entry.get("graph_id")
        node_path, edge_path = get_graph_feature_paths(coll, gid, args.output_base)
        gpath = entry.get("G_path")
        if not gpath:
            continue
        # Skip only if everything exists AND no recompute requested
        node_exists = os.path.exists(node_path)
        edge_exists = os.path.exists(edge_path)
        recalc_requested = bool(args.recalc_missing) or bool(args.recalc_edge_features)
        if (not getattr(args, "force", False)) and node_exists and edge_exists and not recalc_requested:
            logger.info(
                f"Skipping {gid}: both feature files exist at {node_path} and {edge_path}"
            )
            entry["node_features_path"] = node_path
            entry["edge_features_path"] = edge_path
            changed = True
            continue

        work_items.append((entry, node_path, edge_path))

    # Process graphs in parallel using ProcessPoolExecutor
    if work_items:
        # propagate skip-on-tmp policy to worker processes via env var
        # Default behavior: skip per-feature work if tmp CSVs exist. Use
        # `--no-skip-on-tmp` to disable that behavior. `--force` always
        # overrides and disables skipping.
        if getattr(args, "force", False):
            os.environ.pop("HIERARCHY_SKIP_ON_TMP", None)
        elif getattr(args, "no_skip_on_tmp", False):
            os.environ.pop("HIERARCHY_SKIP_ON_TMP", None)
        else:
            os.environ["HIERARCHY_SKIP_ON_TMP"] = "1"

        # compute sensible default for graph-level workers: min(cpu_count, number_of_graphs)
        cpu = multiprocessing.cpu_count() or 1
        num_graphs = len(work_items)
        default_workers = max(1, min(cpu, num_graphs))
        if getattr(args, "workers", None) is None:
            max_workers = default_workers
        else:
            max_workers = max(1, int(args.workers))

        # Wire per-graph feature-workers hint into the process_graph function
        process_graph.feature_workers_hint = int(args.feature_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for entry, node_path, edge_path in work_items:
                futures[
                    ex.submit(process_graph, entry.get("G_path"), node_path, edge_path)
                ] = entry

            for fut in concurrent.futures.as_completed(futures):
                entry = futures[fut]
                try:
                    fut.result()
                    entry["node_features_path"] = get_graph_feature_paths(
                        entry.get("collection") or "",
                        entry.get("graph_id"),
                        args.output_base,
                    )[0]
                    entry["edge_features_path"] = get_graph_feature_paths(
                        entry.get("collection") or "",
                        entry.get("graph_id"),
                        args.output_base,
                    )[1]
                    changed = True
                except Exception:
                    logger.exception(
                        f"Graph processing failed for {entry.get('graph_id')}",
                        exc_info=True,
                    )
    if changed:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
