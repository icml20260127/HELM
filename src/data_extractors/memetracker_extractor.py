"""
MemeTracker Data Extraction and Graph Construction

This module processes MemeTracker data to build:
1. Global hyperlink graph (G) from quote data
2. Cascade trees (T_k) for each phrase cluster

Graph Properties:
- G: Directed graph, no self-loops, no multi-edges
- T_k: Directed arborescence (rooted tree), subset of G
- Invariant: All T_k edges exist in final G (with augmentation)
"""

import gzip
import pickle
import json
import networkx as nx
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
import argparse
import sys
from math import inf
from src.utils import save_graph, load_graph

# Directory and filename constants
GLOBAL_DIR_NAME = "global"
CASCADES_DIR_NAME = "cascades"
PARSED_DIR_NAME = "parsed"

ENTITY_GRAPH_PKL = "entity_graph.pkl"
ENTITY_GRAPH_EDGELIST = "entity_graph.edgelist"
ENTITY_GRAPH_META = "entity_graph_meta.json"
ENTITY_GRAPH_STATS = "entity_graph_stats.json"
HOSTNAME_TO_ID = "hostname_to_id.json"
ID_TO_HOSTNAME = "id_to_hostname.json"

HIERARCHY_TREE_PKL = "hierarchy_tree.pkl"
HIERARCHY_TREE_EDGELIST = "hierarchy_tree.edgelist"
HIERARCHY_TREE_META = "hierarchy_tree_meta.json"
CASCADE_STATS = "cascade_stats.json"
CASCADE_MANIFEST = "manifest.json"


class MemeTrackerExtractor:
    """Extract and process MemeTracker data into graph structures."""

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        min_cascade_nodes: int = 100,
        max_cascade_depth: int = 20,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.min_cascade_nodes = min_cascade_nodes
        self.max_cascade_depth = max_cascade_depth

        # Output structure:
        # data/memetracker/
        #   global/
        #     entity_graph.pkl / entity_graph.edgelist / entity_graph_meta.json / entity_graph_stats.json
        #     hostname_to_id.json / id_to_hostname.json
        #   cascades/
        #     manifest.json
        #     cascade_00001/ {hierarchy_tree.pkl, hierarchy_tree.edgelist, hierarchy_tree_meta.json,
        #                     entity_graph.pkl, entity_graph.edgelist, cascade_stats.json}
        #     cascade_00002/
        #     ...
        #   parsed/  (intermediate cached artifacts)

        self.global_dir = self.output_dir / GLOBAL_DIR_NAME
        self.cascades_dir = self.output_dir / CASCADES_DIR_NAME
        self.parsed_dir = self.output_dir / PARSED_DIR_NAME

        for d in [self.output_dir, self.global_dir, self.cascades_dir, self.parsed_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.cluster_file_default = self.raw_dir / "clust-qt08080902w3mfq5.txt.gz"

        # Main data structures
        self.G = nx.Graph()  # Global hyperlink graph (UNDIRECTED, integer node IDs)
        self.hostname_to_id = {}  # hostname -> integer ID mapping
        self.id_to_hostname = {}  # integer ID -> hostname mapping
        self.next_id = 0  # Counter for assigning new IDs

        # Statistics
        self.stats = {
            "quotes_parsed": 0,
            "clusters_parsed": 0,
            "cascades_built": 0,
            "cascades_skipped_not_arborescence": 0,
            "cascades_skipped_too_small": 0,
            "cascades_skipped_too_deep": 0,
            "hyperlink_edges": 0,
            "cascade_augmented_edges": 0,
            "self_loops_skipped": 0,
            "multi_edges_merged": 0,
            "docs_skipped_no_hostname": 0,
            "docs_skipped_no_links": 0,
        }

    def get_or_create_node_id(self, hostname: str) -> int:
        """Get existing node ID or create new one for hostname."""
        if hostname not in self.hostname_to_id:
            self.hostname_to_id[hostname] = self.next_id
            self.id_to_hostname[self.next_id] = hostname
            self.next_id += 1
        return self.hostname_to_id[hostname]

    @staticmethod
    def extract_hostname(url: str) -> Optional[str]:
        """Extract hostname from URL, return None if invalid."""
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
            # Remove www. prefix
            if hostname.startswith("www."):
                hostname = hostname[4:]
            return hostname if hostname else None
        except:
            return None

    def parse_quotes_file(self, filepath: Path) -> Dict:
        """
        Parse a single quotes file (quotes_YYYY-MM.txt.gz).

        Returns:
            Dict with:
                - documents: List of (hostname, timestamp, phrases, links)
                - hyperlinks: List of (from_host, to_host) tuples
        """
        print(f"\n{'='*60}")
        print(f"Parsing: {filepath.name}")
        print(f"{'='*60}")

        documents = []
        hyperlinks = []

        current_doc = {}
        line_count = 0

        with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_count += 1

                if line_count % 1000000 == 0:
                    print(
                        f"  Processed {line_count:,} lines, {len(documents):,} documents..."
                    )

                line = line.strip()

                # Empty line = document separator
                if not line:
                    if current_doc:
                        # Skip docs with hostname issues or no links (links are essential)
                        if not current_doc.get("hostname"):
                            self.stats["docs_skipped_no_hostname"] += 1
                        elif not current_doc.get("links"):
                            self.stats["docs_skipped_no_links"] += 1
                        else:
                            documents.append(current_doc)
                        current_doc = {}
                    continue

                # Parse line
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue

                record_type, content = parts

                if record_type == "P":
                    # New document
                    hostname = self.extract_hostname(content)
                    if hostname:
                        current_doc = {
                            "hostname": hostname,
                            "url": content,
                            "timestamp": None,
                            "phrases": [],
                            "links": [],
                        }

                elif record_type == "T" and current_doc:
                    try:
                        current_doc["timestamp"] = datetime.strptime(
                            content, "%Y-%m-%d %H:%M:%S"
                        )
                    except:
                        pass

                elif record_type == "Q" and current_doc:
                    current_doc["phrases"].append(content)

                elif record_type == "L" and current_doc:
                    target_hostname = self.extract_hostname(content)
                    if target_hostname and target_hostname != current_doc["hostname"]:
                        current_doc["links"].append(target_hostname)
                        # Add to hyperlinks: reversed edge (linked → linker)
                        hyperlinks.append((target_hostname, current_doc["hostname"]))

        # Don't forget last document
        if current_doc:
            if not current_doc.get("hostname"):
                self.stats["docs_skipped_no_hostname"] += 1
            elif not current_doc.get("links"):
                self.stats["docs_skipped_no_links"] += 1
            else:
                documents.append(current_doc)

        print(f"  ✓ Parsed {len(documents):,} documents")
        print(f"  ✓ Found {len(hyperlinks):,} hyperlinks")

        return {"documents": documents, "hyperlinks": hyperlinks}

    def build_hyperlink_graph(self):
        """
        Build global hyperlink graph G from all quotes files.

        Edges: If document A links to B, add edge B → A (reversed)
        No self-loops, no multi-edges (count as weight)
        """
        print(f"\n{'='*60}")
        print("BUILDING HYPERLINK GRAPH (G)")
        print(f"{'='*60}")

        # Find all quotes files
        quotes_files = sorted(self.raw_dir.glob("quotes_*.txt.gz"))

        if not quotes_files:
            print("ERROR: No quotes files found!")
            return

        print(f"Found {len(quotes_files)} quotes files")

        edge_counts = defaultdict(int)

        # Process each file
        for quotes_file in quotes_files:
            data = self.parse_quotes_file(quotes_file)

            # Count documents
            self.stats["quotes_parsed"] += len(data["documents"])

            # Collect edges
            for from_host, to_host in data["hyperlinks"]:
                # Skip self-loops
                if from_host == to_host:
                    self.stats["self_loops_skipped"] += 1
                    continue

                edge_counts[(from_host, to_host)] += 1

        # Add edges to G (convert hostnames to integer IDs)
        # Make G bidirectional: if A links to B, add both A→B and B→A
        print(f"\nAdding edges to graph...")
        for (from_host, to_host), count in edge_counts.items():
            if count > 1:
                self.stats["multi_edges_merged"] += 1

            # Get or create node IDs
            from_id = self.get_or_create_node_id(from_host)
            to_id = self.get_or_create_node_id(to_host)

            # Single directed edge (from → to as in hyperlink semantics)
            self.G.add_edge(from_id, to_id)
            self.stats["hyperlink_edges"] += 1
        print(f"\n{'='*60}")
        print(f"HYPERLINK GRAPH (G) BUILT")
        print(f"{'='*60}")
        print(f"  Nodes: {self.G.number_of_nodes():,}")
        print(f"  Edges: {self.G.number_of_edges():,}")
        print(f"  Self-loops skipped: {self.stats['self_loops_skipped']:,}")
        print(f"  Multi-edges merged: {self.stats['multi_edges_merged']:,}")

        # Save G immediately (before cascade processing)
        self.save_graph_checkpoint()

    def save_graph_checkpoint(self):
        """Save entity_graph to disk (checkpoint before cascade processing)."""
        print(f"\nSaving entity_graph checkpoint...")

        # Save as pickle
        graph_path = self.global_dir / ENTITY_GRAPH_PKL
        save_graph(self.G, graph_path)
        print(f"  ✓ Saved: {graph_path}")

        # Save as edgelist
        edgelist_path = self.global_dir / ENTITY_GRAPH_EDGELIST
        nx.write_edgelist(self.G, edgelist_path, data=False)
        print(f"  ✓ Saved: {edgelist_path}")

        # Save hostname<->ID mappings
        mapping_path = self.global_dir / HOSTNAME_TO_ID
        with open(mapping_path, "w") as f:
            json.dump(self.hostname_to_id, f, indent=2)
        print(f"  ✓ Saved: {mapping_path}")

        id_mapping_path = self.global_dir / ID_TO_HOSTNAME
        with open(id_mapping_path, "w") as f:
            json.dump(self.id_to_hostname, f, indent=2)
        print(f"  ✓ Saved: {id_mapping_path}")

        # Save metadata
        self.save_entity_graph_meta()

        # Save stats
        self.save_graph_stats()

    def save_graph_stats(self):
        """Compute and save G statistics."""
        print(f"\nComputing graph statistics...")

        stats = {
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "avg_degree": sum(dict(self.G.degree()).values())
            / max(self.G.number_of_nodes(), 1),
            "density": nx.density(self.G),
            "num_connected_components": nx.number_connected_components(self.G),
        }

        # Degree distribution and top hubs
        degrees = [d for n, d in self.G.degree()]
        stats["degree_distribution"] = dict(Counter(degrees).most_common(20))

        # Top hubs by total degree
        deg_sorted = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        stats["top_hubs"] = [
            {
                "node_id": n,
                "hostname": self.id_to_hostname.get(n, "unknown"),
                "degree": d,
            }
            for n, d in deg_sorted[:20]
        ]

        stats["extraction_stats"] = self.stats

        # Save
        stats_path = self.global_dir / ENTITY_GRAPH_STATS
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Saved: {stats_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"  Nodes: {stats['num_nodes']:,}")
        print(f"  Edges: {stats['num_edges']:,}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")
        print(f"  Density: {stats['density']:.6f}")
        print(f"  Connected components: {stats['num_connected_components']:,}")
        print(f"\nTop Hubs:")
        for hub in stats["top_hubs"][:5]:
            print(
                f"    {hub['hostname']} (ID {hub['node_id']}): degree {hub['degree']}"
            )

    def save_entity_graph_meta(self):
        """Save entity_graph metadata (consistent with wiki/microbiome format)."""
        meta = {
            "extraction_date": datetime.now().isoformat(),
            "data_source": "MemeTracker (Stanford SNAP)",
            "graph_type": "entity_graph",
            "description": "Hyperlink-based directed graph of hostnames",
            "edge_semantics": "If A links to B, edge A→B (A points to B)",
            "num_nodes": self.G.number_of_nodes(),
            "num_edges": self.G.number_of_edges(),
            "properties": {"directed": True, "self_loops": False, "multi_edges": False},
        }

        meta_path = self.global_dir / ENTITY_GRAPH_META
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Saved: {meta_path}")

    # ------------------------------------------------------------------
    # Cascade processing
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_timestamp(token: str) -> bool:
        """Check if token matches YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."""
        try:
            datetime.strptime(token, "%Y-%m-%d")
            return True
        except Exception:
            pass
        try:
            datetime.strptime(token, "%Y-%m-%d %H:%M:%S")
            return True
        except Exception:
            return False

    def parse_cluster_file(self, cluster_path: Path) -> List[Dict]:
        """
        Parse the MemeTracker cluster file and return cascades.

        Each cascade dict:
            {
                'cluster_id': str,
                'root_phrase': str,
                'mentions': List[Tuple[hostname, timestamp]],
            }
        """
        print(f"\n{'='*60}")
        print(f"PARSING CLUSTER FILE: {cluster_path.name}")
        print(f"{'='*60}")

        cascades = []
        current = None

        def flush_current():
            if current and current.get("mentions"):
                cascades.append(current)

        with gzip.open(cluster_path, "rt", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.lower().startswith("format"):
                    continue

                parts = line.split("\t")
                if len(parts) == 1:
                    parts = line.split()

                # Cluster header (A): <ClSz> <TotFq> <Root> <ClusterId>
                if (
                    len(parts) >= 4
                    and parts[0].isdigit()
                    and parts[1].isdigit()
                    and parts[-1].isdigit()
                ):
                    flush_current()
                    root = " ".join(parts[2:-1]).strip()
                    cluster_id = parts[-1]
                    current = {
                        "cluster_id": cluster_id,
                        "root_phrase": root,
                        "mentions": [],
                    }
                    continue

                # Mention line (C): <Tm> <Fq> <UrlTy> <Url>
                if parts and self._looks_like_timestamp(parts[0]):
                    try:
                        # Expect tokens: date, time, freq, type, url
                        if len(parts) >= 5:
                            ts = datetime.strptime(
                                f"{parts[0]} {parts[1]}", "%Y-%m-%d %H:%M:%S"
                            )
                            url = parts[4]
                        elif len(parts) >= 4:
                            ts = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                            url = parts[3]
                        else:
                            continue

                        host = self.extract_hostname(url)
                        if host:
                            current.setdefault("mentions", []).append((host, ts))
                    except Exception:
                        continue
                    continue

                # Phrase line (B) is ignored for structure building
                continue

        flush_current()

        print(f"  ✓ Parsed {len(cascades):,} cascades")
        parsed_path = self.parsed_dir / "cascades_parsed.pkl"
        with open(parsed_path, "wb") as f:
            pickle.dump(cascades, f)
        print(f"  ✓ Saved parsed cascades: {parsed_path}")
        self.stats["clusters_parsed"] = len(cascades)
        return cascades

    def ensure_entity_graph_loaded(self):
        """Load entity_graph and mappings from disk if G is empty."""
        if self.G.number_of_nodes() == 0:
            graph_path = self.global_dir / ENTITY_GRAPH_PKL
            if not graph_path.exists():
                raise FileNotFoundError(f"entity_graph.pkl not found at {graph_path}")
            self.G = load_graph(graph_path)

            # Load hostname mappings
            mapping_path = self.global_dir / HOSTNAME_TO_ID
            with open(mapping_path, "r") as f:
                self.hostname_to_id = json.load(f)

            id_mapping_path = self.global_dir / ID_TO_HOSTNAME
            with open(id_mapping_path, "r") as f:
                # JSON keys are strings, convert back to int
                self.id_to_hostname = {int(k): v for k, v in json.load(f).items()}

            self.next_id = (
                max(self.id_to_hostname.keys()) + 1 if self.id_to_hostname else 0
            )

            print(
                f"Loaded entity_graph from {graph_path}: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges"
            )

    def compute_cascade_stats(
        self,
        T: nx.DiGraph,
        mentions_with_ids: List[Tuple[int, str, datetime]],
        augmented_edge_count: int,
        root_id: int,
    ) -> Dict:
        """Compute detailed statistics for a cascade tree.

        mentions_with_ids: list of tuples (node_id, hostname, timestamp)
        """
        num_nodes = T.number_of_nodes()
        num_edges = T.number_of_edges()

        # Tree structure
        depth = 0
        for node in T.nodes():
            dist = nx.shortest_path_length(T, root_id, node)
            depth = max(depth, dist)

        # Branching factor (max children only)
        max_children = max((T.out_degree(n) for n in T.nodes()), default=0)

        # Temporal analysis
        if len(mentions_with_ids) > 1:
            times = sorted([t for _, _, t in mentions_with_ids])
            temporal_span = (times[-1] - times[0]).total_seconds()
            temporal_span_days = temporal_span / 86400
        else:
            temporal_span = 0
            temporal_span_days = 0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "depth": depth,
            "max_children": max_children,
            "augmented_edges": augmented_edge_count,
            "augmented_ratio": (
                round(augmented_edge_count / num_edges, 3) if num_edges > 0 else 0
            ),
            "temporal_span_days": round(temporal_span_days, 2),
        }

    def generate_cascade_manifest(self, cascade_metadata: List[Dict]):
        """Generate manifest.json indexing all cascades (no filtering)."""
        print(f"\nGenerating cascade manifest...")

        # Aggregate totals
        total_nodes = sum(c.get("num_nodes", 0) for c in cascade_metadata)
        total_edges = sum(c.get("num_edges", 0) for c in cascade_metadata)
        total_augmented = sum(c.get("augmented_edges", 0) for c in cascade_metadata)
        avg_cascade_size = (
            total_nodes / len(cascade_metadata) if cascade_metadata else 0
        )

        manifest = {
            "dataset": "memetracker",
            "entity_graph_path": f"data/memetracker/{GLOBAL_DIR_NAME}/{ENTITY_GRAPH_PKL}",
            "num_cascades": len(cascade_metadata),
            "total_cascade_nodes": total_nodes,
            "total_cascade_edges": total_edges,
            "total_augmented_edges": total_augmented,
            "avg_cascade_size": round(avg_cascade_size, 2),
            "cascades": cascade_metadata,
        }

        manifest_path = self.cascades_dir / CASCADE_MANIFEST
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  ✓ Saved: {manifest_path}")
        print(f"  ✓ Total cascades: {len(cascade_metadata):,}")
        print(f"  ✓ Total augmented edges: {total_augmented:,}")

    def build_cascade_trees(self, cascades: List[Dict]):
        """
        Build cascade trees (arborescences) for each cascade.
        Ensures T_k ⊆ G by augmenting G with missing edges.
        """
        print(f"\n{'='*60}")
        print("BUILDING CASCADE TREES")
        print(f"{'='*60}")

        self.ensure_entity_graph_loaded()

        cascade_metadata = []  # For manifest
        augmented_edges_before = self.stats["cascade_augmented_edges"]

        for idx, cascade in enumerate(cascades, start=1):
            cascade_id = f"cascade_{idx:05d}"
            mentions = cascade.get("mentions", [])
            if not mentions:
                continue

            # Sort mentions by timestamp and convert to node IDs
            mentions_sorted = sorted(mentions, key=lambda x: x[1])
            mentions_with_ids = [
                (self.get_or_create_node_id(host), host, timestamp)
                for host, timestamp in mentions_sorted
            ]

            # Get unique node IDs (preserve order)
            unique_ids = list(
                dict.fromkeys([node_id for node_id, _, _ in mentions_with_ids])
            )

            # Build tree (enforce arborescence: each node has exactly one parent)
            T = nx.DiGraph()
            T.add_nodes_from(unique_ids)

            root_id = unique_ids[0]
            augmented_count_before = self.stats["cascade_augmented_edges"]

            for child_idx in range(1, len(mentions_with_ids)):
                child_id, child_host, child_time = mentions_with_ids[child_idx]

                # Skip root node entirely
                if child_id == root_id:
                    continue

                # Skip if child already has a parent (check T itself, handles nodes appearing multiple times)
                if T.in_degree(child_id) > 0:
                    continue

                # Find best parent among earlier mentions with an edge in G
                best_parent_id = None
                best_gap = inf
                for parent_id, parent_host, parent_time in mentions_with_ids[
                    :child_idx
                ]:
                    if parent_id == child_id:
                        continue
                    # For directed graph: edge must point FROM parent TO child
                    if self.G.has_edge(parent_id, child_id):
                        gap = (child_time - parent_time).total_seconds()
                        if 0 <= gap < best_gap:
                            best_gap = gap
                            best_parent_id = parent_id

                # If no edge in G, connect to most recent prior mention (augment G)
                if best_parent_id is None:
                    best_parent_id = None
                    best_gap = inf
                    for parent_id, parent_host, parent_time in mentions_with_ids[
                        :child_idx
                    ]:
                        if parent_id == child_id:
                            continue
                        gap = (child_time - parent_time).total_seconds()
                        if 0 <= gap < best_gap:
                            best_gap = gap
                            best_parent_id = parent_id

                    if best_parent_id is not None:
                        if not self.G.has_edge(best_parent_id, child_id):
                            self.G.add_edge(best_parent_id, child_id)
                            self.stats["cascade_augmented_edges"] += 1

                # Add edge to tree
                if best_parent_id is not None:
                    T.add_edge(best_parent_id, child_id)

            # Count augmented edges for this specific cascade
            cascade_augmented_edges = (
                self.stats["cascade_augmented_edges"] - augmented_count_before
            )

            # Compute cascade statistics for filtering (flat dict)
            cascade_stats = self.compute_cascade_stats(
                T, mentions_with_ids, cascade_augmented_edges, root_id
            )

            # Apply filters: min nodes and max depth
            num_nodes = T.number_of_nodes()
            depth = cascade_stats["depth"]
            if num_nodes < self.min_cascade_nodes:
                self.stats["cascades_skipped_too_small"] += 1
                continue
            if depth > self.max_cascade_depth:
                self.stats["cascades_skipped_too_deep"] += 1
                continue

            # Save cascade tree
            cascade_dir = self.cascades_dir / cascade_id
            cascade_dir.mkdir(parents=True, exist_ok=True)

            tree_path = cascade_dir / HIERARCHY_TREE_PKL
            save_graph(T, tree_path)

            edgelist_path = cascade_dir / HIERARCHY_TREE_EDGELIST
            nx.write_edgelist(T, edgelist_path, data=False)

            # Create and save induced subgraph (as entity_graph to match convention)
            induced_G = self.G.subgraph(T.nodes()).copy()
            induced_path = cascade_dir / ENTITY_GRAPH_PKL
            save_graph(induced_G, induced_path)

            induced_edgelist_path = cascade_dir / ENTITY_GRAPH_EDGELIST
            nx.write_edgelist(induced_G, induced_edgelist_path, data=False)

            # Save hierarchy tree metadata
            meta = {
                "cluster_id": cascade.get("cluster_id"),
                "root_phrase": cascade.get("root_phrase"),
                "num_nodes": T.number_of_nodes(),
                "num_edges": T.number_of_edges(),
                "is_arborescence": nx.is_arborescence(T),
                "root_id": root_id,
                "root_hostname": self.id_to_hostname.get(root_id),
            }
            meta_path = cascade_dir / HIERARCHY_TREE_META
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Save induced subgraph metadata
            induced_meta = {
                "extraction_date": datetime.now().isoformat(),
                "graph_type": "entity_graph",
                "description": "Induced subgraph of global entity_graph restricted to cascade nodes",
                "num_nodes": induced_G.number_of_nodes(),
                "num_edges": induced_G.number_of_edges(),
                "parent_cascade": cascade_id,
            }
            induced_meta_path = cascade_dir / ENTITY_GRAPH_META
            with open(induced_meta_path, "w") as f:
                json.dump(induced_meta, f, indent=2)  # Save cascade stats
            stats_path = cascade_dir / CASCADE_STATS
            with open(stats_path, "w") as f:
                json.dump(cascade_stats, f, indent=2)

            # Track for manifest
            cascade_metadata.append(
                {
                    "cascade_id": cascade_id,
                    "cluster_id": cascade.get("cluster_id"),
                    "root_phrase": cascade.get("root_phrase"),
                    "root_id": root_id,
                    "num_nodes": T.number_of_nodes(),
                    "num_edges": T.number_of_edges(),
                    "is_arborescence": nx.is_arborescence(T),
                    "augmented_edges": cascade_augmented_edges,
                    "augmented_ratio": cascade_stats.get("augmented_ratio", 0),
                    "depth": cascade_stats.get("depth", 0),
                    "path": f"{CASCADES_DIR_NAME}/{cascade_id}",
                    "stats_file": f"{CASCADES_DIR_NAME}/{cascade_id}/{CASCADE_STATS}",
                }
            )

            self.stats["cascades_built"] += 1

        # Generate manifest
        self.generate_cascade_manifest(cascade_metadata)

        # After augmentation, resave entity_graph to capture new edges
        self.save_graph_checkpoint()

        # Print filtering summary
        print(f"\n{'='*60}")
        print("CASCADE FILTERING SUMMARY")
        print(f"{'='*60}")
        print(f"  Parsed from clusters: {self.stats['clusters_parsed']:,}")
        print(
            f"  Skipped (not arborescence): {self.stats['cascades_skipped_not_arborescence']:,}"
        )
        print(
            f"  Skipped (< {self.min_cascade_nodes} nodes): {self.stats['cascades_skipped_too_small']:,}"
        )
        print(
            f"  Skipped (> {self.max_cascade_depth} depth): {self.stats['cascades_skipped_too_deep']:,}"
        )
        print(f"  Built and saved: {self.stats['cascades_built']:,}")
        print(
            f"  Augmented edges added to G: {self.stats['cascade_augmented_edges']:,}"
        )


def main():
    parser = argparse.ArgumentParser(description="Extract MemeTracker graphs")

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/memetracker/raw",
        help="Directory with raw MemeTracker data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/memetracker",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--cluster-file",
        type=str,
        default=None,
        help="Path to cluster file (clust-*.txt.gz). Defaults to raw_dir/clust-qt08080902w3mfq5.txt.gz",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["hyperlink", "cascade", "all"],
        default="all",
        help="Which step to run",
    )
    parser.add_argument(
        "--min-cascade-nodes",
        type=int,
        default=100,
        help="Minimum nodes for cascade trees (default 100)",
    )
    parser.add_argument(
        "--max-cascade-depth",
        type=int,
        default=20,
        help="Maximum depth for cascade trees (default 20)",
    )

    args = parser.parse_args()

    extractor = MemeTrackerExtractor(
        args.raw_dir,
        args.output_dir,
        min_cascade_nodes=args.min_cascade_nodes,
        max_cascade_depth=args.max_cascade_depth,
    )

    if args.step in ["hyperlink", "all"]:
        extractor.build_hyperlink_graph()

    if args.step in ["cascade", "all"]:
        cluster_path = (
            Path(args.cluster_file)
            if args.cluster_file
            else extractor.cluster_file_default
        )

        # If only running cascade step and parsed cascades already exist, reuse them
        parsed_pickle = extractor.parsed_dir / "cascades_parsed.pkl"
        if args.step == "cascade" and parsed_pickle.exists():
            with open(parsed_pickle, "rb") as f:
                cascades = pickle.load(f)
            print(
                f"Loaded parsed cascades from {parsed_pickle} ({len(cascades):,} cascades)"
            )
        else:
            cascades = extractor.parse_cluster_file(cluster_path)

        extractor.build_cascade_trees(cascades)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
