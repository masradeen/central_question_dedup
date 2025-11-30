# clustering.py
# -------------------------------------------------------
# Clustering of duplicate question pairs into groups.
# Supports:
# - Graph-based connected components clustering
# - Optional DBSCAN clustering from embeddings
# -------------------------------------------------------

import pandas as pd
import json
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from src.embedder import QuestionEmbedder

PAIR_PATH = "results/similarity_pairs.csv"
EMB_PATH = "results/embeddings.npy"
CLUSTER_PATH = "results/clusters.json"


# -------------------------------------------------------
# 1. Graph-based clustering (fast & works very well)
# -------------------------------------------------------
def cluster_with_graph(pairs_df):
    print("[Cluster] Building graph connected components...")

    G = nx.Graph()

    # add edges
    for _, row in pairs_df.iterrows():
        G.add_edge(row["id1"], row["id2"], weight=row["similarity"])

    clusters = list(nx.connected_components(G))

    # convert to list of sorted lists
    clusters = [sorted(list(c)) for c in clusters]
    print(f"[Cluster] Formed {len(clusters)} clusters.")
    return clusters


# -------------------------------------------------------
# 2. DBSCAN clustering (optional)
#    Good for fine-grained thematic grouping
# -------------------------------------------------------
def cluster_with_dbscan(embeddings, eps=0.25, min_samples=2):
    print("[Cluster] Running DBSCAN...")
    db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples).fit(embeddings)
    labels = db.labels_
    return labels


# -------------------------------------------------------
# 3. Save cluster dictionary
# -------------------------------------------------------
def save_clusters(clusters, output_path=CLUSTER_PATH):
    with open(output_path, "w") as f:
        json.dump({"clusters": clusters}, f, indent=4, ensure_ascii=False)
    print("[Cluster] Saved clusters to:", output_path)


# -------------------------------------------------------
# 4. Run full clustering pipeline
# -------------------------------------------------------
def main():
    print("[Cluster] Loading duplicate pairs:", PAIR_PATH)
    pairs = pd.read_csv(PAIR_PATH)

    if len(pairs) == 0:
        print("[Cluster] No duplicate pairs found. Nothing to cluster.")
        return

    # GRAPH CLUSTERING (recommended for BPS)
    clusters = cluster_with_graph(pairs)

    # Save JSON
    save_clusters(clusters)

    print("\n[Cluster] Clustering completed successfully.")
    print(f"[Cluster] Total clusters: {len(clusters)}")
    print("Inspect results/clusters.json\n")


if __name__ == "__main__":
    main()
