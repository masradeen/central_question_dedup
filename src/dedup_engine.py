# dedup_engine.py
# -------------------------------------------------------
# Main pipeline for question deduplication:
# - Load questions
# - Embed using SentenceTransformers
# - kNN candidate search
# - Similarity scoring
# - Save output
# -------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os
import json

from src.embedder import QuestionEmbedder


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATA_PATH = "data/raw_questions.csv"
EMB_PATH = "results/embeddings.npy"
PAIR_PATH = "results/similarity_pairs.csv"
CLUSTER_PATH = "results/clusters.json"
HEATMAP_PATH = "results/heatmap.png"

THRESHOLD = 0.78      # similarity threshold for duplicate
TOP_K = 10            # neighbors to examine


# -------------------------------------------------------
# 1. Load data
# -------------------------------------------------------
def load_questions():
    print("[Dedup] Loading questions from:", DATA_PATH)
    df = pd.read_csv(
        DATA_PATH, 
        sep=";",
        encoding="utf-8-sig")
    if "question_text" not in df.columns:
        raise ValueError("raw_questions.csv must have column: question_text")
    return df


# -------------------------------------------------------
# 2. Embedding
# -------------------------------------------------------
def generate_embeddings(df):
    embedder = QuestionEmbedder()
    emb = embedder.encode(df["question_text"].tolist())
    return emb


# -------------------------------------------------------
# 3. Candidate retrieval (kNN)
# -------------------------------------------------------
def find_candidates(embeddings, top_k=TOP_K):
    print(f"[Dedup] Running kNN candidate search (k={top_k})...")
    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return distances, indices


# -------------------------------------------------------
# 4. Compute similarity + extract duplicate pairs
# -------------------------------------------------------
def compute_pairs(df, embeddings, indices, threshold=THRESHOLD):
    print("[Dedup] Computing cosine similarity & extracting pairs...")

    sim_matrix = cosine_similarity(embeddings)
    pairs = []

    n = len(df)
    for i in range(n):
        for j in indices[i]:
            if i >= j:
                continue
            sim = sim_matrix[i][j]
            if sim >= threshold:
                pairs.append({
                    "id1": df.loc[i, "question_id"],
                    "id2": df.loc[j, "question_id"],
                    "survey1": df.loc[i, "survey_name"],
                    "survey2": df.loc[j, "survey_name"],
                    "question1": df.loc[i, "question_text"],
                    "question2": df.loc[j, "question_text"],
                    "similarity": float(sim)
                })

    return pairs, sim_matrix


# -------------------------------------------------------
# 5. Save similarity heatmap
# -------------------------------------------------------
def save_heatmap(sim_matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title("Similarity Matrix Heatmap")
    plt.savefig(HEATMAP_PATH, dpi=300)
    plt.close()
    print("[Dedup] Heatmap saved:", HEATMAP_PATH)


# -------------------------------------------------------
# 6. Run full pipeline
# -------------------------------------------------------
def main():
    os.makedirs("results", exist_ok=True)

    # Load text
    df = load_questions()

    # Embeddings
    embeddings = generate_embeddings(df)
    np.save(EMB_PATH, embeddings)
    print("[Dedup] Embeddings saved:", EMB_PATH)

    # Candidate search
    distances, indices = find_candidates(embeddings)

    # Duplicate pairs
    pairs, sim_matrix = compute_pairs(df, embeddings, indices)
    print(f"[Dedup] Found {len(pairs)} candidate duplicate pairs.")

    # Save pairs
    pd.DataFrame(pairs).to_csv(PAIR_PATH, index=False)
    print("[Dedup] Saved pairs to:", PAIR_PATH)

    # Save heatmap
    save_heatmap(sim_matrix)

    print("\n[Dedup] Pipeline completed successfully.\n")
    print("Next step: Run clustering.py to group duplicates.")


if __name__ == "__main__":
    main()
