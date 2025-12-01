# app_lazy_clusters.py
"""
BPS Deduplication — Lazy-loading per cluster (Streamlit)
Run: streamlit run app_lazy_clusters.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="BPS Dedup (Lazy Clusters)", layout="wide")
st.title("📘 BPS Question Deduplication — Lazy Clusters")

os.makedirs("results", exist_ok=True)
os.makedirs("results/feedback", exist_ok=True)

# -------------------------
# Cached helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_model_cached(name):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def embed_texts(model_name, texts):
    model = load_model_cached(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=32)

@st.cache_data(show_spinner=False)
def compute_similarity_pairs(df, embeddings, threshold):
    sim = cosine_similarity(embeddings)
    pairs = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim[i, j])
            if score >= threshold:
                pairs.append({
                    "idx1": i, "idx2": j,
                    "id1": df.loc[i, "question_id"],
                    "id2": df.loc[j, "question_id"],
                    "question1": df.loc[i, "question_text"],
                    "question2": df.loc[j, "question_text"],
                    "survey1": df.loc[i, "survey_name"],
                    "survey2": df.loc[j, "survey_name"],
                    "similarity": round(score, 4)
                })
    return pairs, sim

def clusters_from_pairs(pairs):
    G = nx.Graph()
    for p in pairs:
        G.add_edge(p["id1"], p["id2"])
    return [sorted(list(c)) for c in nx.connected_components(G)]

def save_heatmap(sim, path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim, cmap="viridis")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Configuration")
uploaded = st.sidebar.file_uploader("Upload raw_questions.csv", type=["csv"])
model_name = st.sidebar.selectbox(
    "Embedding model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ],
)
threshold = st.sidebar.slider("Similarity threshold", 0.60, 0.95, 0.78, 0.01)
run_btn = st.sidebar.button("Run Deduplication")
st.sidebar.markdown("---")
st.sidebar.markdown("Lazy cluster loading: click **Load cluster** per cluster to render its rows.")

# -------------------------
# Basic checks
# -------------------------
if not uploaded:
    st.info("Upload `raw_questions.csv` to start.")
    st.stop()

# Read CSV (try common encodings/delimiters robustly)
try:
    df = pd.read_csv(uploaded, sep=",", encoding="utf-8-sig")
    if "question_text" not in df.columns:
        # try semicolon fallback
        df = pd.read_csv(uploaded, sep=";", encoding="latin1")
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig", on_bad_lines="skip", engine="python")

st.subheader("Input preview")
st.dataframe(df.head())

# -------------------------
# Run pipeline
# -------------------------
if run_btn:
    # embed
    with st.spinner("Encoding questions..."):
        texts = df["question_text"].astype(str).tolist()
        embeddings = embed_texts(model_name, texts)
        np.save("results/embeddings.npy", embeddings)

    # pairs
    with st.spinner("Computing similarity pairs..."):
        pairs, sim = compute_similarity_pairs(df, embeddings, threshold)
        pd.DataFrame(pairs).to_csv("results/similarity_pairs.csv", index=False)
        save_heatmap(sim, "results/heatmap.png")

    st.success(f"Found {len(pairs)} candidate pairs (sim ≥ {threshold})")
    st.markdown("---")

    # -------------------------
    # Layout: left pairs (minimal), right clusters summary
    # -------------------------
    left, right = st.columns([2, 1])

    # Left: show limited pairs summary (not the full HITL UI here)
    with left:
        st.subheader("Candidate Pairs (summary)")
        if len(pairs) == 0:
            st.info("No pairs found at this threshold.")
        else:
            pairs_df = pd.DataFrame(pairs)
            # show top 200 pairs for quick review
            show_n = min(len(pairs_df), 200)
            st.dataframe(pairs_df[["id1", "id2", "similarity"]].head(show_n))
            st.markdown(f"Showing first {show_n} pairs. Use downloads or cluster loader for deeper inspection.")

            if st.button("🔽 Full pairs CSV"):
                st.download_button("Download all pairs", pairs_df.to_csv(index=False).encode("utf-8"), file_name="similarity_pairs.csv", mime="text/csv", key="dl_pairs")

    # Right: clusters summary with lazy load (buttons)
    with right:
        st.subheader("Clusters (lazy load)")
        clusters = clusters_from_pairs(pairs)
        st.write(f"Total clusters: **{len(clusters)}**")

        # session state guard for loaded clusters
        if "loaded_clusters" not in st.session_state:
            st.session_state["loaded_clusters"] = {}

        # show clusters in condensed form with load/unload buttons
        for i, cl in enumerate(clusters):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Cluster {i+1}** — {len(cl)} items")
            with col2:
                load_key = f"load_cluster_{i}"
                if st.session_state["loaded_clusters"].get(str(i), False):
                    if st.button("Unload", key=f"unload_{i}"):
                        st.session_state["loaded_clusters"][str(i)] = False
                else:
                    if st.button("Load cluster", key=load_key):
                        st.session_state["loaded_clusters"][str(i)] = True

            # If requested, render full cluster rows (lazy)
            if st.session_state["loaded_clusters"].get(str(i), False):
                # build DataFrame of items in this cluster
                rows = []
                for qid in cl:
                    row = df[df["question_id"] == qid]
                    if not row.empty:
                        rows.append({
                            "question_id": qid,
                            "question_text": row.iloc[0]["question_text"],
                            "survey_name": row.iloc[0].get("survey_name", ""),
                            "directorate": row.iloc[0].get("directorate", "")
                        })
                if rows:
                    cluster_df = pd.DataFrame(rows)
                    st.dataframe(cluster_df, use_container_width=True)
                else:
                    st.info("No matching rows found for this cluster (maybe ID mismatch).")

        st.markdown("---")
        st.subheader("Cluster tools")
        ann_path = "results/feedback/cluster_annotations.json"
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            st.write("Existing cluster annotations loaded.")
            if st.button("Show saved annotations"):
                st.json(ann)
        else:
            st.info("No saved cluster annotations yet.")

        # Save current loaded clusters as a quick snapshot
        if st.button("Save loaded clusters snapshot"):
            snapshot = {k: clusters[int(k)] for k, v in st.session_state["loaded_clusters"].items() if v}
            with open("results/feedback/loaded_clusters_snapshot.json", "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            st.success("Snapshot saved to results/feedback/loaded_clusters_snapshot.json")

    # Downloads (global)
    st.markdown("---")
    st.subheader("Downloads")
    if os.path.exists("results/heatmap.png"):
        with open("results/heatmap.png", "rb") as f:
            st.download_button("⬇️ Download heatmap", f, file_name="heatmap.png", mime="image/png")
    if os.path.exists("results/embeddings.npy"):
        with open("results/embeddings.npy", "rb") as f:
            st.download_button("⬇️ Download embeddings", f, file_name="embeddings.npy", mime="application/octet-stream")
    if os.path.exists("results/similarity_pairs.csv"):
        with open("results/similarity_pairs.csv", "rb") as f:
            st.download_button("⬇️ Download pairs CSV", f, file_name="similarity_pairs.csv", mime="text/csv")

else:
    st.info("Press 'Run Deduplication' to compute embeddings and clusters.")
