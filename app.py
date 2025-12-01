# ============================================================
#  BPS Question Deduplication — HITL Dashboard (Clean Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# -----------------------------------------------
# PAGE SETUP
# -----------------------------------------------
st.set_page_config(page_title="BPS Deduplication HITL", layout="wide")
st.title("📘 BPS Question Deduplication — HITL Dashboard (Clean Version)")

# Ensure folders
os.makedirs("results", exist_ok=True)
os.makedirs("results/feedback", exist_ok=True)


# -----------------------------------------------
# HELPERS — CACHED
# -----------------------------------------------
@st.cache_data(show_spinner=False)
def load_model_cached(model_name):
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def embed_texts(model_name, texts):
    model = load_model_cached(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=32)

@st.cache_data(show_spinner=False)
def compute_similarity(df, embeddings, threshold):
    sim = cosine_similarity(embeddings)
    n = len(df)
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim[i, j])
            if score >= threshold:
                pairs.append({
                    "idx1": i,
                    "idx2": j,
                    "id1": df.loc[i, "question_id"],
                    "id2": df.loc[j, "question_id"],
                    "question1": df.loc[i, "question_text"],
                    "question2": df.loc[j, "question_text"],
                    "survey1": df.loc[i, "survey_name"],
                    "survey2": df.loc[j, "survey_name"],
                    "similarity": round(score, 4)
                })
    return pairs, sim

def cluster_components(pairs):
    G = nx.Graph()
    for p in pairs:
        G.add_edge(p["id1"], p["id2"])
    return [sorted(list(c)) for c in nx.connected_components(G)]

def save_heatmap(sim, path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(sim, cmap="viridis")
    plt.title("Similarity Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# -----------------------------------------------
# SIDEBAR
# -----------------------------------------------
st.sidebar.header("Configuration")

uploaded = st.sidebar.file_uploader("Upload raw_questions.csv", type=["csv"])

model_name = st.sidebar.selectbox(
    "Embedding model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ],
)

threshold = st.sidebar.slider("Similarity threshold", 0.60, 0.95, 0.78, 0.01)

run_btn = st.sidebar.button("Run Deduplication")


# -----------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------
if not uploaded:
    st.info("Upload `raw_questions.csv` via sidebar.")
    st.stop()

# Load CSV
df = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig")
st.subheader("📄 Input Preview")
st.dataframe(df.head())


# ================================
# STEP 1 — RUN DEDUP
# ================================
if run_btn:

    # ------------ embeddings ------------
    with st.spinner("Encoding questions…"):
        texts = df["question_text"].astype(str).tolist()
        embeddings = embed_texts(model_name, texts)

    # ------------ similarity pairs ------------
    with st.spinner("Computing similarity…"):
        pairs, sim = compute_similarity(df, embeddings, threshold)

        np.save("results/embeddings.npy", embeddings)
        pd.DataFrame(pairs).to_csv("results/similarity_pairs.csv", index=False)
        save_heatmap(sim, "results/heatmap.png")

    st.success(f"Found {len(pairs)} candidate pairs ≥ {threshold}")
    st.markdown("---")

    # layout
    left, right = st.columns([2, 1])


    # ================================
    # STEP 2 — HITL PAIR VERIFICATION
    # ================================
    with left:
        st.subheader("🧾 Step 2: Pair Verification (HITL)")

        fb_path = "results/feedback/labels.csv"
        if os.path.exists(fb_path):
            fb = pd.read_csv(fb_path, dtype=str)
        else:
            fb = pd.DataFrame(columns=["id1", "id2", "label", "note", "timestamp", "similarity"])

        verified = []

        for idx, p in enumerate(pairs):
            with st.expander(
                f"Pair {idx+1}: {p['id1']} ↔ {p['id2']}  (sim={p['similarity']})",
                expanded=False,
            ):

                st.write(f"**Q1 ({p['id1']})**: {p['question1']}")
                st.write(f"**Q2 ({p['id2']})**: {p['question2']}")

                # Prefill label & note
                prev = fb[(fb.id1 == p["id1"]) & (fb.id2 == p["id2"])]
                prev_label = prev["label"].values[0] if len(prev) > 0 else "unlabeled"
                prev_note = prev["note"].values[0] if len(prev) > 0 else ""

                label = st.radio(
                    "Choose label",
                    ["duplicate", "near-duplicate/harmonizable", "different", "needs-review"],
                    index=(
                        0 if prev_label == "unlabeled"
                        else ["duplicate", "near-duplicate/harmonizable", "different", "needs-review"].index(prev_label)
                    ),
                    key=f"label_{idx}",
                )

                note = st.text_input(
                    "Note (optional)",
                    value=prev_note,
                    key=f"note_{idx}",
                )

                verified.append({
                    "id1": p["id1"],
                    "id2": p["id2"],
                    "similarity": p["similarity"],
                    "label": label,
                    "note": note,
                })

        if st.button("💾 Save Feedback"):
            new_fb = pd.DataFrame(verified)
            new_fb["timestamp"] = datetime.utcnow().isoformat()

            if os.path.exists(fb_path):
                old = pd.read_csv(fb_path, dtype=str)
                for _, row in new_fb.iterrows():
                    mask = ~((old["id1"] == row["id1"]) & (old["id2"] == row["id2"]))
                    old = old[mask]
                merged = pd.concat([old, new_fb], ignore_index=True)
            else:
                merged = new_fb

            merged.to_csv(fb_path, index=False)
            st.success("Feedback saved.")


    # ================================
    # STEP 3 — CLUSTER ANNOTATIONS
    # ================================
    with right:
        st.subheader("🧩 Step 3: Cluster Annotation")

        clusters = cluster_components(pairs)
        st.write(f"Total clusters: **{len(clusters)}**")

        cl_path = "results/feedback/cluster_annotations.json"
        if os.path.exists(cl_path):
            cluster_ann = json.load(open(cl_path, "r", encoding="utf-8"))
        else:
            cluster_ann = {}

        annotated = []

        for i, cl in enumerate(clusters):
            st.markdown(f"**Cluster {i+1} — {len(cl)} items**")

            qtexts = [
                df[df.question_id == qid].iloc[0]["question_text"]
                for qid in cl
                if not df[df.question_id == qid].empty
            ]
            default_ann = cluster_ann.get(str(i), {})
            default_name = default_ann.get("cluster_name", "")
            default_can = default_ann.get("canonical_question", qtexts[0])

            name = st.text_input(f"Cluster name {i+1}", value=default_name, key=f"cl_name_{i}")
            canonical = st.selectbox(
                f"Canonical question (cluster {i+1})",
                options=qtexts,
                index=qtexts.index(default_can) if default_can in qtexts else 0,
                key=f"cl_can_{i}",
            )

            annotated.append({
                "cluster_id": i,
                "items": cl,
                "cluster_name": name,
                "canonical_question": canonical,
            })

        if st.button("💾 Save Cluster Annotations"):
            json.dump(
                {str(c["cluster_id"]): c for c in annotated},
                open(cl_path, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )
            st.success("Cluster annotations saved.")


        # ================================
        # DOWNLOADS
        # ================================
        st.markdown("---")
        st.subheader("📥 Downloads")

        # Pairs
        if os.path.exists("results/similarity_pairs.csv"):
            st.download_button(
                "⬇️ Download pairs CSV",
                open("results/similarity_pairs.csv", "rb"),
                file_name="similarity_pairs.csv",
            )

        # Heatmap
        if os.path.exists("results/heatmap.png"):
            st.download_button(
                "⬇️ Download heatmap",
                open("results/heatmap.png", "rb"),
                file_name="heatmap.png",
            )

        # Embeddings
        if os.path.exists("results/embeddings.npy"):
            st.download_button(
                "⬇️ Download embeddings",
                open("results/embeddings.npy", "rb"),
                file_name="embeddings.npy",
            )

        # Feedback files
        if os.path.exists(fb_path):
            st.download_button(
                "⬇️ Download feedback labels",
                open(fb_path, "rb"),
                file_name="feedback_labels.csv",
            )

        if os.path.exists(cl_path):
            st.download_button(
                "⬇️ Download cluster annotations",
                open(cl_path, "rb"),
                file_name="cluster_annotations.json",
            )
