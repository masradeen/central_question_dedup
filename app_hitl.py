# app_hitl.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from datetime import datetime

# ---------- Setup ----------
st.set_page_config(page_title="BPS Dedup + HITL", layout="wide")
st.title("📘 BPS Question Deduplication — HITL Dashboard")

# create directories
os.makedirs("results", exist_ok=True)
os.makedirs("results/feedback", exist_ok=True)

# ---------- Sidebar controls ----------
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload raw_questions.csv", type=["csv"])
model_name = st.sidebar.selectbox(
    "Embedding model",
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ),
)
threshold = st.sidebar.slider("Similarity threshold", 0.60, 0.95, 0.78, 0.01)
run_button = st.sidebar.button("Run Deduplication")

# Active learning / retrain note
st.sidebar.markdown("---")
st.sidebar.markdown("**HITL workflow:** verify pairs → save feedback → retrain offline (see instructions).")
st.sidebar.markdown("⚠️ If you previously uploaded files and they don't load, re‑upload them here.")

# ---------- Helper functions ----------
@st.cache_data
def load_model(name):
    return SentenceTransformer(name)

def compute_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True, batch_size=32)

def compute_pairs(df, emb, thr):
    sim = cosine_similarity(emb)
    pairs = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            s = float(sim[i, j])
            if s >= thr:
                pairs.append({
                    "id1": df.loc[i, "question_id"],
                    "id2": df.loc[j, "question_id"],
                    "survey1": df.loc[i, "survey_name"],
                    "survey2": df.loc[j, "survey_name"],
                    "question1": df.loc[i, "question_text"],
                    "question2": df.loc[j, "question_text"],
                    "similarity": round(s, 4),
                    "idx1": int(i),
                    "idx2": int(j)
                })
    return pairs, sim

def clusters_from_pairs(pairs):
    G = nx.Graph()
    for p in pairs:
        G.add_edge(p["id1"], p["id2"])
    clusters = [sorted(list(c)) for c in nx.connected_components(G)]
    return clusters

def save_heatmap(sim_matrix, path):
    plt.figure(figsize=(10,7))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title("Similarity Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------- Main panel ----------
if uploaded_file:
    df = pd.read_csv(uploaded_file,
        sep=";",
        encoding="utf-8-sig")
    st.markdown("### 🔎 Input preview")
    st.dataframe(df.head())

    # run dedup
    if run_button:
        with st.spinner("Loading model and encoding..."):
            model = load_model(model_name)
            texts = df["question_text"].astype(str).tolist()
            emb = compute_embeddings(model, texts)

        with st.spinner("Computing similarity and candidate pairs..."):
            pairs, sim_matrix = compute_pairs(df, emb, threshold)
            save_heatmap(sim_matrix, "results/heatmap.png")
            pd.DataFrame(pairs).to_csv("results/similarity_pairs.csv", index=False)
            np.save("results/embeddings.npy", emb)
            st.success(f"Found {len(pairs)} candidate pairs (sim >= {threshold})")
            st.write("Results saved to `results/`")
        
        # SAVE TO SESSION STATE
        st.session_state["pairs"] = pairs
        st.session_state["sim_matrix"] = sim_matrix
        st.session_state["embeddings"] = emb
        st.session_state["dedup_ran"] = True
        

        # layout: left = pairs verification, right = clusters & downloads
        left_col, right_col = st.columns([2,1])

        # ---------- Left: HITL pair verification ----------
        with left_col:
            st.subheader("🧾 Candidate Pairs — Human Verification")
            st.markdown("For each pair: choose label and add optional note. Click **Save Feedback** to persist.")
            # load existing feedback if any to prefill
            feedback_path = "results/feedback/labels.csv"
            if os.path.exists(feedback_path):
                feedback_df = pd.read_csv(feedback_path, dtype=str)
            else:
                feedback_df = pd.DataFrame(columns=["id1","id2","label","note","timestamp","similarity"])

            # Build UI for each pair
            verified_rows = []
            for idx, p in enumerate(pairs):
                with st.expander(f"Pair {idx+1} — {p['id1']} ↔ {p['id2']} (sim={p['similarity']})", expanded=False):
                    st.markdown(f"**Q1 ({p['id1']})**: {p['question1']}")
                    st.markdown(f"**Q2 ({p['id2']})**: {p['question2']}")
                    # prefill label if exists
                    existing = feedback_df[(feedback_df.id1==p["id1"]) & (feedback_df.id2==p["id2"])]
                    pre_label = existing["label"].values[0] if len(existing)>0 else "unlabeled"
                    pre_note = existing["note"].values[0] if len(existing)>0 else ""
                    label = st.radio(
                        "Label this pair",
                        options=["duplicate","near-duplicate/harmonizable","different","needs-review"],
                        index=0 if pre_label=="unlabeled" else ["duplicate","near-duplicate/harmonizable","different","needs-review"].index(pre_label)
                    )
                    note = st.text_input("Note / rationale (optional)", value=pre_note, key=f"note_{idx}")
                    # Save inline to verified_rows buffer (not auto-saved)
                    verified_rows.append({
                        "id1": p["id1"],
                        "id2": p["id2"],
                        "similarity": p["similarity"],
                        "label": label,
                        "note": note
                    })

            # Save feedback button
            if st.button("💾 Save Feedback"):
                # merge with existing
                new_fb = pd.DataFrame(verified_rows)
                new_fb["timestamp"] = datetime.utcnow().isoformat()
                # append or update
                if os.path.exists(feedback_path):
                    stored = pd.read_csv(feedback_path, dtype=str)
                    # remove any rows with same id1,id2 in stored then append new
                    for _, r in new_fb.iterrows():
                        mask = ~((stored["id1"]==r["id1"]) & (stored["id2"]==r["id2"]))
                        stored = stored[mask]
                    updated = pd.concat([stored, new_fb], ignore_index=True)
                else:
                    updated = new_fb
                updated.to_csv(feedback_path, index=False)
                st.success(f"Feedback saved to {feedback_path}")

        # ---------- Right: clusters, annotation, downloads ----------
        with right_col:
            st.subheader("🧩 Clusters & Annotation")
            clusters = clusters_from_pairs(pairs)
            st.write(f"Found {len(clusters)} clusters.")
            cluster_annotations_path = "results/feedback/cluster_annotations.json"
            # load existing cluster annotations
            if os.path.exists(cluster_annotations_path):
                with open(cluster_annotations_path, "r", encoding="utf-8") as f:
                    cluster_ann = json.load(f)
            else:
                cluster_ann = {}

            annotated_clusters = []
            for i, cl in enumerate(clusters):
                st.markdown(f"**Cluster {i+1}** — {len(cl)} items")
                # propose a canonical question (pick first occurence)
                candidate_questions = []
                for qid in cl:
                    # find row
                    row = df[df["question_id"]==qid]
                    if not row.empty:
                        candidate_questions.append(row.iloc[0]["question_text"])
                ann = cluster_ann.get(str(i), {})
                default_name = ann.get("cluster_name", "")
                default_can = ann.get("canonical_question", candidate_questions[0] if len(candidate_questions)>0 else "")
                name = st.text_input(f"Cluster {i+1} name", value=default_name, key=f"cl_name_{i}")
                can = st.selectbox(f"Canonical question (cluster {i+1})",
                                   options=candidate_questions,
                                   index=0 if default_can=="" else (candidate_questions.index(default_can) if default_can in candidate_questions else 0),
                                   key=f"cl_can_{i}")
                annotated_clusters.append({
                    "cluster_id": i,
                    "items": cl,
                    "cluster_name": name,
                    "canonical_question": can
                })

            if st.button("💾 Save Cluster Annotations"):
                with open(cluster_annotations_path, "w", encoding="utf-8") as f:
                    json.dump({str(c["cluster_id"]): c for c in annotated_clusters}, f, ensure_ascii=False, indent=2)
                st.success(f"Cluster annotations saved to {cluster_annotations_path}")

            st.markdown("---")
            st.subheader("📥 Download results")
            # download similarity pairs CSV
            pairs_df = pd.DataFrame(pairs)
            if not pairs_df.empty:
                csv_bytes = pairs_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download candidate pairs (CSV)", csv_bytes, file_name="similarity_pairs.csv", mime="text/csv")
            # download heatmap
            if os.path.exists("results/heatmap.png"):
                with open("results/heatmap.png", "rb") as f:
                    st.download_button("⬇️ Download heatmap (PNG)", f, file_name="heatmap.png", mime="image/png")
            # download embeddings (numpy) as binary
            if os.path.exists("results/embeddings.npy"):
                with open("results/embeddings.npy", "rb") as f:
                    st.download_button("⬇️ Download embeddings (npy)", f, file_name="embeddings.npy", mime="application/octet-stream")

            # feedback files
            fb_path = "results/feedback/labels.csv"
            if os.path.exists(fb_path):
                with open(fb_path, "rb") as f:
                    st.download_button("⬇️ Download feedback (CSV)", f, file_name="feedback_labels.csv", mime="text/csv")
            if os.path.exists(cluster_annotations_path):
                with open(cluster_annotations_path, "r", encoding="utf-8") as f:
                    st.download_button("⬇️ Download cluster annotations (JSON)", f.read(), file_name="cluster_annotations.json", mime="application/json")

else:
    st.info("Upload `raw_questions.csv` from the sidebar to start. Make sure file has columns: question_id,question_text,survey_name,directorate.")

# ---------- End ----------