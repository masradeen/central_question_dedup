import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import io
import json

st.set_page_config(page_title="BPS Question Deduplication", layout="wide")

# Title
st.title("📊 BPS Question Deduplication Dashboard")
st.write("Deteksi duplikasi pertanyaan antar-survei menggunakan semantic similarity.")

# Sidebar
st.sidebar.header("⚙ Pengaturan")

uploaded_file = st.sidebar.file_uploader("Upload raw_questions.csv", type="csv")

model_name = st.sidebar.selectbox(
    "Pilih Model Embedding",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L12-v2"
    ]
)

threshold = st.sidebar.slider("Similarity Threshold", 0.60, 0.95, 0.78, 0.01)

run_button = st.sidebar.button("🚀 Run Deduplication")

# Process jika file ada
if uploaded_file:
    df = pd.read_csv(
        uploaded_file,
        sep=";",
        encoding="utf-8-sig")

    st.write("### 📝 Preview Pertanyaan")
    st.dataframe(df.head())

    if run_button:
        # Load model
        st.write(f"### 🔍 Menggunakan Model: `{model_name}`")
        model = SentenceTransformer(model_name)

        # Embedding
        with st.spinner("Menghitung embeddings..."):
            emb = model.encode(df["question_text"].tolist(), show_progress_bar=True)

        # Similarity
        with st.spinner("Menghitung similarity matrix..."):
            sim_matrix = cosine_similarity(emb)

        # Heatmap
        st.write("### 🔥 Similarity Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(sim_matrix, cmap="viridis")
        st.pyplot(fig)

        # Save heatmap
        heatmap_buffer = io.BytesIO()
        fig.savefig(heatmap_buffer, format="png")
        heatmap_buffer.seek(0)

        # Duplicate Pair Detection
        st.write("### 📌 Duplicate Pairs")
        pairs = []
        n = len(df)

        for i in range(n):
            for j in range(i+1, n):
                sim = sim_matrix[i][j]
                if sim >= threshold:
                    pairs.append({
                        "id1": df.loc[i, "question_id"],
                        "id2": df.loc[j, "question_id"],
                        "question1": df.loc[i, "question_text"],
                        "question2": df.loc[j, "question_text"],
                        "similarity": sim
                    })

        pairs_df = pd.DataFrame(pairs)
        st.dataframe(pairs_df)

        # Clustering
        st.write("### 🧩 Clusters")
        if len(pairs_df) > 0:
            G = nx.Graph()
            for _, row in pairs_df.iterrows():
                G.add_edge(row["id1"], row["id2"])

            clusters = [sorted(list(c)) for c in nx.connected_components(G)]
            for i, c in enumerate(clusters):
                st.write(f"**Cluster {i+1}:** {c}")

            # Save cluster JSON
            clusters_json = json.dumps({"clusters": clusters}, indent=4)
        else:
            st.write("Tidak ada clusters.")

        # Download Buttons
        st.write("## 📥 Download Hasil")

        # 1. Duplicate Pairs CSV
        csv_buffer = pairs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Duplicate Pairs (CSV)",
            data=csv_buffer,
            file_name="similarity_pairs.csv",
            mime="text/csv"
        )

        # 2. Clusters JSON
        st.download_button(
            label="⬇️ Download Clusters (JSON)",
            data=clusters_json,
            file_name="clusters.json",
            mime="application/json"
        )

        # 3. Heatmap PNG
        st.download_button(
            label="⬇️ Download Heatmap (PNG)",
            data=heatmap_buffer,
            file_name="heatmap.png",
            mime="image/png"
        )

        st.success("Pipeline selesai dijalankan!")
else:
    st.info("Silakan upload file `raw_questions.csv` untuk memulai.")
