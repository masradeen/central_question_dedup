# app_pairs_pagination.py
"""
BPS Deduplication — Pairs Pagination Viewer (Streamlit)
Run: streamlit run app_pairs_pagination.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------
# Page setup
# -------------------------------------------------------
st.set_page_config(page_title="BPS Dedup Pagination", layout="wide")
st.title("📘 BPS Question Deduplication — Pairs Pagination Viewer")

os.makedirs("results", exist_ok=True)

# -------------------------------------------------------
# Cached Embedding + Similarity
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_model_cached(name):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def embed_texts(model_name, texts):
    model = load_model_cached(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=32)

@st.cache_data(show_spinner=False)
def compute_pairs(df, embeddings, threshold):
    sim = cosine_similarity(embeddings)
    pairs = []
    n = len(df)
    for i in range(n):
        for j in range(i+1, n):
            s = float(sim[i, j])
            if s >= threshold:
                pairs.append({
                    "idx1": i, "idx2": j,
                    "id1": df.loc[i, "question_id"],
                    "id2": df.loc[j, "question_id"],
                    "question1": df.loc[i, "question_text"],
                    "question2": df.loc[j, "question_text"],
                    "survey1": df.loc[i, "survey_name"],
                    "survey2": df.loc[j, "survey_name"],
                    "similarity": round(s, 4),
                })
    return pairs

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Configuration")
uploaded = st.sidebar.file_uploader("Upload raw_questions.csv", type=["csv"])

model_name = st.sidebar.selectbox(
    "Embedding model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ],
)

threshold = st.sidebar.slider("Similarity threshold", 0.60, 0.95, 0.80, 0.01)
run_btn = st.sidebar.button("Run Deduplication")

# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------
if not uploaded:
    st.info("Upload raw_questions.csv to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded, sep=";", encoding="utf-8-sig")
except:
    df = pd.read_csv(uploaded, encoding="latin1")

st.subheader("Input preview")
st.dataframe(df.head())

# -------------------------------------------------------
# Run pipeline
# -------------------------------------------------------
if run_btn:
    with st.spinner("Computing embeddings..."):
        texts = df["question_text"].astype(str).tolist()
        embeddings = embed_texts(model_name, texts)
        np.save("results/embeddings.npy", embeddings)

    with st.spinner("Computing similarity pairs..."):
        pairs = compute_pairs(df, embeddings, threshold)
        pd.DataFrame(pairs).to_csv("results/similarity_pairs.csv", index=False)

    st.success(f"Found **{len(pairs)}** candidate pairs (sim ≥ {threshold})")
    st.markdown("---")

    # ---------------------------------------------------
    # Pagination State
    # ---------------------------------------------------
    
    if "page" not in st.session_state:
        st.session_state.page = 1

    PER_PAGE = st.sidebar.number_input("Pairs per page", 10, 200, 50)

    total_pairs = len(pairs)
    total_pages = max(1, (total_pairs + PER_PAGE - 1) // PER_PAGE)

    # Navigation controls
    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])

    with nav_left:
        if st.button("⬅️ Prev") and st.session_state.page > 1:
            st.session_state.page -= 1

    with nav_right:
        if st.button("Next ➡️") and st.session_state.page < total_pages:
            st.session_state.page += 1

    with nav_mid:
        jump = st.number_input("Go to page", 1, total_pages, st.session_state.page)
        if jump != st.session_state.page:
            st.session_state.page = jump

