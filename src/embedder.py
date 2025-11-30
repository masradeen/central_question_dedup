# embedder.py
# -----------------------------------------
# Module for creating semantic embeddings
# using Sentence-Transformers
# -----------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class QuestionEmbedder:
    """
    Wrapper class for SentenceTransformer embeddings.
    """

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"[Embedder] Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences, batch_size=32, show_progress=True):
        """
        Convert list of sentences into semantic embeddings.
        """
        print("[Embedder] Encoding sentences...")
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        return embeddings

    @staticmethod
    def save_embeddings(path, embeddings):
        """
        Save embeddings to .npy file
        """
        np.save(path, embeddings)
        print(f"[Embedder] Saved embeddings to {path}")

    @staticmethod
    def load_embeddings(path):
        """
        Load embeddings from .npy file
        """
        print(f"[Embedder] Loading embeddings from {path}")
        return np.load(path)
