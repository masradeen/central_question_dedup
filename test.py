import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
FILE_PATH = os.path.join(BASE_DIR, "results\perturbed_embeddings.npy")

emb = np.load(FILE_PATH)
print(emb.shape)
