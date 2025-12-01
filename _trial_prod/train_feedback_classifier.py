# train_feedback_classifier.py
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Paths (ubah jika perlu)
FEEDBACK_PATH = "results/feedback/labels.csv"
EMB_PATH = "results/embeddings.npy"
RAW_QUESTIONS = "data/raw_questions.csv"
OUT_MODEL = "results/feedback_classifier.joblib"
REPORT_PATH = "results/feedback_train_report.txt"

# Load feedback
if not os.path.exists(FEEDBACK_PATH):
    raise FileNotFoundError(f"Feedback file not found: {FEEDBACK_PATH}. Please upload or run HITL to generate.")

fb = pd.read_csv(FEEDBACK_PATH, dtype=str)
# Filter useful labels
fb = fb[fb['label'].isin(['duplicate','near-duplicate/harmonizable','different'])]
if fb.empty:
    raise ValueError("No usable labeled rows in feedback file.")

# Map labels -> binary
fb['y'] = fb['label'].apply(lambda x: 1 if x in ['duplicate','near-duplicate/harmonizable'] else 0)

# Load embeddings and raw questions to map ids -> index
if not os.path.exists(EMB_PATH):
    raise FileNotFoundError(f"Embeddings not found: {EMB_PATH}. Please run dedup pipeline first to generate results/embeddings.npy")

emb = np.load(EMB_PATH)  # shape (N, D)
raw = pd.read_csv(RAW_QUESTIONS, dtype=str)

# build id -> index
id_to_idx = {row['question_id']: idx for idx, row in raw.reset_index().iterrows()}

# build dataset X,y
rows = []
labels = []
for _, r in fb.iterrows():
    id1, id2 = r['id1'], r['id2']
    if id1 not in id_to_idx or id2 not in id_to_idx:
        # skip if mapping missing
        continue
    e1 = emb[id_to_idx[id1]]
    e2 = emb[id_to_idx[id2]]
    # feature engineering: absolute diff + elementwise product (concatenated)
    feat = np.concatenate([np.abs(e1 - e2), e1 * e2])
    rows.append(feat)
    labels.append(int(r['y']))

X = np.vstack(rows)
y = np.array(labels)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# eval
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# save model and report
joblib.dump(clf, OUT_MODEL)
with open(REPORT_PATH, "w") as f:
    f.write("Classification report:\n")
    f.write(report + "\n")
    f.write(f"ROC AUC: {auc:.4f}\n")

print("Training finished.")
print(report)
print("ROC AUC:", auc)
print("Model saved to", OUT_MODEL)
print("Report saved to", REPORT_PATH)