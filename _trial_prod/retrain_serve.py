# retrain_serve.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import numpy as np
import json

RAW_QUESTIONS = "data/raw_questions.csv"
FEEDBACK_PATH = "results/feedback/labels.csv"
OUT_DIR = "results/finetuned_model"
REPORT_PATH = "results/finetuned_report.json"

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
TRAIN_VAL_SPLIT = 0.9


def load_feedback():
    if not os.path.exists(FEEDBACK_PATH):
        raise FileNotFoundError(
            f"Feedback file not found: {FEEDBACK_PATH}. Submit annotations first."
        )

    fb = pd.read_csv(FEEDBACK_PATH, dtype=str)
    fb = fb[fb['label'].isin(['duplicate', 'near-duplicate/harmonizable', 'different'])]

    fb['y'] = fb['label'].apply(
        lambda x: 1 if x in ['duplicate', 'near-duplicate/harmonizable'] else 0
    )
    return fb


def prepare_examples():
    fb = load_feedback()
    raw = pd.read_csv(RAW_QUESTIONS, dtype=str)

    id_to_text = {
        row['question_id']: row['question_text']
        for _, row in raw.iterrows()
    }

    examples = []
    for _, r in fb.iterrows():
        q1 = id_to_text.get(r['id1'])
        q2 = id_to_text.get(r['id2'])
        if q1 and q2:
            examples.append(InputExample(texts=[q1, q2], label=float(r['y'])))

    if len(examples) < 20:
        raise ValueError(
            f"Too few labeled examples ({len(examples)}). Need ≥ 20 for reliable fine‑tuning."
        )

    return examples


def retrain_model():
    os.makedirs(OUT_DIR, exist_ok=True)

    examples = prepare_examples()
    split = int(len(examples) * TRAIN_VAL_SPLIT)

    train = examples[:split]
    val = examples[split:]

    print(f"> Training samples: {len(train)}")
    print(f"> Validation samples: {len(val)}")

    model = SentenceTransformer(BASE_MODEL)

    train_dataloader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    # Evaluator
    val_labels = [e.label for e in val]
    val_texts1 = [e.texts[0] for e in val]
    val_texts2 = [e.texts[1] for e in val]

    evaluator = BinaryClassificationEvaluator(
        val_texts1, val_texts2, val_labels, name="val-eval"
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=100,
        warmup_steps=50,
        optimizer_params={'lr': LR},
        output_path=OUT_DIR,
    )

    # Evaluation
    model = SentenceTransformer(OUT_DIR)
    sim_scores = model.similarity(val_texts1, val_texts2)
    sim_scores = sim_scores.cpu().numpy().tolist()

    # pick best threshold
    thresholds = np.linspace(0.5, 0.9, 21)
    best_thr = 0.5
    best_acc = 0

    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in sim_scores]
        acc = (np.array(preds) == np.array(val_labels)).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    report = {
        "num_training_samples": len(train),
        "num_validation_samples": len(val),
        "best_threshold": float(best_thr),
        "best_accuracy": float(best_acc),
        "eval_scores": sim_scores,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Fine‑tuning complete. Report saved to {REPORT_PATH}")
    return report


# 🔥 Function siap digunakan Streamlit
def load_model_for_inference():
    if os.path.exists(OUT_DIR):
        print(">> Using finetuned model")
        return SentenceTransformer(OUT_DIR)
    else:
        print(">> Using base model")
        return SentenceTransformer(BASE_MODEL)


if __name__ == "__main__":
    retrain_model()
