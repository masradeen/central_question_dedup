# BPS Question Deduplication System

### Semantic Similarity & Cross‑Survey Redundancy Detection

### Statistics Indonesia (BPS)

---

## 📌 Overview

BPS menjalankan **puluhan survei nasional** setiap tahun melalui berbagai direktorat.  
Masalah yang muncul:

- Banyak **pertanyaan survei berbeda namun menanyakan hal yang sama**.
- Redundansi antar direktorat → beban responden tinggi.
- Ketidakharmonisan metadata → sulit integrasi data lintas survei.

Repository ini menyediakan **sistem otomatis** untuk:

1. Menemukan pertanyaan yang duplikat antar survei.
2. Menghitung kemiripan semantik antar pertanyaan.
3. Mengelompokkan pertanyaan-pertanyaan mirip dalam _clusters_.
4. Menyediakan output siap evaluasi untuk unit statistik tematik di BPS.

Sistem ini dirancang menggunakan pendekatan **state‑of‑the‑art NLP**  
(`sentence-transformers`, kNN, cosine similarity, connected-components clustering).

---

## 🚀 Features

### ✅ Semantic Embedding

Menggunakan Sentence Transformers (MiniLM, multilingual models).

### ✅ kNN Candidate Retrieval

Mengambil kandidat tetangga terdekat tanpa menghitung semua kombinasi.

### ✅ Cosine Similarity

Skor kemiripan 0–1.

### ✅ Duplicate Pair Detection

Default threshold:

```
similarity ≥ 0.78
```

### ✅ Graph-Based Clustering

Mengelompokkan pertanyaan yang mirip ke dalam cluster.

### ✅ Visual Analytics

- similarity matrix heatmap
- CSV hasil pasangan mirip
- JSON cluster hasil grouping

---

## 📂 Repository Structure

```
central_question_dedup/
│
├── data/
│   └── raw_questions.csv
│
├── results/
│   ├── embeddings.npy
│   ├── similarity_pairs.csv
│   ├── heatmap.png
│   └── clusters.json
│
├── src/
│   ├── embedder.py
│   ├── dedup_engine.py
│   └── clustering.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```
pip install -r requirements.txt
```

---

## 🏃 How to Run

### 🔥 Jalankan pipeline lengkap

```
python main.py --mode all
```

### 🧩 Jalankan dedup-only

```
python main.py --mode dedup
```

### 🧮 Jalankan cluster-only

```
python main.py --mode cluster
```

---

## 📄 Input Format (raw_questions.csv)

| question_id | question_text | survey_name | directorate |
| ----------- | ------------- | ----------- | ----------- |

Contoh:

```
Q001,"Apa penghasilan utama rumah tangga Anda?",Susenas,Direktorat Statistik Sosial
Q502,"Berapa pendapatan utama keluarga Anda?",Sakernas,Direktorat Tenaga Kerja
```

---

## 📤 Outputs

### 1️⃣ similarity_pairs.csv

Pasangan pertanyaan mirip (≥ threshold).

### 2️⃣ heatmap.png

Visualisasi similarity matrix.

### 3️⃣ clusters.json

Contoh:

```json
{
  "clusters": [
    ["Q001", "Q502", "Q722"],
    ["Q018", "Q019"]
  ]
}
```

---

## 🧠 Why This Matters for BPS

Sistem ini membantu:

- harmonisasi metadata antar direktorat
- mengurangi duplikasi pertanyaan antar survei
- menurunkan _respondent burden_
- meningkatkan _statistical coherence_
- rekomendasi penggabungan survei

Dapat dikembangkan menjadi:

- survey harmonization engine
- metadata knowledge graph
- inter-survey alignment recommender

---

## 🏛️ Research Contribution

Repository ini dapat digunakan untuk riset:

- Semantic Matching for Large‑Scale National Surveys
- Optimization for Cross‑Survey Metadata Harmonization
- Automatic Redundancy Detection in Official Statistics

Cocok sebagai material aplikasi **MS/PhD KAUST**.

---

## 🔧 Potential Extensions

- Multilingual models
- Hierarchical clustering
- Integrasi ke Metadata Warehouse
- Thematic grouping per direktorat

---

## 🙌 Credits

Developed by:  
**Sigit Nugroho Putra**  
Statistics Indonesia (BPS) — ICT & Statistical Computing  
2025
