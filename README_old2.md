# Central Question Deduplication System

## Overview

This repository implements a semantic deduplication pipeline for large-scale survey question repositories, developed in the context of national statistical systems at Statistics Indonesia (BPS). The system aims to identify semantically redundant or highly similar survey questions across different survey instruments and organizational units, supporting metadata harmonization and reducing redundancy in official statistics.

Large statistical organizations often face duplication of survey questions due to independent survey design processes across directorates. This project addresses that issue using embedding-based semantic similarity methods combined with scalable retrieval and clustering techniques.

## Problem Motivation

Managing survey question repositories at scale involves several challenges:

- Large volumes of textual questions across heterogeneous surveys
- Semantic overlap expressed through different wording or structures
- Instability of similarity thresholds in high-dimensional embedding spaces
- Noise in sentence embeddings affecting similarity graph construction

These challenges motivate the use of robust semantic representations and efficient similarity search algorithms.

## Methodology

The deduplication pipeline consists of the following stages:

1. Text preprocessing and normalization of survey questions
2. Sentence embedding generation using sentence-transformer models
3. Approximate nearest neighbor search for efficient similarity retrieval
4. Similarity graph construction based on cosine similarity thresholds
5. Graph-based clustering to identify redundant or highly similar questions

## Repository Structure

central_question_dedup/
├── data/ # Input survey question data
├── notebooks/ # Exploratory analysis and experiments
├── results/ # Embeddings, similarity outputs, and visualizations
├── src/ # Core pipeline modules
├── main.py # Pipeline entry point
├── requirements.txt # Python dependencies
└── README.md

## Installation

Ensure Python 3.8 or later is installed, then install dependencies:
s
pip install -r requirements.txt

## Usage

Run the full deduplication pipeline:
python main.py

Depending on configuration, the pipeline performs embedding generation, similarity search, and clustering sequentially.

## Input Format

The expected input is a CSV file containing survey questions and metadata:

question_id,question_text,survey_name,directorate

Example:

Q001,"Apa penghasilan utama rumah tangga Anda?",Susenas,Statistik Sosial
Q245,"Berapa pendapatan utama keluarga Anda?",Sakernas,Statistik Ketenagakerjaan

## Output

The pipeline produces:

- Pairwise similarity results between survey questions
- Clusters of semantically similar or redundant questions
- Optional visualizations such as similarity heatmaps

## Research Contribution

This project contributes to applied and methodological research in:

- Semantic similarity and deduplication for official statistics
- Scalable similarity graph construction for large text collections
- Empirical analysis of embedding noise and threshold sensitivity
- Metadata harmonization across distributed survey systems
  Observed instability due to noisy embeddings and threshold selection motivates further investigation into optimization-based and theoretically grounded approaches for large-scale semantic deduplication.

## Limitations and Future Work

Planned extensions include:

- Improved robustness to threshold sensitivity
- Support for multilingual survey question repositories
- Hierarchical and structured question similarity modeling
- Integration with distributed or federated similarity computation frameworks

## Author

Sigit Nugroho Putra
Statistics Indonesia (BPS)
2025

## Citation

If you use or reference this repository, please cite:
Putra, S. N. (2025). Central Question Deduplication System.
GitHub repository: https://github.com/masradeen/central_question_dedup
