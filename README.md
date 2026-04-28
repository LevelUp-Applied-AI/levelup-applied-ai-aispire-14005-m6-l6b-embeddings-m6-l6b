# Lab 6B — Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and DistilBERT — on a corpus of climate articles.

## Objectives

- Build TF-IDF representations and compute pairwise document similarity
- Load pre-trained GloVe vectors and compute average text embeddings
- Extract contextual embeddings from DistilBERT using the Hugging Face transformers library
- Compare similarity rankings across all three methods

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

DistilBERT uses PyTorch. `torch` is listed in `requirements.txt` (version 2.4 or higher) and `pip install` will download the correct version for your computer automatically.

**Note:** The first time you import DistilBERT, the model files (~250 MB) will be downloaded and cached (saved) on your computer. This may take a few minutes on slower internet connections. After the first time, the model loads from the cache on your computer.

## Data

- `data/climate_articles.csv` — Climate article corpus
- `data/glove_5k_50d.txt` — Pre-extracted GloVe vectors (5000 words, 50 dimensions)

## Tasks

Complete all six functions in `embeddings_lab.py`:

1. **`build_tfidf(texts)`** — Build TF-IDF representations using `TfidfVectorizer`
2. **`compute_tfidf_similarity(tfidf_matrix)`** — Compute pairwise cosine similarity matrix
3. **`load_glove(filepath)`** — Load GloVe vectors into a dictionary
4. **`text_to_glove(text, embeddings)`** — Compute average GloVe embedding for a text
5. **`extract_bert_embedding(text, tokenizer, model)`** — Extract a DistilBERT sentence embedding
6. **`compare_similarities(texts, queries, ...)`** — Compare top-3 similar texts across all methods

## Submission

1. Create a branch named `lab-6b-embeddings`
2. Complete all functions in `embeddings_lab.py`
3. Run `pytest tests/ -v` to verify your work
4. Open a PR to `main` — the autograder will run automatically
5. Your PR description must include:
   - Comparison table for 5 queries: top-3 similar texts per method
   - Analysis: where methods agree/disagree
   - Paste your PR URL into TalentLMS → Module 6 Week B → Lab 6B to submit this assignment

Resubmissions are accepted through Saturday of the assignment week.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
