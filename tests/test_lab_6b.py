"""Autograder tests for Lab 6B — Embeddings Comparison."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings_lab import (
    build_tfidf, compute_tfidf_similarity, load_glove,
    text_to_glove, extract_bert_embedding, compare_similarities,
)


GLOVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "glove_5k_50d.txt")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_articles.csv")

# Sample texts for tests that don't need the full dataset
SAMPLE_TEXTS = [
    "Climate change affects global temperatures and weather patterns.",
    "Renewable energy sources like solar and wind reduce carbon emissions.",
    "Deforestation leads to loss of biodiversity and increased greenhouse gases.",
    "Rising sea levels threaten coastal communities worldwide.",
    "International climate agreements aim to limit global warming.",
]


# ── TF-IDF ───────────────────────────────────────────────────────────────

def test_tfidf_matrix():
    """build_tfidf should return a matrix with one row per text."""
    result = build_tfidf(SAMPLE_TEXTS)
    assert result is not None, "build_tfidf returned None"
    tfidf_matrix, vectorizer = result
    assert tfidf_matrix.shape[0] == len(SAMPLE_TEXTS), (
        f"Expected {len(SAMPLE_TEXTS)} rows, got {tfidf_matrix.shape[0]}"
    )
    assert tfidf_matrix.shape[1] > 0, "Vocabulary size should be > 0"
    # All values should be non-negative
    assert (tfidf_matrix.toarray() >= 0).all(), "TF-IDF values must be non-negative"


def test_tfidf_similarity_matrix():
    """Pairwise similarity should be a square matrix with diagonal ~ 1.0."""
    result = build_tfidf(SAMPLE_TEXTS)
    assert result is not None, "build_tfidf returned None"
    tfidf_matrix, _ = result
    sim = compute_tfidf_similarity(tfidf_matrix)
    assert sim is not None, "compute_tfidf_similarity returned None"
    n = len(SAMPLE_TEXTS)
    assert sim.shape == (n, n), f"Expected shape ({n}, {n}), got {sim.shape}"
    # Diagonal should be approximately 1.0
    for i in range(n):
        assert abs(sim[i, i] - 1.0) < 0.01, (
            f"Diagonal element [{i},{i}] should be ~1.0, got {sim[i, i]:.4f}"
        )
    # All values should be in [0, 1] for TF-IDF cosine
    assert (sim >= -0.01).all() and (sim <= 1.01).all(), "Similarity values should be in [0, 1]"


# ── GloVe ────────────────────────────────────────────────────────────────

def test_glove_loaded():
    """load_glove should return ~5000 words with 50-d vectors."""
    glove = load_glove(GLOVE_PATH)
    assert glove is not None, "load_glove returned None"
    assert isinstance(glove, dict), "load_glove must return a dict"
    assert len(glove) >= 4000, f"Expected ~5000 words, got {len(glove)}"
    sample_vec = next(iter(glove.values()))
    assert isinstance(sample_vec, np.ndarray), "Values must be numpy arrays"
    assert sample_vec.shape == (50,), f"Expected shape (50,), got {sample_vec.shape}"


def test_text_to_glove():
    """text_to_glove should return a 50-d vector for normal text."""
    glove = load_glove(GLOVE_PATH)
    assert glove is not None
    embedding = text_to_glove("climate change affects global temperatures", glove)
    assert embedding is not None, "text_to_glove returned None"
    assert isinstance(embedding, np.ndarray), "Must return a numpy array"
    assert embedding.shape == (50,), f"Expected shape (50,), got {embedding.shape}"
    assert not np.allclose(embedding, 0), "Embedding should not be all zeros for common words"


def test_text_to_glove_oov():
    """text_to_glove should handle texts with out-of-vocabulary words."""
    glove = load_glove(GLOVE_PATH)
    assert glove is not None
    # Mix of likely in-vocab and OOV words
    embedding = text_to_glove("the xyzzyplugh climate", glove)
    assert embedding is not None, "text_to_glove returned None for text with OOV words"
    assert isinstance(embedding, np.ndarray), "Must return a numpy array"
    assert embedding.shape == (50,), f"Expected shape (50,), got {embedding.shape}"


def test_text_to_glove_all_oov():
    """text_to_glove should return a zero vector when every word is OOV."""
    glove = load_glove(GLOVE_PATH)
    assert glove is not None
    embedding = text_to_glove("xyzzyplugh qwertyuiopasdf zxcvbnmlkjhg", glove)
    assert embedding is not None, "text_to_glove returned None for all-OOV text"
    assert isinstance(embedding, np.ndarray), "Must return a numpy array"
    assert embedding.shape == (50,), f"Expected shape (50,), got {embedding.shape}"
    assert np.allclose(embedding, 0), (
        "All-OOV text must return a zero vector of shape (50,); "
        f"got non-zero values (mean={embedding.mean():.4f})"
    )


# ── BERT ─────────────────────────────────────────────────────────────────

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_bert_embedding():
    """extract_bert_embedding should return a 768-d mean-pooled vector."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    text = "Climate change is a global challenge."
    embedding = extract_bert_embedding(text, tokenizer, model)
    assert embedding is not None, "extract_bert_embedding returned None"
    assert isinstance(embedding, np.ndarray), "Must return a numpy array"
    assert embedding.shape == (768,), f"Expected shape (768,), got {embedding.shape}"
    assert not np.allclose(embedding, 0), "BERT embedding should not be all zeros"
    # Spec (docstring): mean-pool last_hidden_state across the token dimension
    # with attention-mask handling. Reject CLS-token output (last_hidden_state[:,0,:]).
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**enc)
    last_hidden = outputs.last_hidden_state  # (1, seq, 768)
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (1, seq, 1)
    expected_mean = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    expected_mean = expected_mean.squeeze(0).numpy()
    cls_only = last_hidden[:, 0, :].squeeze(0).numpy()
    assert np.allclose(embedding, expected_mean, atol=1e-4), (
        "Embedding must be the attention-mask-weighted mean of last_hidden_state "
        "across the token dimension (spec: mean-pool with attention-mask handling). "
        "Returning CLS-token output (last_hidden_state[:,0,:]) is not mean pooling."
    )
    assert not np.allclose(embedding, cls_only, atol=1e-4), (
        "Embedding equals the CLS-token vector — spec requires mean pooling, "
        "not CLS-token extraction."
    )


# ── Comparison ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed")
def test_compare_similarities():
    """compare_similarities should return top-3 results per method per query."""
    # Build representations
    result = build_tfidf(SAMPLE_TEXTS)
    assert result is not None
    tfidf_matrix, _ = result
    tfidf_sim = compute_tfidf_similarity(tfidf_matrix)

    glove = load_glove(GLOVE_PATH)
    assert glove is not None

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    queries = SAMPLE_TEXTS[:2]
    comparison = compare_similarities(
        SAMPLE_TEXTS, queries, tfidf_sim, glove, model, tokenizer
    )
    assert comparison is not None, "compare_similarities returned None"
    assert isinstance(comparison, dict), "Must return a dict"
    assert len(comparison) == len(queries), (
        f"Expected exactly {len(queries)} query keys, got {len(comparison)}"
    )
    for q in queries:
        assert q in comparison, f"Missing query key: {q[:50]}..."
        for method in ["tfidf", "glove", "bert"]:
            assert method in comparison[q], f"Missing method '{method}' for query"
            results = comparison[q][method]
            assert isinstance(results, list), f"{method} results must be a list"
            assert len(results) == 3, (
                f"Expected exactly 3 results for {method}, got {len(results)} "
                "(spec: top-3 most similar texts excluding the query itself)"
            )
            # Spec: results must exclude the query itself
            result_texts = [t for t, _ in results]
            assert q not in result_texts, (
                f"{method} results for query must exclude the query itself; "
                f"query was returned in its own top-3"
            )
            # Spec: "top-3 most similar" — results must be sorted by score descending
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True), (
                f"{method} results must be sorted by similarity descending "
                f"(spec: top-3 most similar); got scores {scores}"
            )
