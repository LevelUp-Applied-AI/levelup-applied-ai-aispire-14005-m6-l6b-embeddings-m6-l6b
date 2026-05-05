"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    similarity_matrix = sklearn_cosine(tfidf_matrix)

    return similarity_matrix


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            word = parts[0]
            vector = np.array(parts[1:], dtype=float)

            embeddings[word] = vector

    return embeddings


def validate_word_categories(word_categories, glove):
    missing_words = []

    for category, words in word_categories.items():
        for word in words:
            if word not in glove:
                missing_words.append((category, word))

    print(f"Total selected words: {sum(len(words) for words in word_categories.values())}")
    print(f"Missing words from GloVe: {len(missing_words)}")

    if missing_words:
        print("\nMissing words:")
        for category, word in missing_words:
            print(f"  {category}: {word}")


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()
    vectors = []

    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])

    if len(vectors) == 0:
        return np.zeros(50)

    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    import torch

    model.eval()

    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask

    summed_embeddings = masked_embeddings.sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)

    mean_pooled = summed_embeddings / token_counts

    return mean_pooled.squeeze(0).cpu().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself and exact duplicate texts. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """
    results = {}

    # Build one GloVe document embedding for every text.
    # Shape: (number_of_texts, 50)
    glove_doc_embeddings = np.vstack([
        text_to_glove(text, glove_embeddings)
        for text in texts
    ])

    # Build one BERT document embedding for every text.
    # Shape: (number_of_texts, 768)
    bert_doc_embeddings = np.vstack([
        extract_bert_embedding(text, bert_tokenizer, bert_model)
        for text in texts
    ])

    def normalize_for_duplicate_check(text):
        """Normalize whitespace so exact duplicate checks are more reliable."""
        return " ".join(text.split())

    def get_top_3(scores, query_index, query_text):
        """Return top-3 (text, score) pairs, excluding the query and duplicates."""
        scores = np.asarray(scores, dtype=float).copy()

        normalized_query = normalize_for_duplicate_check(query_text)

        # Exclude:
        # 1. the query row itself
        # 2. any other article with the exact same text after whitespace normalization
        for idx, text in enumerate(texts):
            normalized_text = normalize_for_duplicate_check(text)

            if idx == query_index or normalized_text == normalized_query:
                scores[idx] = -np.inf

        sorted_indices = np.argsort(scores)[::-1]

        top_indices = [
            idx for idx in sorted_indices
            if np.isfinite(scores[idx])
        ][:3]

        return [
            (texts[idx], float(scores[idx]))
            for idx in top_indices
        ]

    for query in queries:
        if query not in texts:
            raise ValueError("Each query must be one of the texts in the corpus.")

        query_index = texts.index(query)

        # 1. TF-IDF similarity:
        # Already computed as a full pairwise matrix in Task 1.
        tfidf_scores = tfidf_sim[query_index]

        # 2. GloVe similarity:
        # Convert the query to a GloVe average vector, then compare to all documents.
        glove_query_embedding = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(
            glove_query_embedding,
            glove_doc_embeddings
        )[0]

        # 3. BERT similarity:
        # The query is already one of the documents, so we can reuse its embedding.
        bert_query_embedding = bert_doc_embeddings[query_index].reshape(1, -1)
        bert_scores = sklearn_cosine(
            bert_query_embedding,
            bert_doc_embeddings
        )[0]

        results[query] = {
            "tfidf": get_top_3(tfidf_scores, query_index, query),
            "glove": get_top_3(glove_scores, query_index, query),
            "bert": get_top_3(bert_scores, query_index, query),
        }

    return results


def compute_oov_stats(texts, embeddings):
    """Compute OOV statistics for the corpus against the GloVe vocabulary.

    Returns:
        oov_rate: percentage of tokens not found in GloVe
        total_tokens: total number of whitespace tokens
        oov_count: number of out-of-vocabulary tokens
        top_oov: list of the most frequent OOV tokens
    """
    token_counts = {}
    total_tokens = 0
    oov_count = 0

    for text in texts:
        words = text.lower().split()

        for word in words:
            total_tokens += 1

            if word not in embeddings:
                oov_count += 1
                token_counts[word] = token_counts.get(word, 0) + 1

    oov_rate = oov_count / total_tokens if total_tokens > 0 else 0.0

    top_oov = sorted(
        token_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:20]

    return oov_rate, total_tokens, oov_count, top_oov


def get_text_category_lookup(df):
    """Create a mapping from article text to its category."""
    return dict(zip(df["text"], df["category"]))


def summarize_comparison_by_category(df, comparison):
    """Print category-level agreement and exact-article overlap for Task 5."""
    text_to_category = get_text_category_lookup(df)

    print("\n" + "=" * 80)
    print("TASK 5 ANALYSIS: CATEGORY AGREEMENT")
    print("=" * 80)

    for query, method_results in comparison.items():
        query_category = text_to_category.get(query, "UNKNOWN")

        print(f"\nQuery category: {query_category}")
        print(f"Query preview: {query[:120]}...")

        method_categories = {}

        for method in ["tfidf", "glove", "bert"]:
            top_results = method_results.get(method, [])

            categories = [
                text_to_category.get(text, "UNKNOWN")
                for text, score in top_results
            ]

            method_categories[method] = categories

            print(f"\n  {method.upper()}:")
            for rank, (text, score) in enumerate(top_results, start=1):
                category = text_to_category.get(text, "UNKNOWN")
                print(
                    f"    {rank}. [{category}] score={score:.4f} "
                    f"{text[:100]}..."
                )

        all_same_as_query = all(
            category == query_category
            for method in method_categories
            for category in method_categories[method]
        )

        tfidf_texts = {text for text, _ in method_results.get("tfidf", [])}
        glove_texts = {text for text, _ in method_results.get("glove", [])}
        bert_texts = {text for text, _ in method_results.get("bert", [])}

        exact_overlap = tfidf_texts & glove_texts & bert_texts

        print("\n  Agreement summary:")
        print(f"    All top-3 results in query category? {all_same_as_query}")
        print(f"    Exact articles shared by all methods: {len(exact_overlap)}")

        if exact_overlap:
            for text in exact_overlap:
                print(f"      - {text[:100]}...")


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate.
    if result and glove and tfidf_sim is not None:
        queries = [
            df[df["category"] == cat]["text"].iloc[0]
            for cat in df["category"].unique()
        ]

        comparison = compare_similarities(
            texts,
            queries,
            tfidf_sim,
            glove,
            model,
            tokenizer
        )

        # Task 5: Analysis output
        if comparison:
            summarize_comparison_by_category(df, comparison)

            oov_rate, total_tokens, oov_count, top_oov = compute_oov_stats(
                texts,
                glove
            )

            print("\n" + "=" * 80)
            print("TASK 5 ANALYSIS: GLOVE OOV RATE")
            print("=" * 80)
            print(f"Total tokens: {total_tokens}")
            print(f"OOV tokens: {oov_count}")
            print(f"OOV rate: {oov_rate:.2%}")

            print("\nTop 20 OOV tokens:")
            for token, count in top_oov:
                print(f"  {token}: {count}")