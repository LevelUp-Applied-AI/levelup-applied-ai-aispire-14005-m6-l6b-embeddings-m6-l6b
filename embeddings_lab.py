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
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    
    if not vectors:
        return np.zeros(50)
    
    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Task 3: Mean Pooling
    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]
    attention_mask = inputs['attention_mask']      # [batch_size, seq_len]
    
    # Mask padding tokens before averaging
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    embedding = sum_embeddings / sum_mask
    return embedding.squeeze().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself.
    """
    comparison = {}
    
    # Pre-compute corpus embeddings to speed up the comparison
    glove_corpus = np.array([text_to_glove(t, glove_embeddings) for t in texts])
    
    # Note: BERT corpus embedding takes time; usually done once in __main__ 
    # but implemented here to follow function signature requirements.
    bert_corpus = np.array([extract_bert_embedding(t, bert_tokenizer, bert_model) for t in texts])

    for q_text in queries:
        try:
            q_idx = texts.index(q_text)
        except ValueError:
            continue
            
        comparison[q_text] = {"tfidf": [], "glove": [], "bert": []}

        # 1. TF-IDF (Matrix is pre-computed)
        tfidf_scores = tfidf_sim[q_idx]
        
        # 2. GloVe Similarity
        q_glove = text_to_glove(q_text, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(q_glove, glove_corpus)[0]

        # 3. BERT Similarity
        q_bert = extract_bert_embedding(q_text, bert_tokenizer, bert_model).reshape(1, -1)
        bert_scores = sklearn_cosine(q_bert, bert_corpus)[0]

        # Process top results for each method
        methods_data = [("tfidf", tfidf_scores), ("glove", glove_scores), ("bert", bert_scores)]
        
        for method_name, scores in methods_data:
            # Get top 4 indices (to ensure we have 3 after excluding self)
            top_indices = np.argsort(scores)[::-1][:4]
            results = []
            for idx in top_indices:
                if texts[idx] != q_text:
                    results.append((texts[idx], float(scores[idx])))
            comparison[q_text][method_name] = results[:3]

    return comparison


if __name__ == "__main__":
    import torch
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

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    # Task 4: Compare
    if result and glove:
        # Pick first text from each unique category
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        
        print("\nComparing similarities (this may take a minute)...")
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        
        if comparison:
            for q in comparison:
                print(f"\nQUERY (Category: {df[df['text'] == q]['category'].values[0]}):")
                print(f"Text: {q[:100]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method.upper()}:")
                    for res_text, score in top:
                        print(f"    - [{score:.4f}] {res_text[:60]}...")