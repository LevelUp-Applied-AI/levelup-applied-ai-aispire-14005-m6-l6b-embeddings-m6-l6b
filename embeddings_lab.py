"""
Module 6 Week B — Lab: Embeddings Comparison"""

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    print (f"Shape:{tfidf_matrix.shape}")
    print(f"Feature names:{vectorizer.get_feature_names_out()}")

    return tfidf_matrix,vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix."""
    similarity_matrix = sklearn_cosine(tfidf_matrix)
    print (similarity_matrix)
    return sklearn_cosine(tfidf_matrix)


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file."""
    embeddings = {}
    with open (filepath,"r",encoding ="utf-8") as f :
        for line in f :
            parts = line.strip().split()
            word =parts[0]
            vector = np.array(parts[1:],dtype=np.float32)
            embeddings[word] = vector

    return embeddings

glove = load_glove("data/glove_50k_50d.txt")  
print(f"Loaded {len(glove)} words")
print (f"Vector shape:{glove['climate'].shape}")      



def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text."""
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    if not vectors:
        return np.zeros(50)
    
    return np.mean(vectors ,axis=0)

def mean_pool(hidden_states,attention_mask):
    """Mean pool hidden states, accounting for padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked_hidden = hidden_states*mask
    summed =masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1)
    return summed/counts




def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT."""
    inputs = tokenizer(text,return_tensors ="pt", truncation =True,max_length=512)
    with torch.no_grad():
        outputs=model(**inputs)
        hidden_states = outputs.last_hidden_state
        embedding = mean_pool(hidden_states,inputs["attention_mask"])
        return embedding.squeeze().numpy()
    

def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT."""
    results = {}
    glove_text_embeddings = np.array([text_to_glove(t,glove_embeddings)for t in texts])
    bert_text_embeddings = np.array([extract_bert_embedding(t,bert_tokenizer,bert_model) for t in texts])
    for query in queries:
        query_idx=texts.index(query)

        tfidf_scores = tfidf_sim[query_idx]

        tfidf_top3=sorted(
            [(texts[i],tfidf_scores[i]) for i in range (len(texts)) if i !=query_idx],
            key=lambda x:x[1],
            reverse = True)[:3]

        query_glove = text_to_glove(query,glove_embeddings).reshape(1,-1)
        glove_scores = sklearn_cosine(query_glove,glove_text_embeddings)[0]
        glove_top3=sorted(
            [(texts[i],glove_scores[i]) for i in range (len(texts)) if i !=query_idx],
            key=lambda x:x[1],
            reverse = True)[:3]

        query_bert =extract_bert_embedding(query,bert_tokenizer,bert_model).reshape(1,-1)
        bert_scores = sklearn_cosine(query_bert,bert_text_embeddings)[0]
        
        bert_top3=sorted(
            [(texts[i],bert_scores[i]) for i in range (len(texts)) if i !=query_idx],
            key=lambda x:x[1],
            reverse = True)[:3]        
        
        results[query] = {
            "tfidf": tfidf_top3,
            "glove": glove_top3,
            "bert": bert_top3
        }
    
    return results



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
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys())[:2]:
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")
