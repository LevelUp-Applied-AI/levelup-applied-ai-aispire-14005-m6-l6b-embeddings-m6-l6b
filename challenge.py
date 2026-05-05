import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer,AutoModel
from collections import Counter


def l2_normalize(vectors):
    return normalize(vectors, norm="l2")

def compare_metrics(query,embedding):
    query = query.reshape(1, -1) 

    cos_sim =cosine_similarity(query,embedding)[0]
    euc_dist = euclidean_distances(query,embedding)[0]
    euc_sim =-euc_dist

    cos_rank = np.argsort(cos_sim)[::-1]
    dist_rank = np.argsort(euc_dist)
    return cos_sim,euc_dist,cos_rank,dist_rank

tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased") 
model =AutoModel.from_pretrained("distilbert-base-uncased") 



def subword_analysis(text):
    split_counts=[]
    subword_counter = Counter()

    for text in texts:
        words = text.split()
        bert_tokens = tokenizer.tokenize(text)

        if len(words)>0:
           split_counts.append(len(bert_tokens)/len(words))
           subword_counter.update(bert_tokens)

        avg_split =np.mean(split_counts)
        top_subwords = subword_counter.most_common(10)

        return avg_split,top_subwords
    

def precision_at_k(ranked_indices,relevant_indices,k):
    top_k =ranked_indices[:k]
    hits = len(set(top_k)&set(relevant_indices))
    return hits/k
    
def mean_reciprocal_rank(ranked_indices,relevant_indices):
    for i ,idx in enumerate(ranked_indices):
       if idx in relevant_indices:
          return 1/(i+1)
    return 0


def evaluate_retrieval(embeddings,queries_idx,relevance_dict):
    results ={}

    for q_idx in queries_idx:
        query_vec =embeddings[q_idx].reshape(1,-1)
        sims =cosine_similarity(query_vec,embeddings)[0]
        ranked = np.argsort(sims)[::-1]
        relevant =relevance_dict.get(q_idx,[])

        p3 = precision_at_k(ranked,relevant,3)
        p5 = precision_at_k(ranked,relevant,5)
        mrr =mean_reciprocal_rank(ranked,relevant)

        results[q_idx]={
            "p@3":p3,
            "p@5":p5,
            "MRR":mrr
        }

    return results

if __name__ == "__main__":
    texts = [
        "climate change affects oceans",
        "global warming impacts biodiversity",
        "stock market rises today",
        "football team wins match",
        "new technology in AI"
    ]

    embeddings = np.random.rand(len(texts), 50)

    norm_emb = l2_normalize(embeddings)

    cos_sim, dist, cos_rank, dist_rank = compare_metrics(norm_emb[0], norm_emb)

    print("Cosine top-3:", cos_rank[:3])
    print("Euclidean top-3:", dist_rank[:3])

    avg_split, top_sub = subword_analysis(texts)

    print("Avg subword split:", avg_split)
    print("Top subwords:", top_sub)

    relevance = {
        0: [1],  
        2: [3]
    }

    results = evaluate_retrieval(norm_emb, [0, 2], relevance)

    print("Retrieval results:", results)