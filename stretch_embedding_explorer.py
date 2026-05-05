WORD_CATEGORIES = {
    "sports": [
        "football", "soccer", "tennis", "rugby", "cricket",
        "basketball", "baseball", "golf", "boxing", "athletics",
        "running", "cycling", "swimming", "skiing", "racing",
        "coach", "player", "team", "match", "game",
        "league", "season", "tournament", "championship", "cup",
        "goal", "score", "winner", "defeat", "victory",
        "stadium", "club", "fans", "referee", "striker",
        "defender", "midfielder", "captain", "injury", "training"
    ],

    "technology": [
        "computer", "software", "hardware", "internet", "network",
        "server", "database", "algorithm", "data", "digital",
        "mobile", "phone", "device", "screen", "keyboard",
        "processor", "chip", "memory", "system", "program",
        "website", "online", "browser", "email", "security",
        "virus", "download", "upload", "search", "engine",
        "robot", "machine", "automation", "code", "application",
        "platform", "technology", "electronic", "wireless", "media"
    ],

    "business_finance": [
        "market", "stock", "shares", "bank", "banking",
        "finance", "financial", "economy", "economic", "trade",
        "trading", "investment", "investor", "profit", "loss",
        "revenue", "sales", "price", "cost", "budget",
        "company", "corporate", "business", "industry", "firm",
        "consumer", "customer", "retail", "growth", "decline",
        "tax", "debt", "loan", "credit", "cash",
        "currency", "export", "import", "merger", "deal"
    ],

    "politics_government": [
        "government", "minister", "president", "parliament", "senate",
        "election", "campaign", "vote", "voter", "party",
        "policy", "law", "legal", "court", "justice",
        "democracy", "leader", "leadership", "opposition", "debate",
        "speech", "reform", "rights", "public", "state",
        "national", "international", "defense", "military", "war",
        "peace", "treaty", "diplomacy", "foreign", "domestic",
        "official", "authority", "power", "crisis", "scandal"
    ],

    "entertainment_media": [
        "film", "movie", "cinema", "actor", "actress",
        "director", "producer", "script", "television", "tv",
        "radio", "music", "song", "album", "singer",
        "band", "concert", "festival", "award", "show",
        "series", "drama", "comedy", "theatre", "stage",
        "performance", "artist", "celebrity", "star", "audience",
        "viewer", "channel", "broadcast", "advertising", "magazine",
        "newspaper", "journalist", "interview", "story", "culture"
    ],
}


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel

from embeddings_lab import load_glove, extract_bert_embedding


def validate_word_categories(word_categories, glove_embeddings):
    """Validate that selected words are usable for the GloVe visualization.

    Checks:
    - total selected words
    - duplicate words
    - missing words from the GloVe vocabulary
    """
    all_words = []

    for category, words in word_categories.items():
        all_words.extend(words)

    total_words = len(all_words)
    unique_words = set(all_words)

    duplicate_words = sorted([
        word for word in unique_words
        if all_words.count(word) > 1
    ])

    missing_words = []

    for category, words in word_categories.items():
        for word in words:
            if word not in glove_embeddings:
                missing_words.append((category, word))

    print("=" * 80)
    print("WORD CATEGORY VALIDATION")
    print("=" * 80)
    print(f"Total selected words: {total_words}")
    print(f"Unique selected words: {len(unique_words)}")
    print(f"Duplicate words: {len(duplicate_words)}")
    print(f"Missing words from GloVe: {len(missing_words)}")

    if duplicate_words:
        print("\nDuplicate words:")
        for word in duplicate_words:
            print(f"  - {word}")

    if missing_words:
        print("\nMissing words:")
        for category, word in missing_words:
            print(f"  - [{category}] {word}")

    if total_words != 200:
        raise ValueError(
            f"Expected exactly 200 selected words, but found {total_words}."
        )

    if duplicate_words:
        raise ValueError(
            "Duplicate words found. Each selected word should appear only once."
        )

    if missing_words:
        raise ValueError(
            "Some selected words are missing from GloVe. Replace them before plotting."
        )


def extract_word_vectors(word_categories, glove_embeddings):
    """Extract GloVe vectors for the selected categorized words.

    Returns:
        word_labels: list of selected words
        category_labels: list of semantic category labels
        word_vectors: numpy array of shape (n_words, 50)
    """
    validate_word_categories(word_categories, glove_embeddings)

    word_labels = []
    category_labels = []
    vectors = []

    for category, words in word_categories.items():
        for word in words:
            word_labels.append(word)
            category_labels.append(category)
            vectors.append(glove_embeddings[word])

    word_vectors = np.vstack(vectors)

    print("\n" + "=" * 80)
    print("WORD VECTOR EXTRACTION")
    print("=" * 80)
    print(f"Extracted word vectors shape: {word_vectors.shape}")
    print(f"Number of word labels: {len(word_labels)}")
    print(f"Number of category labels: {len(category_labels)}")

    return word_labels, category_labels, word_vectors



def reduce_embeddings_to_2d(embeddings, method="tsne", perplexity=30,
                            random_state=42):
    """Reduce high-dimensional embeddings to 2D.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        method: "tsne" or "pca"
        perplexity: t-SNE perplexity value
        random_state: random seed for reproducible output

    Returns:
        points_2d: numpy array of shape (n_samples, 2)
    """
    embeddings = np.asarray(embeddings, dtype=float)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D array, but got shape {embeddings.shape}."
        )

    n_samples = embeddings.shape[0]

    if n_samples < 2:
        raise ValueError("Need at least 2 embeddings to reduce to 2D.")

    print("\n" + "=" * 80)
    print("DIMENSIONALITY REDUCTION")
    print("=" * 80)
    print(f"Input embedding shape: {embeddings.shape}")
    print(f"Reduction method: {method}")

    if method == "tsne":
        if perplexity >= n_samples:
            raise ValueError(
                f"t-SNE perplexity must be less than number of samples. "
                f"Got perplexity={perplexity}, n_samples={n_samples}."
            )

        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto"
        )

        points_2d = reducer.fit_transform(embeddings)

    elif method == "pca":
        reducer = PCA(
            n_components=2,
            random_state=random_state
        )

        points_2d = reducer.fit_transform(embeddings)

    else:
        raise ValueError("method must be either 'tsne' or 'pca'.")

    print(f"Output 2D shape: {points_2d.shape}")

    return points_2d



def plot_word_embedding_space(points_2d, word_labels, category_labels,
                              output_path="glove_word_space.png"):
    """Plot the 2D GloVe word embedding space.

    Args:
        points_2d: numpy array of shape (n_words, 2)
        word_labels: list of word strings
        category_labels: list of category strings
        output_path: filename for the saved plot
    """
    points_2d = np.asarray(points_2d, dtype=float)

    if points_2d.shape[1] != 2:
        raise ValueError(
            f"Expected 2D points with shape (n, 2), got {points_2d.shape}."
        )

    if not (len(word_labels) == len(category_labels) == len(points_2d)):
        raise ValueError(
            "word_labels, category_labels, and points_2d must have the same length."
        )

    plt.figure(figsize=(14, 10))

    unique_categories = list(dict.fromkeys(category_labels))

    for category in unique_categories:
        indices = [
            i for i, cat in enumerate(category_labels)
            if cat == category
        ]

        x_values = [points_2d[i, 0] for i in indices]
        y_values = [points_2d[i, 1] for i in indices]

        plt.scatter(
            x_values,
            y_values,
            label=category,
            alpha=0.75,
            s=45
        )

    # Annotate at least 10 representative words.
    # Here we annotate 15 words: 3 from each category.
    annotation_words = {
        "football", "team", "goal",
        "computer", "internet", "mobile",
        "market", "bank", "profit",
        "government", "election", "minister",
        "film", "music", "television"
    }

    for i, word in enumerate(word_labels):
        if word in annotation_words:
            plt.annotate(
                word,
                (points_2d[i, 0], points_2d[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

    plt.title("GloVe Word Embedding Space (t-SNE)", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Semantic Category")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("WORD EMBEDDING PLOT")
    print("=" * 80)
    print(f"Saved word embedding plot to: {output_path}")



def select_bbc_articles(csv_path="data/bbc_news.csv",
                        articles_per_category=4,
                        random_state=42):
    """Select a balanced sample of BBC News articles for document visualization.

    The stretch assignment requires 20 articles total, with at least 3 articles
    from each of the 5 BBC News categories. Using 4 per category gives exactly
    20 articles.

    Args:
        csv_path: path to the BBC News CSV file
        articles_per_category: number of articles to sample from each category
        random_state: random seed for reproducible sampling

    Returns:
        sample_df: DataFrame with selected articles and short plot labels
    """
    df = pd.read_csv(csv_path)

    required_columns = {"text", "category"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV file is missing required columns: {missing_columns}"
        )

    category_counts = df["category"].value_counts().sort_index()

    print("\n" + "=" * 80)
    print("BBC ARTICLE SELECTION")
    print("=" * 80)
    print("Available articles by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    if (category_counts < articles_per_category).any():
        raise ValueError(
            f"Each category must contain at least {articles_per_category} articles."
        )

    sample_df = (
        df.groupby("category", group_keys=False)
          .sample(n=articles_per_category, random_state=random_state)
          .reset_index(drop=True)
    )

    # Create short labels for annotation in the document plot.
    sample_df["doc_id"] = (
        sample_df.groupby("category").cumcount() + 1
    )

    sample_df["plot_label"] = (
        sample_df["category"].str[:4] + "_" + sample_df["doc_id"].astype(str)
    )

    print("\nSelected articles by category:")
    selected_counts = sample_df["category"].value_counts().sort_index()
    for category, count in selected_counts.items():
        print(f"  {category}: {count}")

    print(f"\nSelected document sample shape: {sample_df.shape}")

    return sample_df

def extract_document_embeddings(sample_df, model_name="distilbert-base-uncased"):
    """Extract DistilBERT embeddings for selected BBC News articles.

    Args:
        sample_df: DataFrame containing at least a "text" column
        model_name: Hugging Face model name

    Returns:
        document_embeddings: numpy array of shape (n_documents, 768)
    """
    if "text" not in sample_df.columns:
        raise ValueError("sample_df must contain a 'text' column.")

    print("\n" + "=" * 80)
    print("DISTILBERT DOCUMENT EMBEDDINGS")
    print("=" * 80)
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []

    texts = sample_df["text"].tolist()
    total_texts = len(texts)

    for index, text in enumerate(texts, start=1):
        print(f"Extracting embedding {index}/{total_texts}")

        embedding = extract_bert_embedding(
            text,
            tokenizer,
            model
        )

        embeddings.append(embedding)

    document_embeddings = np.vstack(embeddings)

    print(f"\nExtracted document embeddings shape: {document_embeddings.shape}")

    if document_embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected DistilBERT embeddings with 768 dimensions, "
            f"but got shape {document_embeddings.shape}."
        )

    return document_embeddings


def plot_document_embedding_space(points_2d, sample_df,
                                  output_path="bert_document_space.png"):
    """Plot the 2D DistilBERT document embedding space.

    Args:
        points_2d: numpy array of shape (n_documents, 2)
        sample_df: DataFrame with "category" and "plot_label" columns
        output_path: filename for the saved plot
    """
    points_2d = np.asarray(points_2d, dtype=float)

    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(
            f"Expected 2D points with shape (n, 2), got {points_2d.shape}."
        )

    required_columns = {"category", "plot_label"}
    missing_columns = required_columns - set(sample_df.columns)

    if missing_columns:
        raise ValueError(
            f"sample_df is missing required columns: {missing_columns}"
        )

    if len(points_2d) != len(sample_df):
        raise ValueError(
            "points_2d and sample_df must contain the same number of documents."
        )

    plt.figure(figsize=(12, 8))

    unique_categories = list(dict.fromkeys(sample_df["category"].tolist()))

    for category in unique_categories:
        category_mask = sample_df["category"] == category
        category_indices = sample_df[category_mask].index.tolist()

        x_values = [points_2d[i, 0] for i in category_indices]
        y_values = [points_2d[i, 1] for i in category_indices]

        plt.scatter(
            x_values,
            y_values,
            label=category,
            alpha=0.8,
            s=80
        )

    # Annotate all 20 documents because the plot is small enough.
    for i, row in sample_df.iterrows():
        plt.annotate(
            row["plot_label"],
            (points_2d[i, 0], points_2d[i, 1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9
        )

    plt.title("BBC News DistilBERT Document Embedding Space (t-SNE)", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="BBC Category")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("DOCUMENT EMBEDDING PLOT")
    print("=" * 80)
    print(f"Saved document embedding plot to: {output_path}")


if __name__ == "__main__":
    glove = load_glove("data/glove_50k_50d.txt")
    print(f"Loaded {len(glove)} GloVe vectors")

    word_labels, category_labels, word_vectors = extract_word_vectors(
        WORD_CATEGORIES,
        glove
    )

    word_points_2d = reduce_embeddings_to_2d(
        word_vectors,
        method="tsne",
        perplexity=30,
        random_state=42
    )

    plot_word_embedding_space(
        word_points_2d,
        word_labels,
        category_labels,
        output_path="glove_word_space.png"
    )

    sample_df = select_bbc_articles(
        csv_path="data/bbc_news.csv",
        articles_per_category=4,
        random_state=42
    )

    document_embeddings = extract_document_embeddings(
        sample_df,
        model_name="distilbert-base-uncased"
    )

    document_points_2d = reduce_embeddings_to_2d(
    document_embeddings,
    method="tsne",
    perplexity=5,
    random_state=42
    )


    plot_document_embedding_space(
        document_points_2d,
        sample_df,
        output_path="bert_document_space.png"
    )