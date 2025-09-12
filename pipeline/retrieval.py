# fewshot_module.py
import faiss
import numpy as np
import pandas as pd
import re
from embedder import Embedder  # OpenAI or HuggingFace Embedder
from rank_bm25 import BM25Okapi
import logging

# ----------------- Logger Setup -----------------
def get_logger(silent=False):
    if silent:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(__name__)

log = get_logger(silent=False)

# ----------------- Text Normalization -----------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------- Embedding -----------------
def embed_text(query: str, embedder: Embedder):
    """
    Return a normalized FAISS-ready embedding vector for a single query.
    """
    embedding = embedder.embed(query)  # returns list
    vec = np.array(embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

# ----------------- Exact Match Bonus -----------------
def exact_match_bonus(user_question: str, example_question: str):
    uq = set(user_question.lower().split())
    eq = set(example_question.lower().split())
    return len(uq & eq) / (len(uq) + 1e-8)

# ----------------- Hybrid Similarity Search -----------------
def hybrid_similarity_search(
    query: str,
    examples_df: pd.DataFrame,
    faiss_index,
    embedder: Embedder,
    bm25_model: BM25Okapi = None,
    tokenized_corpus: list = None,
    semantic_threshold: float = 0.2,
    syntactic_threshold: float = 0.5,
):
    query_clean = normalize(query)

    # --- Semantic FAISS ---
    vec = embed_text(query_clean, embedder)

    if vec.shape[1] != faiss_index.d:
        raise ValueError(f"FAISS index dimension ({faiss_index.d}) does not match embedding ({vec.shape[1]})")

    distances, indices = faiss_index.search(vec, len(examples_df))
    examples_df['semantic_score'] = np.array(distances[0])

    # --- Optional Syntactic BM25 ---
    if bm25_model and tokenized_corpus:
        bm25_scores = bm25_model.get_scores(query_clean.split())
        examples_df['syntactic_score'] = bm25_scores
    else:
        examples_df['syntactic_score'] = 0

    # --- Filter by thresholds ---
    semantic_selection = examples_df[examples_df['semantic_score'] >= semantic_threshold]
    syntactic_selection = examples_df[examples_df['syntactic_score'] > syntactic_threshold]
    filtered_df = pd.concat([semantic_selection, syntactic_selection]).drop_duplicates().reset_index(drop=True)

    return filtered_df

# ----------------- Fetch Few-Shot Examples -----------------
def fetch_few_shots(
    user_question: str,
    faiss_index,
    examples_df: pd.DataFrame,
    embedder: Embedder,
    bm25_model: BM25Okapi = None,
    tokenized_corpus: list = None,
    top_k: int = 2
):
    candidates_df = hybrid_similarity_search(
        query=user_question,
        examples_df=examples_df.copy(),
        faiss_index=faiss_index,
        embedder=embedder,
        bm25_model=bm25_model,
        tokenized_corpus=tokenized_corpus
    )

    similarity_flag = not candidates_df.empty
    few_shots = {}
    matched_indices = []

    if similarity_flag:
        if bm25_model and tokenized_corpus:
            max_bm25 = candidates_df['syntactic_score'].max()
            candidates_df['syntactic_score_norm'] = candidates_df['syntactic_score'] / (max_bm25 + 1e-8)
        else:
            candidates_df['syntactic_score_norm'] = 0

        # FIXED: use correct column name "Question"
        candidates_df['exact_bonus'] = candidates_df['Question'].apply(lambda x: exact_match_bonus(user_question, x))
        candidates_df['combined_score'] = (
            0.4 * candidates_df['semantic_score'] +
            0.4 * candidates_df['syntactic_score_norm'] +
            0.2 * candidates_df['exact_bonus']
        )

        # Logging top candidates
        log.info("\n--- Candidate Examples and Scores ---\n%s",
                 candidates_df[['Question','semantic_score','syntactic_score','combined_score']].sort_values('combined_score', ascending=False))

        # Select top-k few-shot examples
        selected = candidates_df.sort_values("combined_score", ascending=False).head(top_k)
        for i, row in enumerate(selected.itertuples()):
            few_shots[f"Example Question {i+1}"] = row.Question
            few_shots[f"Answer {i+1}"] = row.Answer  # retrieve corresponding answer
            matched_indices.append(row.Index)

    return {
        "similarity_flag": similarity_flag,
        "few_shot_examples": few_shots,
        "matched_indices": matched_indices
    }

# ----------------- Initialize FAISS and BM25 -----------------
def init_index_and_bm25(examples_df: pd.DataFrame, embedder: Embedder):
    """
    Create FAISS index and BM25 model from few-shot examples
    """
    embeddings = np.array([embedder.embed(normalize(q)) for q in examples_df["Question"]], dtype="float32")
    faiss.normalize_L2(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    tokenized_corpus = [q.split(" ") for q in examples_df["Question"]]
    bm25_model = BM25Okapi(tokenized_corpus)

    return faiss_index, tokenized_corpus, bm25_model
