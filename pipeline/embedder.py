# embedder.py

import os
import numpy as np
import pandas as pd
import faiss
import pickle
import time
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---- Logger Setup ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---- Flexible Embedder using Hugging Face ----
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize Hugging Face embedding model
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, query: str) -> np.ndarray:
        """
        Return normalized embedding vector as numpy array (compatible with FAISS)
        """
        vec = self.model.encode(query, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

# ---- Embedding wrapper ----
def _embed(query, embedder: Embedder):
    return embedder.embed(query)

def embedding_creation(df_summ, embedding_column_name: str, output_name: str, embedder: Embedder):
    embeddings_list_chunks = []
    for index, row in df_summ.iterrows():
        Column_chunk = row[embedding_column_name]
        try:
            embeddings_chunk = _embed(Column_chunk, embedder)
            embeddings_list_chunks.append(embeddings_chunk)
        except Exception as e:
            log.error(f"Error {e} at row {index}, retrying in 10s...")
            time.sleep(10)
            embeddings_chunk = _embed(Column_chunk, embedder)
            embeddings_list_chunks.append(embeddings_chunk)

        log.info(f"✅ {index} embedding created")

    # Convert to FAISS index
    dimension = embeddings_list_chunks[0].shape[0]
    array_chunk = np.vstack(embeddings_list_chunks).astype(np.float32)
    faiss.normalize_L2(array_chunk)
    index_1 = faiss.IndexFlatIP(dimension)
    index_1.add(array_chunk)

    output = f"{output_name}.faiss"
    dir_path = os.path.dirname(output)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    faiss.write_index(index_1, output)
    log.info(f"✅ Saved FAISS index at {output}")

# ---- Sparse BM25 ----
def create_sparse_model(documents: list, bm25_model_name: str):
    tokenized_docs = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    dir_path = os.path.dirname(bm25_model_name)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(bm25_model_name, "wb") as f:
        pickle.dump(bm25, f)

    log.info(f"✅ Saved BM25 model at {bm25_model_name}")

# ---- Run pipeline ----
if __name__ == "__main__":
    csv_path = "resources/train.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ {csv_path} not found. Place it in the resources folder.")

    df_few_shots = pd.read_csv(csv_path)
    log.info(f"Columns: {df_few_shots.columns.tolist()}")

    embedder = Embedder(model_name="all-MiniLM-L6-v2")  # Hugging Face embeddings

    embedding_creation(df_few_shots, "Question", r"resources/embeddings/med_embeddings", embedder)
    create_sparse_model(df_few_shots["Question"].to_list(), r"resources/pickles/syntactic_model_med.pkl")
