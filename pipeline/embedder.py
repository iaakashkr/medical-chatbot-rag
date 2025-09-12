# embedder.py

import os
import numpy as np
import pandas as pd
import faiss
import pickle
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---- Flexible Embedder using Hugging Face ----
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize Hugging Face embedding model
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, query: str):
        """
        Return normalized embedding vector as list (compatible with FAISS)
        """
        vec = self.model.encode(query, normalize_embeddings=True)
        return vec.tolist()


# ---- Embedding wrapper ----
def _embed(query, embedder):
    return embedder.embed(query)


def embedding_creation(df_summ, embedding_column_name: str, output_name: str, embedder):
    embeddings_list_chunks = []
    for index, row in df_summ.iterrows():
        Column_chunk = row[embedding_column_name]
        try:
            embeddings_chunk = _embed(Column_chunk, embedder)
            embeddings_list_chunks.append(embeddings_chunk)
        except Exception as e:
            print(f'error {e} at {index}, retrying in 10s...')
            time.sleep(10)
            embeddings_chunk = _embed(Column_chunk, embedder)
            embeddings_list_chunks.append(embeddings_chunk)

        print(f'{index} embedding created')

    # Convert to FAISS index
    dimension = len(embeddings_list_chunks[0])
    array_chunk = np.asarray(embeddings_list_chunks).astype(np.float32)
    faiss.normalize_L2(array_chunk)
    index_1 = faiss.IndexFlatIP(dimension)
    index_1.add(array_chunk)

    output = f'{output_name}.faiss'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    faiss.write_index(index_1, output)
    print(f"✅ Saved FAISS index at {output}")


# ---- Sparse BM25 ----
def create_sparse_model(documents: list, bm25_model_name: str):
    tokenized_docs = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    os.makedirs(os.path.dirname(bm25_model_name), exist_ok=True)
    with open(bm25_model_name, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"✅ Saved BM25 model at {bm25_model_name}")


# ---- Run pipeline ----
if __name__ == "__main__":
    df_few_shots = pd.read_csv("train.csv")
    print("Columns:", df_few_shots.columns)

    embedder = Embedder(model_name="all-MiniLM-L6-v2")  # Hugging Face embeddings

    embedding_creation(df_few_shots, "Question", r"resources/embeddings/med_embeddings", embedder)
    create_sparse_model(df_few_shots["Question"].to_list(), r"resources/pickles/syntactic_model_med.pkl")
