from pipeline.embedder import Embedder

def test_embedding_dimension():
    embedder = Embedder()
    text = "What is diabetes?"
    vector = embedder.embed(text)
    assert len(vector) > 0, "Embedding vector should not be empty"
    assert isinstance(vector, list) or hasattr(vector, "shape"), "Embedding should be a vector-like object"
