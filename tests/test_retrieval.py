from pipeline.retrieval import fetch_few_shots

def test_few_shots_retrieval():
    query = "What are the symptoms of fever?"
    results = fetch_few_shots(query, faiss_index=None, examples_df=None, embedder=None, top_k=2)
    assert isinstance(results, dict), "fetch_few_shots should return a dict"
    assert "few_shot_examples" in results, "Response should contain few_shot_examples"
