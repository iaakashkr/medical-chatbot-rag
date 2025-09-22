from pipeline.llm import call_medical_llm

def test_llm_response_format():
    query = "What is hypertension?"
    response, usage = call_medical_llm("unit-test", query)
    assert isinstance(response, dict), "LLM response should be a dictionary"
    assert "answer" in response, "Response must contain 'answer'"
    assert "source_examples" in response, "Response must contain 'source_examples'"
