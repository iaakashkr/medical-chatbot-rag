import tiktoken
from google.generativeai import GenerativeModel

def count_tokens(text: str, model) -> int:
    """
    Count tokens for both OpenAI (GPT) and Google (Gemini) models.
    Falls back to estimate for non-OpenAI models.
    """
    if isinstance(model, GenerativeModel) or not isinstance(model, str):
        return len(text) // 4

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    
    return len(enc.encode(text))
