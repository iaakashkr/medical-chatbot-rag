# llm_medical.py
import os
import json
import re
import google.generativeai as genai

from pipeline.token_counter import count_tokens
from pipeline.token_tracker import token_tracker

# ----------------- GEMINI API Key -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

# ----------------- Custom Exception -----------------
class LLMCallError(Exception):
    """Custom exception for LLM call failures in medical chatbot."""
    pass

# ----------------- LLM Call -----------------
def call_medical_llm(
    step_name: str,
    user_question: str,
    retrieved_context: str = "",
    model_name: str = "gemini-1.5-flash",
    response_format: str = "json",
):
    prompt = "You are a knowledgeable medical assistant.\n"
    if retrieved_context:
        prompt += f"Reference info:\n{retrieved_context}\n"
    prompt += (
        "Answer the following question concisely and accurately.\n"
        "Output must be a valid JSON with keys 'answer' and 'source_examples'."
        "Do NOT add extra text or markdown.\n"
        f"User Question: {user_question}"
    )

    prompt_tokens = count_tokens(prompt, model_name)

    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        output = response.text.strip() if response.text else ""
    except Exception as e:
        msg = str(e)
        if "ResourceExhausted" in msg or "quota" in msg.lower() or "token" in msg.lower():
            raise LLMCallError(f"[{step_name}] Token exhaustion: {msg}")
        raise LLMCallError(f"[{step_name}] LLM call failed: {msg}")

    completion_tokens = count_tokens(output, model_name)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "step": step_name,
        "model": model_name,
    }

    if response_format == "json":
        cleaned = re.sub(r"```(json|text)?", "", output, flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned), usage
        except json.JSONDecodeError:
            return {"answer": f"Failed to parse LLM response: {cleaned}", "source_examples": []}, usage

    return output, usage
