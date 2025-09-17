# # llm.py
# import os
# import json
# import re
# import google.generativeai as genai

# from pipeline.token_counter import count_tokens
# from pipeline.token_tracker import token_tracker

# # ----------------- GEMINI API Key -----------------
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("❌ GEMINI_API_KEY environment variable not set.")
# genai.configure(api_key=GEMINI_API_KEY)

# # ----------------- Custom Exception -----------------
# class LLMCallError(Exception):
#     """Custom exception for LLM call failures in medical chatbot."""
#     pass

# # ----------------- LLM Call -----------------
# def call_medical_llm(
#     step_name: str,
#     user_question: str,
#     retrieved_context: str = "",
#     model_name: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
#     response_format: str = "json",
# ):
#     print(f"\n🚀 [LLM] Step: {step_name}")
#     print(f"👉 Model: {model_name}")
#     print(f"👉 User Question: {user_question}")
#     if retrieved_context:
#         print(f"📖 Retrieved Context Provided: YES ({len(retrieved_context.split())} words)")
#     else:
#         print("📖 Retrieved Context Provided: NO")

#     # ----------------- Prompt Construction -----------------
#     prompt = (
#         "You are a knowledgeable medical assistant. "
#         "⚠️ Disclaimer: This is not a substitute for professional medical advice. "
#         "Always consult a licensed doctor for serious concerns.\n"
#     )
#     if retrieved_context:
#         prompt += f"Reference info:\n{retrieved_context}\n"
#     prompt += (
#         "Answer the following question concisely and accurately.\n"
#         "Output must be a valid JSON with keys 'answer' and 'source_examples'. "
#         "Do NOT add extra text or markdown.\n"
#         f"User Question: {user_question}"
#     )

#     try:
#         prompt_tokens = count_tokens(prompt, model_name)
#     except Exception:
#         prompt_tokens = len(prompt.split())
#         print("⚠️ [LLM] Token counting failed for prompt, falling back to word count.")

#     print(f"📝 Prompt built ({prompt_tokens} tokens approx)")

#     # ----------------- Call Gemini -----------------
#     model = genai.GenerativeModel(model_name)
#     try:
#         response = model.generate_content(prompt)
#         output = ""

#         if hasattr(response, "text") and response.text:
#             output = response.text.strip()
#         elif getattr(response, "candidates", None):
#             try:
#                 output = response.candidates[0].content.parts[0].text.strip()
#             except Exception:
#                 print("⚠️ [LLM] Could not extract text from candidates, defaulting empty.")
#                 output = ""
#         else:
#             print("⚠️ [LLM] No response text found, defaulting empty.")

#         print(f"✅ [LLM] Raw Output: {output[:200]}{'...' if len(output) > 200 else ''}")

#     except Exception as e:
#         msg = str(e)
#         print(f"❌ [LLM] Exception during model call: {msg}")
#         if "ResourceExhausted" in msg or "quota" in msg.lower() or "token" in msg.lower():
#             raise LLMCallError(f"[{step_name}] Token exhaustion: {msg}")
#         raise LLMCallError(f"[{step_name}] LLM call failed: {msg}")

#     # ----------------- Token Counting (Output) -----------------
#     try:
#         completion_tokens = count_tokens(output, model_name)
#     except Exception:
#         completion_tokens = len(output.split())
#         print("⚠️ [LLM] Token counting failed for output, falling back to word count.")

#     usage = {
#         "prompt_tokens": prompt_tokens,
#         "completion_tokens": completion_tokens,
#         "total_tokens": prompt_tokens + completion_tokens,
#         "step": step_name,
#         "model": model_name,
#     }

#     print(f"📊 Token Usage: {usage}")

#     # ----------------- Parse JSON -----------------
#     if response_format == "json":
#         cleaned = re.sub(r"```(json|text)?", "", output, flags=re.IGNORECASE)
#         cleaned = cleaned.replace("```", "").strip()

#         try:
#             parsed = json.loads(cleaned)
#             print("✅ [LLM] Successfully parsed JSON response")
#             return parsed, usage
#         except json.JSONDecodeError:
#             print("❌ [LLM] JSON parsing failed. Returning fallback response.")
#             return {
#                 "answer": f"Failed to parse LLM response: {cleaned}",
#                 "source_examples": []
#             }, usage

#     return output, usage



# llm.py
import os
import json
import re
import google.generativeai as genai

from pipeline.token_counter import count_tokens
from pipeline.token_tracker import token_tracker

# ----------------- GEMINI API Keys (Rotation) -----------------
api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
api_keys = [k.strip() for k in api_keys if k.strip()]

if not api_keys:
    raise RuntimeError("❌ GEMINI_API_KEYS environment variable not set or empty.")

_current_key_index = 0

def get_next_api_key():
    """Return the next API key in round-robin rotation."""
    global _current_key_index
    key = api_keys[_current_key_index]
    _current_key_index = (_current_key_index + 1) % len(api_keys)
    return key

# ----------------- Custom Exception -----------------
class LLMCallError(Exception):
    """Custom exception for LLM call failures in medical chatbot."""
    pass

# ----------------- LLM Call -----------------
def call_medical_llm(
    step_name: str,
    user_question: str,
    retrieved_context: str = "",
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    response_format: str = "json",
):
    print(f"\n🚀 [LLM] Step: {step_name}")
    print(f"👉 Model: {model_name}")
    print(f"👉 User Question: {user_question}")
    if retrieved_context:
        print(f"📖 Retrieved Context Provided: YES ({len(retrieved_context.split())} words)")
    else:
        print("📖 Retrieved Context Provided: NO")

    # ----------------- Prompt Construction -----------------
    prompt = (
        "You are a knowledgeable medical assistant. "
        "⚠️ Disclaimer: This is not a substitute for professional medical advice. "
        "Always consult a licensed doctor for serious concerns.\n"
    )
    if retrieved_context:
        prompt += f"Reference info:\n{retrieved_context}\n"
    prompt += (
        "Answer the following question concisely and accurately.\n"
        "Output must be a valid JSON with keys 'answer' and 'source_examples'. "
        "Do NOT add extra text or markdown.\n"
        f"User Question: {user_question}"
    )

    try:
        prompt_tokens = count_tokens(prompt, model_name)
    except Exception:
        prompt_tokens = len(prompt.split())
        print("⚠️ [LLM] Token counting failed for prompt, falling back to word count.")

    print(f"📝 Prompt built ({prompt_tokens} tokens approx)")

    # ----------------- Call Gemini with Round-Robin Keys -----------------
    response, output = None, ""
    last_error = None

    for attempt in range(len(api_keys)):
        key_to_use = get_next_api_key()
        print(f"🔑 Using Gemini key {attempt+1}/{len(api_keys)}")
        genai.configure(api_key=key_to_use)
        model = genai.GenerativeModel(model_name)
        try:
            response = model.generate_content(prompt)
            if hasattr(response, "text") and response.text:
                output = response.text.strip()
            elif getattr(response, "candidates", None):
                try:
                    output = response.candidates[0].content.parts[0].text.strip()
                except Exception:
                    print("⚠️ [LLM] Could not extract text from candidates, defaulting empty.")
                    output = ""
            else:
                print("⚠️ [LLM] No response text found, defaulting empty.")

            print(f"✅ [LLM] Raw Output: {output[:200]}{'...' if len(output) > 200 else ''}")
            break  # success → exit loop
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ [LLM] Key failed ({key_to_use}): {last_error}")
            continue

    if not output:
        raise LLMCallError(f"[{step_name}] All keys failed. Last error: {last_error}")

    # ----------------- Token Counting (Output) -----------------
    try:
        completion_tokens = count_tokens(output, model_name)
    except Exception:
        completion_tokens = len(output.split())
        print("⚠️ [LLM] Token counting failed for output, falling back to word count.")

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "step": step_name,
        "model": model_name,
    }

    print(f"📊 Token Usage: {usage}")

    # ----------------- Parse JSON -----------------
    if response_format == "json":
        cleaned = re.sub(r"```(json|text)?", "", output, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        try:
            parsed = json.loads(cleaned)
            print("✅ [LLM] Successfully parsed JSON response")
            return parsed, usage
        except json.JSONDecodeError:
            print("❌ [LLM] JSON parsing failed. Returning fallback response.")
            return {
                "answer": f"Failed to parse LLM response: {cleaned}",
                "source_examples": []
            }, usage

    return output, usage
