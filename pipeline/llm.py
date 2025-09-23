# # llm.py
# import os
# import json
# import re
# import google.generativeai as genai

# from pipeline.token_counter import count_tokens
# from pipeline.token_tracker import token_tracker

# # ----------------- GEMINI API Keys (Rotation) -----------------
# api_keys = []

# for k, v in os.environ.items():
#     if k.startswith("GEMINI_API_KEY_") and v.strip():
#         api_keys.append((k, v.strip()))

# # Sort by key name so order is predictable: _1, _2, _3...
# api_keys = [v for k, v in sorted(api_keys, key=lambda x: x[0])]

# if not api_keys:
#     raise RuntimeError("❌ No Gemini API keys found (expected GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...).")

# _current_key_index = 0

# def get_next_api_key():
#     """Return the next API key in round-robin rotation."""
#     global _current_key_index
#     key = api_keys[_current_key_index]
#     _current_key_index = (_current_key_index + 1) % len(api_keys)
#     return key

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
#     "You are a highly knowledgeable medical assistant. "
#     "⚠️ Only answer questions related to human medicine, symptoms, diagnosis, treatments, and health conditions. "
#     "Do NOT answer any questions unrelated to medicine, including coding, math, programming, or general knowledge. "
#     "Answer the user question directly and concisely with relevant medical information. "
#     "Do not add any disclaimers, warnings, or extra text. Only give the medical answer. "
#     "If the question is not medical, respond with: 'I can only answer medical questions.' "
# )

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

#     # ----------------- Call Gemini with Round-Robin Keys -----------------
#     response, output = None, ""
#     last_error = None

#     for attempt in range(len(api_keys)):
#         key_to_use = get_next_api_key()
#         print(f"🔑 Using Gemini key {attempt+1}/{len(api_keys)}")
#         genai.configure(api_key=key_to_use)
#         model = genai.GenerativeModel(model_name)
#         try:
#             response = model.generate_content(prompt)
#             if hasattr(response, "text") and response.text:
#                 output = response.text.strip()
#             elif getattr(response, "candidates", None):
#                 try:
#                     output = response.candidates[0].content.parts[0].text.strip()
#                 except Exception:
#                     print("⚠️ [LLM] Could not extract text from candidates, defaulting empty.")
#                     output = ""
#             else:
#                 print("⚠️ [LLM] No response text found, defaulting empty.")

#             print(f"✅ [LLM] Raw Output: {output[:200]}{'...' if len(output) > 200 else ''}")
#             break  # success → exit loop
#         except Exception as e:
#             last_error = str(e)
#             print(f"⚠️ [LLM] Key failed ({key_to_use}): {last_error}")
#             continue

#     if not output:
#         raise LLMCallError(f"[{step_name}] All keys failed. Last error: {last_error}")

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
import logging
import asyncio

from pipeline.token_counter import count_tokens
from pipeline.token_tracker import token_tracker

# ---- Logger ----
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------- GEMINI API Keys (Rotation) -----------------
api_keys = sorted(
    [v for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY_") and v.strip()]
)
if not api_keys:
    raise RuntimeError("❌ No Gemini API keys found (expected GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...).")

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

# ----------------- Async LLM Call -----------------
async def call_medical_llm(
    step_name: str,
    user_question: str,
    retrieved_context: str = "",
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    response_format: str = "json",
):
    log.info(f"🚀 [LLM] Step: {step_name}")
    log.info(f"👉 Model: {model_name}")
    log.info(f"👉 User Question: {user_question}")
    if retrieved_context:
        log.info(f"📖 Retrieved Context Provided: YES ({len(retrieved_context.split())} words)")
    else:
        log.info("📖 Retrieved Context Provided: NO")

    prompt = f"""
        You are a highly knowledgeable medical assistant.
        ⚠️ Only answer questions related to human medicine, symptoms, diagnosis, treatments, and health conditions.
        Do NOT answer any questions unrelated to medicine, including coding, math, programming, or general knowledge.
        Answer the user question directly and concisely with relevant medical information.
        Do not add any disclaimers, warnings, or extra text. Only give the medical answer.
        If the question is not medical, respond with: 'I can only answer medical questions.'
        """ + (f"\nReference info:\n{retrieved_context}" if retrieved_context else "") + f"""

        Answer the following question concisely and accurately in JSON with keys 'answer' and 'source_examples':
        User Question: {user_question}
        """


    try:
        prompt_tokens = count_tokens(prompt, model_name)
    except Exception:
        prompt_tokens = len(prompt.split())
        log.warning("⚠️ Token counting failed for prompt, falling back to word count.")

    log.info(f"📝 Prompt built ({prompt_tokens} tokens approx)")

    output = ""
    last_error = None

    for attempt in range(len(api_keys)):
        key_to_use = get_next_api_key()
        log.info(f"🔑 Using Gemini key {attempt+1}/{len(api_keys)}")
        genai.configure(api_key=key_to_use)
        model = genai.GenerativeModel(model_name)

        try:
            # Run blocking Gemini call in a separate thread
            response = await asyncio.to_thread(model.generate_content, prompt)

            if hasattr(response, "text") and response.text:
                output = response.text.strip()
            elif getattr(response, "candidates", None):
                try:
                    output = response.candidates[0].content.parts[0].text.strip()
                except Exception:
                    log.warning("⚠️ Could not extract text from candidates, defaulting empty.")
                    output = ""
            else:
                log.warning("⚠️ No response text found, defaulting empty.")
            log.info(f"✅ Raw Output: {output[:200]}{'...' if len(output) > 200 else ''}")
            break
        except Exception as e:
            last_error = str(e)
            log.warning(f"⚠️ Key failed ({key_to_use}): {last_error}")
            continue

    if not output:
        raise LLMCallError(f"[{step_name}] All keys failed. Last error: {last_error}")

    try:
        completion_tokens = count_tokens(output, model_name)
    except Exception:
        completion_tokens = len(output.split())
        log.warning("⚠️ Token counting failed for output, falling back to word count.")

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "step": step_name,
        "model": model_name,
    }
    log.info(f"📊 Token Usage: {usage}")

    # ----------------- Parse JSON Robustly -----------------
    if response_format == "json":
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                log.info("✅ Successfully parsed JSON response")
            except json.JSONDecodeError:
                log.error("❌ JSON parsing failed despite regex match")
                parsed = {"answer": f"Failed to parse LLM response: {output}", "source_examples": []}
        else:
            log.error("❌ No JSON found in LLM output")
            parsed = {"answer": f"Failed to parse LLM response: {output}", "source_examples": []}
        return parsed, usage

    return output, usage
