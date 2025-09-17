# # app.py
# import gradio as gr
# import os
# import logging
# import pandas as pd
# import pickle
# import faiss
# import uuid
# import requests
# from datetime import datetime

# from pipeline.embedder import Embedder
# from pipeline.retrieval import fetch_few_shots
# from pipeline.llm import call_medical_llm, LLMCallError
# from app.dto import QueryDTO

# # ----------------- Logging -----------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# # ----------------- Download Large Files -----------------
# def download_file(url, local_path):
#     os.makedirs(os.path.dirname(local_path), exist_ok=True)
#     if not os.path.exists(local_path):
#         logger.info(f"Downloading {url} -> {local_path}")
#         r = requests.get(url, stream=True)
#         with open(local_path, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#     else:
#         logger.info(f"File already exists: {local_path}")

# # ----------------- Load Resources -----------------
# def init_resources(
#     faiss_url="https://raw.githubusercontent.com/iaakashkr/medical-chatbot-rag/main/resources/embeddings/med_embeddings.faiss",
#     train_url="https://raw.githubusercontent.com/iaakashkr/medical-chatbot-rag/main/resources/train.csv",
#     bm25_file="https://raw.githubusercontent.com/iaakashkr/medical-chatbot-rag/main/resources/pickles/syntactic_model_med.pkl"
# ):
#     # Download large files at runtime
#     faiss_file = "resources/embeddings/med_embeddings.faiss"
#     examples_file = "resources/train.csv"
#     download_file(faiss_url, faiss_file)
#     download_file(train_url, examples_file)

#     # Load CSV examples
#     examples_df = pd.read_csv(examples_file)
#     logger.info(f"Loaded {len(examples_df)} examples from {examples_file}")

#     # Embedder
#     embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     dimension = embedder.embed("test").shape[0]

#     # Load FAISS
#     if os.path.exists(faiss_file):
#         try:
#             faiss_index = faiss.read_index(faiss_file)
#             logger.info(f"Loaded FAISS index from {faiss_file}")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load FAISS index: {e}. Creating empty index.")
#             faiss_index = faiss.IndexFlatIP(dimension)
#     else:
#         logger.warning(f"⚠️ FAISS file not found at {faiss_file}, creating empty index")
#         faiss_index = faiss.IndexFlatIP(dimension)

#     # Load BM25
#     if os.path.exists(bm25_file):
#         try:
#             with open(bm25_file, "rb") as f:
#                 bm25_model = pickle.load(f)
#             tokenized_corpus = [q.split() for q in examples_df["Question"]]
#             logger.info(f"Loaded BM25 model from {bm25_file}")
#         except Exception as e:
#             logger.warning(f"⚠️ Failed to load BM25 model: {e}")
#             bm25_model, tokenized_corpus = None, None
#     else:
#         logger.warning(f"⚠️ BM25 pickle not found at {bm25_file}")
#         bm25_model, tokenized_corpus = None, None

#     return examples_df, faiss_index, bm25_model, tokenized_corpus, embedder

# examples_df, faiss_index, bm25_model, tokenized_corpus, embedder = init_resources()

# # ----------------- Stateful Chat Histories -----------------
# chat_histories = {}
# MAX_HISTORY = 4

# # ----------------- Chat Function -----------------
# def chat_fn(user_question, session_id=None):
#     dto = QueryDTO(user_question=user_question)

#     # Session ID handling
#     if not session_id:
#         session_id = str(uuid.uuid4())
#         logger.info(f"Generated new session ID: {session_id}")

#     if session_id not in chat_histories:
#         chat_histories[session_id] = []

#     # Few-shot retrieval
#     fewshot_result = {}
#     if faiss_index is not None or bm25_model is not None:
#         fewshot_result = fetch_few_shots(
#             user_question=dto.user_question,
#             faiss_index=faiss_index,
#             examples_df=examples_df.copy(),
#             embedder=embedder,
#             bm25_model=bm25_model,
#             tokenized_corpus=tokenized_corpus,
#             top_k=2
#         )
#         logger.info(f"Returned {len(fewshot_result['few_shot_examples']) // 2} few-shot examples")
#     else:
#         logger.warning("⚠️ Both FAISS and BM25 are missing. Skipping few-shot retrieval.")
#         fewshot_result = {"few_shot_examples": {}, "matched_indices": []}

#     dto.few_shot_examples = fewshot_result["few_shot_examples"]
#     dto.matched_indices = fewshot_result["matched_indices"]

#     # Full context
#     rag_items = list(dto.few_shot_examples.items())[-5:]
#     rag_context = "\n".join([f"{k}: {v}" for k, v in rag_items])
#     rag_context = " ".join(rag_context.split()[:1000])

#     recent_history = chat_histories[session_id][-MAX_HISTORY:]
#     history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in recent_history])

#     full_context = rag_context
#     if history_str:
#         full_context += "\nPrevious conversation:\n" + history_str

#     # LLM call
#     try:
#         model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
#         response_json, usage = call_medical_llm(
#             step_name="gradio_chat",
#             user_question=dto.user_question,
#             retrieved_context=full_context,
#             model_name=model_name,
#             response_format="json"
#         )
#         dto.answer = response_json.get("answer", "N/A")
#         dto.source_examples = response_json.get("source_examples", [])
#         dto.usage = usage
#         logger.info(f"LLM returned answer ({len(dto.answer.split())} words approx)")
#         logger.debug(f"Full Response JSON: {response_json}")  # internal debug
#     except LLMCallError as e:
#         logger.error(f"❌ LLM error: {str(e)}")
#         dto.answer = f"LLM Error: {str(e)}"
#         dto.source_examples = []
#         dto.usage = {}

#     # Update chat history
#     timestamp = str(datetime.now())
#     chat_histories[session_id].append({"role": "user", "content": user_question, "timestamp": timestamp})
#     chat_histories[session_id].append({"role": "assistant", "content": dto.answer, "timestamp": timestamp})

#     # Return only clean answer for UI
#     return dto.answer

# # ----------------- Gradio Interface -----------------
# with gr.Blocks() as demo:
#     gr.Markdown("## 🩺 Medical FAQ Chatbot (RAG + LLM)")
#     session_id_input = gr.Textbox(label="Session ID (optional)", placeholder="Leave empty for auto UUID")
#     user_question_input = gr.Textbox(label="Enter Your Question :", placeholder="Type a medical question here...")
#     output_box = gr.Textbox(label="Answer")  # Show only answer to user

#     submit_btn = gr.Button("Ask")
#     submit_btn.click(chat_fn, inputs=[user_question_input, session_id_input], outputs=output_box)

# # Launch
# demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


# app.py
import gradio as gr
import os
import logging
import pandas as pd
import pickle
import faiss
import uuid
import requests
from datetime import datetime

from pipeline.embedder import Embedder
from pipeline.retrieval import fetch_few_shots
from pipeline.llm import call_medical_llm, LLMCallError
from app.dto import QueryDTO

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------- Download Large Files -----------------
def download_file(url, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        logger.info(f"Downloading {url} -> {local_path}")
        r = requests.get(url, stream=True)
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        logger.info(f"File already exists: {local_path}")

# ----------------- Load Resources -----------------
def init_resources(
    faiss_url="https://raw.githubusercontent.com/iaakashkr/medical-chatbot-rag/main/resources/embeddings/med_embeddings.faiss",
    train_url="https://raw.githubusercontent.com/iaakashkr/medical-chatbot-rag/main/resources/train.csv",
    bm25_url="https://raw.githubusercontent.com/iaakashkr/medical-chatbot-rag/main/resources/pickles/syntactic_model_med.pkl"
):
    # Local file paths
    faiss_file = "resources/embeddings/med_embeddings.faiss"
    examples_file = "resources/train.csv"
    bm25_file = "resources/pickles/syntactic_model_med.pkl"

    # Download large files
    download_file(faiss_url, faiss_file)
    download_file(train_url, examples_file)
    download_file(bm25_url, bm25_file)

    # Load CSV examples
    examples_df = pd.read_csv(examples_file)
    logger.info(f"Loaded {len(examples_df)} examples from {examples_file}")

    # Embedder
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dimension = embedder.embed("test").shape[0]

    # Load FAISS
    if os.path.exists(faiss_file):
        try:
            faiss_index = faiss.read_index(faiss_file)
            logger.info(f"Loaded FAISS index from {faiss_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load FAISS index: {e}. Creating empty index.")
            faiss_index = faiss.IndexFlatIP(dimension)
    else:
        logger.warning(f"⚠️ FAISS file not found at {faiss_file}, creating empty index")
        faiss_index = faiss.IndexFlatIP(dimension)

    # Load BM25
    if os.path.exists(bm25_file):
        try:
            with open(bm25_file, "rb") as f:
                bm25_model = pickle.load(f)
            tokenized_corpus = [q.split() for q in examples_df["Question"]]
            logger.info(f"Loaded BM25 model from {bm25_file}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load BM25 model: {e}")
            bm25_model, tokenized_corpus = None, None
    else:
        logger.warning(f"⚠️ BM25 pickle not found at {bm25_file}")
        bm25_model, tokenized_corpus = None, None

    return examples_df, faiss_index, bm25_model, tokenized_corpus, embedder

examples_df, faiss_index, bm25_model, tokenized_corpus, embedder = init_resources()

# ----------------- Stateful Chat Histories -----------------
chat_histories = {}
MAX_HISTORY = 4

# ----------------- Chat Function -----------------
def chat_fn(user_question, session_id=None):
    dto = QueryDTO(user_question=user_question)

    # Session ID handling
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id}")

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    # Few-shot retrieval
    fewshot_result = {}
    if faiss_index is not None or bm25_model is not None:
        fewshot_result = fetch_few_shots(
            user_question=dto.user_question,
            faiss_index=faiss_index,
            examples_df=examples_df.copy(),
            embedder=embedder,
            bm25_model=bm25_model,
            tokenized_corpus=tokenized_corpus,
            top_k=2
        )
        logger.info(f"Returned {len(fewshot_result['few_shot_examples']) // 2} few-shot examples")
    else:
        logger.warning("⚠️ Both FAISS and BM25 are missing. Skipping few-shot retrieval.")
        fewshot_result = {"few_shot_examples": {}, "matched_indices": []}

    dto.few_shot_examples = fewshot_result["few_shot_examples"]
    dto.matched_indices = fewshot_result["matched_indices"]

    # Full context
    rag_items = list(dto.few_shot_examples.items())[-5:]
    rag_context = "\n".join([f"{k}: {v}" for k, v in rag_items])
    rag_context = " ".join(rag_context.split()[:1000])

    recent_history = chat_histories[session_id][-MAX_HISTORY:]
    history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in recent_history])

    full_context = rag_context
    if history_str:
        full_context += "\nPrevious conversation:\n" + history_str

    # LLM call
    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        response_json, usage = call_medical_llm(
            step_name="gradio_chat",
            user_question=dto.user_question,
            retrieved_context=full_context,
            model_name=model_name,
            response_format="json"
        )
        dto.answer = response_json.get("answer", "N/A")
        dto.source_examples = response_json.get("source_examples", [])
        dto.usage = usage
        logger.info(f"LLM returned answer ({len(dto.answer.split())} words approx)")
        logger.debug(f"Full Response JSON: {response_json}")  # internal debug
    except LLMCallError as e:
        logger.error(f"❌ LLM error: {str(e)}")
        dto.answer = f"LLM Error: {str(e)}"
        dto.source_examples = []
        dto.usage = {}

    # Update chat history
    timestamp = str(datetime.now())
    chat_histories[session_id].append({"role": "user", "content": user_question, "timestamp": timestamp})
    chat_histories[session_id].append({"role": "assistant", "content": dto.answer, "timestamp": timestamp})

    # Return only clean answer for UI
    return dto.answer

# ----------------- Gradio Interface -----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Medical FAQ Chatbot (RAG + LLM)")

    # Question first
    user_question_input = gr.Textbox(
        label="Enter Your Question :",
        placeholder="Type a medical question here..."
    )

    # Session ID second
    session_id_input = gr.Textbox(
        label="Session ID (optional)",
        placeholder="Leave empty for auto UUID"
    )

    # Answer last, auto-expand
    output_box = gr.Textbox(
        label="Answer",
        lines=5,
        max_lines=None,
        interactive=False
    )

    submit_btn = gr.Button("Ask")
    submit_btn.click(chat_fn, inputs=[user_question_input, session_id_input], outputs=output_box)

# Launch
demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
