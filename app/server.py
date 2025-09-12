# server.py
import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import faiss

from pipeline.embedder import Embedder
from pipeline.retrieval import fetch_few_shots
from pipeline.llm import call_medical_llm, LLMCallError
from app.dto import QueryDTO

# ----------------- Logging Setup -----------------
# Only stdout logging for Render (no file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ----------------- Load Precomputed Resources -----------------
def init_resources(
    faiss_file="resources/embeddings/med_embeddings.faiss",
    bm25_file="resources/pickles/syntactic_model_med.pkl",
    examples_file="resources/train.csv"
):
    examples_df = pd.read_csv(examples_file)

    # FAISS
    dimension = 384
    if os.path.exists(faiss_file):
        faiss_index = faiss.read_index(faiss_file)
    else:
        logger.warning(f"⚠️ FAISS file not found at {faiss_file}, creating empty index")
        faiss_index = faiss.IndexFlatIP(dimension)

    # BM25
    if os.path.exists(bm25_file):
        with open(bm25_file, "rb") as f:
            bm25_model = pickle.load(f)
        tokenized_corpus = [q.split() for q in examples_df["Question"]]
    else:
        logger.warning(f"⚠️ BM25 pickle not found at {bm25_file}")
        bm25_model, tokenized_corpus = None, None

    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return examples_df, faiss_index, bm25_model, tokenized_corpus, embedder

examples_df, faiss_index, bm25_model, tokenized_corpus, embedder = init_resources()

# ----------------- Flask App -----------------
app = Flask(__name__)

# ----------------- Stateful Chat History -----------------
chat_histories = {}
MAX_HISTORY = 4  # last 4 turns

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        logger.error("❌ Missing 'question' in request")
        return jsonify({"error": "Please provide a 'question' field"}), 400

    user_question = data["question"]
    session_id = data.get("session_id", "default")
    logger.info(f"💬 Session '{session_id}' new question: {user_question}")

    dto = QueryDTO(user_question=user_question)

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    fewshot_result = fetch_few_shots(
        user_question=dto.user_question,
        faiss_index=faiss_index,
        examples_df=examples_df.copy(),
        embedder=embedder,
        bm25_model=bm25_model,
        tokenized_corpus=tokenized_corpus,
        top_k=2
    )
    dto.few_shot_examples = fewshot_result["few_shot_examples"]
    dto.matched_indices = fewshot_result["matched_indices"]

    rag_context = "\n".join([f"{k}: {v}" for k, v in dto.few_shot_examples.items()])

    recent_history = chat_histories[session_id][-MAX_HISTORY:]
    history_str = ""
    for turn in recent_history:
        history_str += f"{turn['role']}: {turn['content']}\n"

    full_context = rag_context
    if history_str:
        full_context += "\nPrevious conversation:\n" + history_str

    try:
        response_json, usage = call_medical_llm(
            step_name="api_chat",
            user_question=dto.user_question,
            retrieved_context=full_context,
            model_name="gemini-1.5-flash",
            response_format="json"
        )
        dto.answer = response_json.get("answer", "N/A")
        dto.source_examples = response_json.get("source_examples", [])
        dto.usage = usage
        logger.info(f"✅ Session '{session_id}' answer: {dto.answer}")
    except LLMCallError as e:
        logger.error(f"❌ LLM error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    chat_histories[session_id].append({"role": "user", "content": user_question})
    chat_histories[session_id].append({"role": "assistant", "content": dto.answer})

    return jsonify({
        "question": dto.user_question,
        "answer": dto.answer,
        "source_examples": dto.source_examples,
        "usage": dto.usage
    })

# ----------------- Run App -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
