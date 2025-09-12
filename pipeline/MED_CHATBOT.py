import logging
import pickle
import os
import pandas as pd
import faiss

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.embedder import Embedder
from pipeline.retrieval import fetch_few_shots
from pipeline.llm import call_medical_llm, LLMCallError
from app.dto import QueryDTO




# ---- Configure Logging ----
log_file = "logs/chatbot.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # ensure folder exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),              # logs to console
        logging.FileHandler(log_file, mode='a')  # logs to file (append mode)
    ]
)
logger = logging.getLogger(__name__)

# ---- Load Precomputed Few-Shot Resources ----
def init_fewshot_precomputed(
    faiss_file="resources/embeddings/med_embeddings.faiss",
    bm25_file="resources/pickles/syntactic_model_med.pkl",
    examples_file="resources/train.csv"
):
    examples_df = pd.read_csv(examples_file)

    # FAISS
    dimension = 384  # MiniLM embedding size
    if os.path.exists(faiss_file):
        faiss_index = faiss.read_index(faiss_file)
    else:
        print(f"⚠️ FAISS file not found at {faiss_file}")
        faiss_index = faiss.IndexFlatIP(dimension)

    # BM25
    if os.path.exists(bm25_file):
        with open(bm25_file, "rb") as f:
            bm25_model = pickle.load(f)
        tokenized_corpus = [q.split() for q in examples_df["Question"]]
    else:
        print(f"⚠️ BM25 pickle not found at {bm25_file}")
        bm25_model, tokenized_corpus = None, None

    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return examples_df, faiss_index, bm25_model, tokenized_corpus, embedder


# ---- Main Q&A Loop with Chat History ----
if __name__ == "__main__":
    examples_df, faiss_index, bm25_model, tokenized_corpus, embedder = init_fewshot_precomputed()

    chat_history = []  # store previous turns
    MAX_HISTORY = 4    # keep last 4 turns to avoid slowing down

    while True:
        user_question = input("\n🩺 Your medical question (or 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        dto = QueryDTO(user_question=user_question)

        # 1️⃣ Fetch relevant few-shots
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

        # Build RAG context
        context_parts = [f"{k}: {v}" for k, v in dto.few_shot_examples.items()]
        rag_context = "\n".join(context_parts)

        # Add recent chat history
        recent_history = chat_history[-MAX_HISTORY:]
        history_str = ""
        for turn in recent_history:
            history_str += f"{turn['role']}: {turn['content']}\n"

        full_context = rag_context
        if history_str:
            full_context += "\nPrevious conversation:\n" + history_str

        # 2️⃣ Call LLM
        try:
            response_json, usage = call_medical_llm(
                step_name="tester",
                user_question=dto.user_question,
                retrieved_context=full_context,
                model_name="gemini-1.5-flash",
                response_format="json"
            )
            dto.answer = response_json.get("answer", "N/A")
            dto.source_examples = response_json.get("source_examples", [])
            dto.usage = usage
        except LLMCallError as e:
            dto.answer = f"❌ LLM error: {e}"
            dto.source_examples = []
            dto.usage = {}

        # 3️⃣ Show Results
        print("\n=== Final Answer ===")
        print("Answer:", dto.answer)
        print("Source Examples:", dto.source_examples)
        print("Token Usage:", dto.usage)

        # 4️⃣ Append current turn to chat history
        chat_history.append({"role": "user", "content": dto.user_question})
        chat_history.append({"role": "assistant", "content": dto.answer})
