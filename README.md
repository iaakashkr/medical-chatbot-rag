# Medical Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot designed for answering medical FAQs.  
It leverages semantic embeddings (FAISS) and syntactic search (BM25) to fetch relevant context, and Google Gemini LLM to generate accurate, context-aware responses.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Folder Structure](#folder-structure)
5. [Setup](#setup)
6. [Deployment (Docker + Modes)](#deployment-docker--modes)
7. [Usage](#usage)
8. [Generating Resources](#generating-resources)
9. [Environment Variables](#environment-variables)
10. [Pipeline Details](#pipeline-details)
11. [FAQ](#faq)
12. [Contributing](#contributing)
13. [License](#license)

---

## Overview
This chatbot combines a **hybrid retrieval system** with LLM-based generation:

- **Semantic Search:** FAISS embeddings for vector similarity.
- **Syntactic Search:** BM25 keyword matching for exact phrase relevance.
- **RAG Integration:** Combines retrieved examples with user questions for LLM input.
- **Conversational Memory:** Keeps chat history for context-aware responses.
- **Structured Output:** Returns JSON containing `answer`, `source_examples`, and usage stats.

Ideal for **hospitals, clinics, medical students, or educational platforms**.

---

## Try It Online
You can try the chatbot live **without installing anything**:

[Click here to test the Medical BOT](https://huggingface.co/spaces/iaakashkr/Medical_BOT)

Sample: <img width="1918" height="923" alt="image" src="https://github.com/user-attachments/assets/7e6de580-483a-4c1c-9f9d-ee55f58671b6" />
:
```
**RESPONSE**
```
<img width="1919" height="924" alt="image" src="https://github.com/user-attachments/assets/0275ea2c-705f-4d91-9809-2c83f360f706" />

---

## Features
- Retrieval-Augmented Generation (RAG)
- Few-shot example selection using FAISS + BM25
- Context-aware responses from Google Gemini LLM
- JSON output with `answer` and `source_examples`
- Chat history management
- Modular, reproducible pipeline
- Token usage tracking

---

## Architecture
```
User Input
   │
   ▼
[Few-Shot Retrieval] ←─ FAISS Embeddings + BM25 ──→ Candidate Examples
   │
   ▼
[Context Builder] → Merge examples + chat history
   │
   ▼
[LLM Generation] → Google Gemini LLM
   │
   ▼
[JSON Output] → Answer + Source Examples + Token Usage
   │
   ▼
User receives response via CLI or API
```

---

## Folder Structure

```
medical-chatbot-rag/
│
├── app/                     
│   ├── __init__.py
│   └── dto.py
│
├── DTO/                     
│   └── dto.py
│
├── pipeline/                 # Core logic for retrieval & LLM calls
│   ├── embedder.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── token_counter.py
│   └── token_tracker.py
│
├── resources/                # Datasets & precomputed embeddings
│   ├── train.csv
│   ├── embeddings/
│   └── pickles/
│
├── tests/                    # Unit tests
│   ├── test_embedder.py
│   ├── test_llm.py
│   └── test_retrieval.py
│
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
```

---

## Setup

1. **Clone Repo**
```bash
git clone https://github.com/iaakashkr/medical-chatbot-rag.git
cd medical-chatbot-rag
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Create `.env`**
```text
GEMINI_API_KEY=your_google_gemini_api_key
```

4. **Generate FAISS Embeddings & BM25 Pickle**
```bash
python pipeline/embedder.py
```
*(if resources already exist, skip this step)*

---

## Deployment (Docker + Modes)

### 🩺 Modes

#### 1) FastAPI (default)
- Set: `RUN_MODE=fastapi`  
- Start server locally:  
  ```bash
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
  ```
- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  

#### 2) Gradio (GUI)
- Set: `RUN_MODE=gradio`  
- Run:  
  ```bash
  python app.py
  ```
- Browser: [http://localhost:7860](http://localhost:7860)  

---

### 📦 Docker Setup

#### Pull prebuilt image
```bash
docker pull iakashkr/medical-chatbot:latest
```

#### Run FastAPI mode
```bash
docker run -d -p 8000:8000 -e GEMINI_API_KEY_1=<YOUR_API_KEY> iakashkr/medical-chatbot
```

#### Run Gradio mode
```bash
docker run -d -p 7860:7860 -e RUN_MODE=gradio -e GEMINI_API_KEY_1=<YOUR_API_KEY> iakashkr/medical-chatbot
```

---

### ⚡ API Usage Example
**Endpoint:** `POST /chat`  
```json
{
  "user_question": "What are the symptoms of flu?",
  "session_id": ""
}
```
**Response:**
```json
{
  "answer": "Flu symptoms can include fever, cough, sore throat, muscle aches, fatigue, and headache.",
  "session_id": "04759a64-2505-47d9-accb-21f1b40b5e4b"
}
```

---

## Usage

### CLI
```bash
python app/MED_CHATBOT.py
```

Type medical questions and get structured answers with source examples.

---

## Generating Resources
If you want to update your embeddings or BM25 models:

1. Load dataset: `resources/train.csv`
2. Run embedding script:
```bash
python scripts/generate_embeddings.py
```
3. FAISS index saved to `resources/embeddings/med_embeddings.faiss`
4. BM25 pickle saved to `resources/pickles/syntactic_model_med.pkl`

---

## Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `RUN_MODE`: `"fastapi"` (default) or `"gradio"`
- Optionally, adjust model name or thresholds in `pipeline/retrieval.py`

---

## Pipeline Details
- **Few-Shot Retrieval:** Combines FAISS semantic similarity and BM25 syntactic scores to select top-K examples.
- **Exact Match Bonus:** Slightly increases score if user query exactly matches example questions.
- **LLM Call:** Uses `llm_medical.py` to send context and question to Gemini, returns structured JSON.
- **Chat History:** Maintains last N turns for continuity.

---

## FAQ

**Q:** Do I need API keys to run locally?  
**A:** Yes, `.env` must contain your Gemini API key.

**Q:** Can I use precomputed embeddings?  
**A:** Yes, keep `med_embeddings.faiss` and BM25 pickle in `resources/`.

**Q:** How many examples does the bot fetch?  
**A:** Top 2 by default, configurable in `fetch_few_shots()`.

---

## Contributing
- Fork repo
- Create branch: `git checkout -b feature/your-feature`
- Commit changes
- Push: `git push origin feature/your-feature`
- Open PR

---

## License
This project is licensed under the [Apache 2.0 License](https://github.com/iaakashkr/medical-chatbot-rag/blob/main/LICENSE) © 2025 Akash Kumar
