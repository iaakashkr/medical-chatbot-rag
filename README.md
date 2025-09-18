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
6. [Usage](#usage)
7. [Generating Resources](#generating-resources)
8. [Environment Variables](#environment-variables)
9. [Pipeline Details](#pipeline-details)
10. [FAQ](#faq)
11. [Contributing](#contributing)
12. [License](#license)

---

## Overview
This chatbot combines a **hybrid retrieval system** with LLM-based generation:

- **Semantic Search:** FAISS embeddings for vector similarity.
- **Syntactic Search:** BM25 keyword matching for exact phrase relevance.
- **RAG Integration:** Combines retrieved examples with user questions for LLM input.
- **Conversational Memory:** Keeps chat history for context-aware responses.
- **Structured Output:** Returns JSON containing `answer`, `source_examples`, and usage stats.

Ideal for **hospitals, clinics, medical students, or educational platforms**.
TRY :- https://tinyurl.com/4mf5uwzj

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
│   ├── MED_CHATBOT.py
│   └── dto/                 # DTO folder
│       └── dto.py
│
├── pipeline/                # Core logic for retrieval & LLM calls
│   ├── embedder.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── token_counter.py
│   └── token_tracker.py
│
├── resources/               # Datasets & precomputed embeddings
│   ├── train.csv
│   ├── embeddings/
│   │   └── med_embeddings.faiss
│   └── pickles/
│       └── syntactic_model_med.pkl
│
├── logs/                    # Log files
│
├── .env                     # API keys (ignored in git)
├── requirements.txt         # Dependencies
└── README.md
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
python scripts/setup_resources.py
```
*(if resources already exist, skip this step)*

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
