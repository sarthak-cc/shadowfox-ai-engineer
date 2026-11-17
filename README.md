# ShadowFox — AI Engineer Internship Tasks

Three AI applications of increasing depth, all built on the Google Gemini API.

| Level | Project | Highlights | Status |
|---|---|---|---|
| **Beginner** | [StudyPilot — AI study assistant](level1-student-utility/) | 4 prompt-engineered study tools, structured JSON output, validation, retries with model fallback, interactive quiz UI | ✅ Done |
| **Intermediate** | [Document Q&A assistant](level2-doc-qa/) | PDF/TXT upload, chunking, embeddings, similarity search, grounded answers with visible sources | 🚧 In progress |
| **Advanced** | [Production-style RAG assistant](level3-rag-assistant/) | FastAPI + Pydantic, LangGraph pipeline, FAISS, reranking, groundedness checks, streaming, Docker | ⏳ Planned |

Each level has its own README covering setup, architecture, prompt design and error handling.

## Quick start

Every level runs on its own. You need Python 3.11+ and a free Gemini API key from
[Google AI Studio](https://aistudio.google.com/apikey).

```bash
cd level1-student-utility
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your GEMINI_API_KEY
uvicorn app.main:app --port 8001
```

## Author

**Sarthak** — [github.com/sarthak-cc](https://github.com/sarthak-cc)
