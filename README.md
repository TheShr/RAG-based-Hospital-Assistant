# 🏥 RAG-based Hospital Patient Query Assistant

A **production-grade** Retrieval-Augmented Generation (RAG) system that answers patient queries strictly from hospital documents — with zero hallucination.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
│                                                                     │
│  PDF Upload                                                         │
│     │                                                               │
│     ▼                                                               │
│  PyMuPDF Extraction ──► Page-aware Text                             │
│     │                                                               │
│     ▼                                                               │
│  Sentence-aware Chunker (size=400, overlap=80)                      │
│     │                                                               │
│     ▼                                                               │
│  BGE-small Embeddings (384-dim, L2-normalised, FREE)                │
│     │                                                               │
│     ▼                                                               │
│  Supabase pgvector (IVFFlat index, cosine distance)                 │
│     │                                                               │
│     ▼                                                               │
│  BM25 Index Rebuilt (rank-bm25, in-memory)                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
│                                                                     │
│  User Question                                                      │
│     │                                                               │
│     ├──► BGE Query Embedding (with instruction prefix)              │
│     │         │                                                     │
│     │         ▼                                                     │
│     │    Dense Search (pgvector cosine, top-15)                     │
│     │                                                               │
│     └──► BM25 Sparse Search (keyword, top-15)                       │
│               │                                                     │
│               ▼                                                     │
│          Reciprocal Rank Fusion (RRF, k=60)                         │
│               │                                                     │
│               ▼                                                     │
│          Top-5 chunks + page metadata                               │
│               │                                                     │
│               ▼                                                     │
│          Grounded RAG Prompt (strict no-hallucination)              │
│               │                                                     │
│               ▼                                                     │
│          LLM (Claude / Groq / OpenAI) ──► Answer + [Page X] cites  │
│               │                                                     │
│               ▼                                                     │
│          { "answer": "...", "sources": ["page 1", "page 3"] }       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rag-hospital-assistant/
├── backend/
│   ├── main.py                    # FastAPI app + middleware
│   ├── api/
│   │   └── routes.py              # /upload, /query, /documents
│   ├── core/
│   │   ├── config.py              # Pydantic settings (env-based)
│   │   ├── schemas.py             # Request/Response models
│   │   └── dependencies.py        # Singleton service container (DI)
│   └── services/
│       ├── ingestion.py           # PDF extraction + smart chunking
│       ├── embeddings.py          # BGE-small sentence-transformers
│       ├── vector_store.py        # Supabase pgvector CRUD + search
│       ├── hybrid_search.py       # BM25 + dense + RRF fusion
│       └── rag_pipeline.py        # Orchestrator + prompt + LLM call
├── frontend/
│   └── app.py                     # Streamlit UI
├── tests/
│   └── test_rag.py                # pytest test suite
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Clone & Install

```bash
git clone <your-repo>
cd rag-hospital-assistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Supabase and LLM API keys
```

### 3. Set Up Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the schema from the API:

```bash
# Start the backend first, then hit:
curl http://localhost:8000/api/v1/setup-schema
# Copy the printed SQL and run it in Supabase SQL Editor
```

Or manually enable pgvector and create tables:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    total_pages INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    total_pages INTEGER,
    embedding vector(384)
);

CREATE INDEX idx_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- Similarity search function
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5,
    filter_doc_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    id TEXT, document_id TEXT, filename TEXT,
    page_number INTEGER, chunk_index INTEGER,
    content TEXT, total_pages INTEGER, similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT dc.id, dc.document_id, dc.filename, dc.page_number,
           dc.chunk_index, dc.content, dc.total_pages,
           1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE (filter_doc_id IS NULL OR dc.document_id = filter_doc_id)
      AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

### 4. Run the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 5. Run the Streamlit UI

```bash
cd frontend
streamlit run app.py
```

---

## API Usage

### Upload a Document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@hospital_document.pdf"
```

Response:
```json
{
  "document_id": "3fa85f64-...",
  "filename": "hospital_document.pdf",
  "total_chunks": 47,
  "total_pages": 10,
  "message": "Document ingested successfully.",
  "processing_time_ms": 2340.5
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are OPD timings?"}'
```

Response:
```json
{
  "answer": "The OPD (Outpatient Department) timings are 08:00 AM to 08:00 PM, Monday to Saturday. On Sundays, a limited Sunday Clinic operates from 09:00 AM to 01:00 PM for select departments. [Page 1]",
  "sources": ["page 1"],
  "source_chunks": [
    {
      "page": 1,
      "chunk_index": 2,
      "text_preview": "[Page 1] Operating Hours: Emergency Services: 24/7...",
      "similarity_score": 0.9124
    }
  ],
  "document_id": null,
  "latency_ms": 842.3,
  "retrieval_method": "hybrid"
}
```

### Fallback (not in document)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -d '{"question": "What is the wifi password?"}'
```

```json
{
  "answer": "I don't have that information in the provided document.",
  "sources": [],
  ...
}
```

---

## Sample Queries

| Question | Expected Answer Source |
|----------|----------------------|
| What are OPD timings? | Page 1 |
| Who is the cardiologist? | Page 2 |
| What is the cost of MRI? | Page 3 |
| Can I cancel appointment within 24 hours? | Page 4 |
| What is ICU cost per day? | Page 5 |
| Emergency number? | Page 1 |
| What is a private room cost? | Page 5 |
| Does the hospital support Hindi? | Page 1 |

---

## Design Decisions

### Why BGE-small over OpenAI embeddings?
- **Free** — no API costs for ingestion or retrieval
- **384-dim** — small, fast, fits in Supabase free tier
- **SOTA** for its size on MTEB benchmark
- Instruction-tuned: query prefix boosts retrieval accuracy

### Why Hybrid Search?
- Dense alone misses exact matches: `"1066"`, `"$600"`, `"ICU"`
- BM25 alone misses semantic queries: `"heart doctor"` → cardiologist
- **RRF fusion** gives the best of both with no tuning needed

### Why temperature=0.0?
Deterministic generation prevents the LLM from "creativity-ing" its way into hallucinations on factual medical data.

### Why sentence-aware chunking?
Mid-sentence cuts destroy context. Our chunker finds natural sentence boundaries, ensuring each chunk is semantically complete.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| PDF Extraction | PyMuPDF (fitz) |
| Embeddings | BGE-small-en-v1.5 (sentence-transformers) |
| Vector DB | Supabase (PostgreSQL + pgvector) |
| Sparse Search | rank-bm25 |
| LLM | Claude / Groq / OpenAI (configurable) |
| UI | Streamlit |
| Config | Pydantic Settings |
#   R A G - b a s e d - H o s p i t a l - A s s i s t a n t  
 