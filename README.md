# 🛒 E-Commerce Product Image Analyzer

Auto-generate product **titles, descriptions and tags** from product images.

Upload product photos in bulk → the backend captions each image with **BLIP**,
retrieves relevant product-copy context via **RAG**, and writes a polished
listing with a **GPT-style** text model → review/edit the cards in the Angular
UI → **export to CSV**.

---

## ✨ Features

- **Vision captioning** — `Salesforce/blip-image-captioning-large` describes each image.
- **RAG** — caption is embedded (`all-MiniLM-L6-v2`) and matched against a product-copy
  knowledge base to ground the category, selling points and tags.
- **GPT-style generation** — `google/flan-t5-base` writes the description from the
  caption + retrieved context; titles and tags are derived deterministically.
- **Graceful mock mode** — *no model downloads required to run.* Every ML step falls
  back to a deterministic heuristic, so the full app works end-to-end immediately and
  "lights up" once you install the ML extras.
- **Angular UI** — drag-and-drop **bulk uploader**, **editable product cards**, and
  **one-click CSV export**.

## 🏗️ Architecture

```
Angular (bulk upload, editable cards, CSV)
        │  multipart POST /api/analyze/bulk
        ▼
FastAPI
        ├── CaptioningService   → BLIP  (image → caption)
        ├── RagService          → embed caption, retrieve KB context
        └── GeneratorService    → FLAN-T5 (caption + context → listing)
                                   ↳ title / description / tags / category
```

Pipeline: `backend/app/services/pipeline.py`
Knowledge base: `backend/data/knowledge_base.json`

---

## 🚀 Quick start

### 1. Backend (FastAPI)

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # core only — runs in MOCK mode

# Run the API
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/api/health (tells you mock vs real models)

#### Enable the real models (optional, large download)

```powershell
pip install -r requirements-ml.txt        # torch + transformers + sentence-transformers
```

First request downloads ~3-4 GB of weights to `~/.cache/huggingface` (CPU works,
CUDA GPU is much faster). Copy `.env.example` → `.env` to tune model ids / device.

### 2. Frontend (Angular)

```powershell
cd frontend
npm install
npm start                                 # ng serve → http://localhost:4200
```

The UI talks to `http://localhost:8000` by default
(see `frontend/src/environments/`).

---

## 🔌 API

| Method | Endpoint            | Body                          | Returns                         |
|--------|---------------------|-------------------------------|---------------------------------|
| GET    | `/api/health`       | —                             | model/mock status               |
| POST   | `/api/analyze`      | `file` (image)                | one `ProductAnalysis`           |
| POST   | `/api/analyze/bulk` | `files[]` (images, ≤50)       | `{ count, results[] }`          |

`ProductAnalysis`: `filename, caption, title, description, tags[], category,
rag_sources[], mock, error`.

### Example

```powershell
curl.exe -F "file=@product.jpg" http://localhost:8000/api/analyze
```

---

## 🧩 Configuration (`backend/.env`)

| Var             | Default                                   | Purpose                              |
|-----------------|-------------------------------------------|--------------------------------------|
| `MOCK_MODE`     | `false`                                   | Force mock even if ML libs installed |
| `CAPTION_MODEL` | `Salesforce/blip-image-captioning-large`  | Vision model                         |
| `TEXT_MODEL`    | `google/flan-t5-base`                     | Description generator                |
| `EMBED_MODEL`   | `sentence-transformers/all-MiniLM-L6-v2`  | RAG embeddings                       |
| `DEVICE`        | auto                                      | `cpu` / `cuda`                       |
| `RAG_TOP_K`     | `3`                                       | KB entries retrieved per image       |
| `CORS_ORIGINS`  | `http://localhost:4200,...`               | Allowed frontend origins             |

## 📚 Extending the RAG knowledge base

Add entries to `backend/data/knowledge_base.json` — each has `category`,
`keywords`, `snippet` (style/selling-points reference) and `tag_pool`. Restart
the backend to re-index.

## 📁 Project layout

```
backend/
  app/
    main.py            FastAPI app + /api/health
    config.py          env-driven settings
    schemas.py         Pydantic models
    routers/analyze.py /api/analyze, /api/analyze/bulk
    services/
      captioning.py    BLIP + mock fallback
      rag.py           embeddings + keyword fallback
      generator.py     FLAN-T5 + template fallback
      pipeline.py      orchestration
  data/knowledge_base.json
  requirements.txt / requirements-ml.txt
frontend/              Angular app (uploader, cards, CSV export)
```
