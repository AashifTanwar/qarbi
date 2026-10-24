# Atlas — Interactive Knowledge Base

Atlas is a browser-based knowledge exploration workspace with a sample graph, source viewer, topic details, contextual chat, document upload, and graph export. The FastAPI backend now provides persistent SQLite storage, document parsing/OCR hooks, configurable AI topic-and-relationship extraction, source-grounded chat retrieval, graph snapshots, document management, and health checks.

## Features implemented

- `index.html` — interactive frontend; serves the demo graph, handles uploads, connects to the backend, sends contextual chat requests, displays source passages, and saves graph snapshots.
- `backend/main.py` — FastAPI backend.
- PDF, DOCX, TXT, Markdown, CSV, JSON and image upload handling.
- PDF text extraction with page markers; image OCR when Tesseract is installed; optional scanned-PDF OCR via Poppler/pdf2image.
- Configurable OpenAI-compatible chat completions API for semantic topic/edge extraction and contextual answers.
- SQLite persistence for documents, extracted topics, graph snapshots, and feedback.
- Duplicate file detection by SHA-256, document listing/get/delete endpoints, graph read/write endpoints, feedback endpoint.
- Source passage retrieval and citation metadata for chatbot answers.
- Heuristic extraction fallback when no AI API key is configured.
- Frontend JSON/CSV export and browser-side workspace restoration.

## 1. Configure the backend

Python 3.10+ is recommended. From the project folder:

### Windows

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### macOS / Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env` and add your own API key:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
ATLAS_AI_EXTRACTION=true
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

The API key is not included in this project and must never be placed in the HTML/frontend. Without a key, the backend still parses documents, persists them, creates heuristic term nodes, and returns an explicit fallback response for chat.

Start the backend from the `backend` directory:

```bash
uvicorn main:app --reload --port 8000
```

Check health at `http://127.0.0.1:8000/api/health` and interactive API docs at `http://127.0.0.1:8000/docs`.

## 2. Run the frontend

In a second terminal, from the project root:

```bash
python -m http.server 5173
```

Open `http://127.0.0.1:5173`. Keep the backend running on port 8000. The frontend checks `/api/health` automatically. If your backend uses another URL, click the status label in the top bar and enter the API base URL.

Opening the HTML via `file://` can block API calls due to browser CORS restrictions; serve it over HTTP as above.

## API endpoints

- `GET /api/health` — API, AI configuration, OCR, and database status.
- `POST /api/upload` — upload, extract, analyze, and persist a document.
- `POST /api/extract` — extract topics from supplied text.
- `POST /api/chat` — contextual answer plus retrieved source passage metadata.
- `GET /api/documents` — list persisted documents.
- `GET /api/documents/{id}` — retrieve document, text, and graph snapshot.
- `DELETE /api/documents/{id}` — remove a document and stored file.
- `GET /api/documents/{id}/graph` — retrieve graph snapshot.
- `PUT /api/documents/{id}/graph` — persist graph changes.
- `POST /api/feedback` — store feedback.

SQLite data is saved to `backend/atlas.db` by default. Override it with `ATLAS_DB_PATH`. Uploaded files are stored under `backend/uploads/`.

## OCR setup

For images, install Tesseract OCR and ensure `tesseract` is on PATH. For scanned PDFs, install Poppler as well as the Python `pdf2image` dependency. OCR support depends on those operating-system tools being installed.

## Important limitations

- Real AI calls require your own valid provider key, an accessible model, and network access from the backend. I cannot provision or test a private API key on your behalf.
- AI extraction uses a structured JSON response from the configured model. Unsupported providers/models may not support JSON response mode; choose a compatible model or adapt the provider call.
- Semantic search currently uses a lightweight lexical passage retriever, not a vector database/embedding index.
- Graph storage is per document. Cross-document concept deduplication/merging is not yet implemented.
- Source page citations are available when parsers retain `[Page N]` markers. Image bounding-box detection and exact visual-region linking are not implemented.
- Authentication, per-user access control, cloud object storage, background job queues, rate limiting, and production deployment hardening are not included. Do not expose this development server publicly with private documents until those protections are added.

## Security

Keep `backend/.env` private. Do not commit credentials. Validate CORS origins before deployment. Uploaded files and extracted text are stored locally until deleted. Add authentication, authorization, rate limiting, retention policies, and secure hosting before production use.
