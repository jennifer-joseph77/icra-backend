"""
ICRA — FastAPI server.
Exposes the RAG pipeline as a simple POST /ask endpoint.

Run:  uvicorn server:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from knowledge_base import get_or_create_collection
from rag_pipeline import generate_answer

logger = logging.getLogger(__name__)

# ── Lifespan: load ChromaDB once at startup ─────────────────────────────────

collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    logging.basicConfig(
        level=logging.INFO if config.VERBOSE else logging.WARNING,
    )
    logger.info("Loading knowledge base into ChromaDB...")
    collection = get_or_create_collection()
    logger.info(f"Ready — {collection.count()} documents indexed.")
    yield


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ICRA", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    id: str
    name: str
    type: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ICRA</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #333;
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; padding: 3rem 1rem; }
  h1 { margin-bottom: .25rem; }
  p.subtitle { color: #666; margin-bottom: 2rem; }
  form { display: flex; gap: .5rem; width: 100%; max-width: 600px; }
  input { flex: 1; padding: .75rem 1rem; border: 1px solid #ccc;
          border-radius: 8px; font-size: 1rem; }
  button { padding: .75rem 1.5rem; background: #2563eb; color: #fff;
           border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  #result { margin-top: 2rem; width: 100%; max-width: 600px; }
  .answer { background: #fff; padding: 1.25rem; border-radius: 8px;
            border: 1px solid #ddd; white-space: pre-wrap; line-height: 1.5; }
  .sources { margin-top: .75rem; font-size: .85rem; color: #666; }
</style>
</head>
<body>
  <h1>ICRA</h1>
  <p class="subtitle">Intelligent Campus Resource Assistant</p>
  <form id="askForm">
    <input id="question" placeholder="Ask about campus facilities and services..." autofocus>
    <button type="submit">Ask</button>
  </form>
  <div id="result"></div>
<script>
  const form = document.getElementById('askForm');
  const input = document.getElementById('question');
  const result = document.getElementById('result');
  const btn = form.querySelector('button');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    btn.disabled = true;
    result.innerHTML = '<p style="color:#888">Thinking...</p>';
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q}),
      });
      const data = await res.json();
      let html = '<div class="answer">' + escapeHtml(data.answer) + '</div>';
      if (data.sources && data.sources.length) {
        const names = data.sources.map(s => s.name).join(', ');
        html += '<div class="sources">Sources: ' + escapeHtml(names) + '</div>';
      }
      result.innerHTML = html;
    } catch (err) {
      result.innerHTML = '<p style="color:red">Error: ' + escapeHtml(err.message) + '</p>';
    } finally {
      btn.disabled = false;
    }
  });

  function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  }
</script>
</body>
</html>"""


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = generate_answer(req.question, collection)
    return AskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
    )
