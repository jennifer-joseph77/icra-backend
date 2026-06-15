"""
ICRA — FastAPI server.
Exposes the RAG pipeline as a simple POST /ask endpoint,
plus CRUD API and management UI for Q&A entries.

Run:  uvicorn server:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import config
from database import init_db, seed_from_json, get_entries, get_entry, create_entry, update_entry, delete_entry
from knowledge_base import get_or_create_collection, add_document, update_document, delete_document
from rag_pipeline import generate_answer

logger = logging.getLogger(__name__)

collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    logging.basicConfig(
        level=logging.INFO if config.VERBOSE else logging.WARNING,
    )
    # Initialize SQLite and seed
    logger.info("Initializing SQLite database...")
    init_db()
    seed_from_json()

    logger.info("Loading knowledge base into ChromaDB...")
    collection = get_or_create_collection()
    logger.info(f"Ready — {collection.count()} documents indexed.")
    yield

app = FastAPI(title="ICRA Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------

class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    id: str
    name: str
    type: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class EntryCreate(BaseModel):
    name: str
    type: str
    location: str = ""
    hours: dict = {}
    description: str = ""
    contact: str = ""
    additional_info: list[str] = []


class EntryUpdate(BaseModel):
    name: str
    type: str
    location: str = ""
    hours: dict = {}
    description: str = ""
    contact: str = ""
    additional_info: list[str] = []


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ICRA Chatbot</title>

<style>
* { margin:0; padding:0; box-sizing:border-box; font-family:system-ui; }

body { background:#343541; color:#ececf1; height:100vh; display:flex; }

.app { display:flex; width:100%; }

.sidebar {
  width:260px;
  background:#202123;
  padding:1rem;
}

.sidebar h2 { margin-bottom:1rem; }

.new-chat {
  background:#343541;
  padding:0.8rem;
  border-radius:6px;
  cursor:pointer;
  border:1px solid #4d4d4f;
  text-align:center;
}

.new-chat:hover { background:#40414f; }

.main {
  flex:1;
  display:flex;
  flex-direction:column;
}

.chat-container {
  flex:1;
  overflow-y:auto;
  padding:2rem 20%;
}

.message {
  padding:1.2rem;
  margin-bottom:1rem;
  border-radius:8px;
  line-height:1.6;
  white-space:pre-wrap;
}

.user { background:#40414f; }
.bot { background:#444654; }

.sources {
  font-size:0.8rem;
  color:#bbb;
  margin-top:0.5rem;
}

.input-area {
  display:flex;
  padding:1rem 20%;
  border-top:1px solid #4d4d4f;
}

textarea {
  flex:1;
  padding:0.9rem;
  border-radius:8px;
  border:none;
  background:#40414f;
  color:white;
  resize:none;
  height:50px;
}

button {
  margin-left:0.5rem;
  padding:0 1.2rem;
  border-radius:8px;
  border:none;
  background:#19c37d;
  font-weight:bold;
  cursor:pointer;
}

button:hover { opacity:0.9; }

.typing::after {
  content:'...';
  animation:dots 1s steps(3,end) infinite;
}

@keyframes dots {
  0%{content:'';}
  33%{content:'.';}
  66%{content:'..';}
  100%{content:'...';}
}
</style>
</head>

<body>

<div class="app">

  <aside class="sidebar">
    <h2>ICRA</h2>
    <div class="new-chat" onclick="newChat()">+ New Chat</div>
  </aside>

  <main class="main">
    <div id="chat" class="chat-container"></div>

    <form id="form" class="input-area">
      <textarea id="question" placeholder="Message ICRA..." required></textarea>
      <button type="submit">Send</button>
    </form>
  </main>

</div>

<script>
const form = document.getElementById("form");
const input = document.getElementById("question");
const chat = document.getElementById("chat");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";

  const typing = addMessage("ICRA is typing", "bot typing");

  try {
    const res = await fetch("/ask", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ question })
    });

    const data = await res.json();
    typing.remove();

    const botMsg = addMessage(data.answer, "bot");

    if (data.sources && data.sources.length) {
      const src = document.createElement("div");
      src.className = "sources";
      src.textContent = "Sources: " + data.sources.map(s=>s.name).join(", ");
      botMsg.appendChild(src);
    }

  } catch(err) {
    typing.remove();
    addMessage("Error connecting to server.", "bot");
  }
});

function addMessage(text, type) {
  const msg = document.createElement("div");
  msg.className = "message " + type;
  msg.textContent = text;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
  return msg;
}

function newChat() {
  chat.innerHTML = "";
}
</script>

</body>
</html>
"""

# ---------------- Backend Route ----------------

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = generate_answer(req.question, collection)
    return AskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
    )


# ── CRUD API for entries ─────────────────────────────────────────────────────


@app.get("/api/entries")
async def list_entries(page: int = Query(1, ge=1), per_page: int = Query(20, ge=1, le=100)):
    entries, total = get_entries(page=page, per_page=per_page)
    return {
        "entries": entries,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


@app.get("/api/entries/{entry_id}")
async def get_single_entry(entry_id: str):
    entry = get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@app.post("/api/entries", status_code=201)
async def create_new_entry(data: EntryCreate):
    entry = create_entry(data.model_dump())
    add_document(collection, entry)
    return entry


@app.put("/api/entries/{entry_id}")
async def update_existing_entry(entry_id: str, data: EntryUpdate):
    entry = update_entry(entry_id, data.model_dump())
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    update_document(collection, entry)
    return entry


@app.delete("/api/entries/{entry_id}", status_code=204)
async def delete_existing_entry(entry_id: str):
    deleted = delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Entry not found")
    delete_document(collection, entry_id)


# ── Management UI ────────────────────────────────────────────────────────────


@app.get("/manage", response_class=HTMLResponse)
async def manage_page():
    return MANAGE_HTML


MANAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ICRA — Manage Entries</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #333;
         padding: 2rem; max-width: 1100px; margin: 0 auto; }
  h1 { margin-bottom: .5rem; }
  .subtitle { color: #666; margin-bottom: 1.5rem; }
  .toolbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
  button { padding: .5rem 1rem; border: none; border-radius: 6px; cursor: pointer; font-size: .9rem; }
  .btn-primary { background: #2563eb; color: #fff; }
  .btn-primary:hover { background: #1d4ed8; }
  .btn-danger { background: #dc2626; color: #fff; }
  .btn-danger:hover { background: #b91c1c; }
  .btn-secondary { background: #e5e7eb; color: #333; }
  .btn-secondary:hover { background: #d1d5db; }
  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th, td { padding: .75rem 1rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
  th { background: #f9fafb; font-weight: 600; font-size: .85rem; color: #6b7280; text-transform: uppercase; }
  td { font-size: .9rem; }
  .actions { display: flex; gap: .5rem; }
  .pagination { display: flex; gap: .5rem; justify-content: center; margin-top: 1rem; align-items: center; }
  .pagination button { padding: .4rem .8rem; }
  .pagination span { font-size: .9rem; color: #666; }
  /* Modal */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                   background: rgba(0,0,0,.4); z-index: 100; justify-content: center; align-items: center; }
  .modal-overlay.active { display: flex; }
  .modal { background: #fff; border-radius: 12px; padding: 2rem; width: 90%; max-width: 600px;
           max-height: 90vh; overflow-y: auto; }
  .modal h2 { margin-bottom: 1rem; }
  .form-group { margin-bottom: 1rem; }
  .form-group label { display: block; font-weight: 500; margin-bottom: .25rem; font-size: .9rem; }
  .form-group input, .form-group textarea { width: 100%; padding: .5rem .75rem; border: 1px solid #d1d5db;
           border-radius: 6px; font-size: .9rem; font-family: inherit; }
  .form-group textarea { min-height: 80px; resize: vertical; }
  .form-actions { display: flex; gap: .5rem; justify-content: flex-end; margin-top: 1.5rem; }
  .empty { text-align: center; padding: 3rem; color: #999; }
  .toast { position: fixed; bottom: 2rem; right: 2rem; background: #065f46; color: #fff;
           padding: .75rem 1.25rem; border-radius: 8px; font-size: .9rem; display: none; z-index: 200; }
  .toast.error { background: #991b1b; }
  .toast.show { display: block; }
</style>
</head>
<body>
  <h1>ICRA — Manage Entries</h1>
  <p class="subtitle">Add, edit, and delete knowledge base entries.</p>

  <div class="toolbar">
    <span id="totalCount"></span>
    <button class="btn-primary" onclick="openCreate()">+ Add Entry</button>
  </div>

  <table>
    <thead>
      <tr><th>Name</th><th>Type</th><th>Location</th><th>Actions</th></tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
  <div id="emptyState" class="empty" style="display:none">No entries yet. Click "Add Entry" to get started.</div>

  <div class="pagination" id="pagination"></div>

  <!-- Modal -->
  <div class="modal-overlay" id="modal">
    <div class="modal">
      <h2 id="modalTitle">Add Entry</h2>
      <form id="entryForm" onsubmit="handleSubmit(event)">
        <input type="hidden" id="entryId">
        <div class="form-group"><label>Name *</label><input id="fName" required></div>
        <div class="form-group"><label>Type *</label><input id="fType" required></div>
        <div class="form-group"><label>Location</label><input id="fLocation"></div>
        <div class="form-group"><label>Description</label><textarea id="fDescription"></textarea></div>
        <div class="form-group"><label>Contact</label><input id="fContact"></div>
        <div class="form-group"><label>Hours (JSON)</label><textarea id="fHours" placeholder='{"monday_friday": "9am-5pm"}'></textarea></div>
        <div class="form-group"><label>Additional Info (one per line)</label><textarea id="fAdditional" placeholder="One item per line"></textarea></div>
        <div class="form-actions">
          <button type="button" class="btn-secondary" onclick="closeModal()">Cancel</button>
          <button type="submit" class="btn-primary" id="submitBtn">Save</button>
        </div>
      </form>
    </div>
  </div>

  <div class="toast" id="toast"></div>

<script>
  const API = '/api/entries';
  let currentPage = 1;
  const perPage = 15;

  async function loadEntries() {
    const res = await fetch(`${API}?page=${currentPage}&per_page=${perPage}`);
    const data = await res.json();
    renderTable(data.entries);
    renderPagination(data);
    document.getElementById('totalCount').textContent = `${data.total} entries`;
  }

  function renderTable(entries) {
    const tbody = document.getElementById('tableBody');
    const empty = document.getElementById('emptyState');
    if (!entries.length) { tbody.innerHTML = ''; empty.style.display = 'block'; return; }
    empty.style.display = 'none';
    tbody.innerHTML = entries.map(e => `
      <tr>
        <td>${esc(e.name)}</td>
        <td>${esc(e.type)}</td>
        <td>${esc(e.location)}</td>
        <td class="actions">
          <button class="btn-secondary" onclick="openEdit('${esc(e.id)}')">Edit</button>
          <button class="btn-danger" onclick="confirmDelete('${esc(e.id)}', '${esc(e.name)}')">Delete</button>
        </td>
      </tr>
    `).join('');
  }

  function renderPagination(data) {
    const el = document.getElementById('pagination');
    if (data.total_pages <= 1) { el.innerHTML = ''; return; }
    el.innerHTML = `
      <button class="btn-secondary" ${currentPage <= 1 ? 'disabled' : ''} onclick="goPage(${currentPage-1})">← Prev</button>
      <span>Page ${data.page} of ${data.total_pages}</span>
      <button class="btn-secondary" ${currentPage >= data.total_pages ? 'disabled' : ''} onclick="goPage(${currentPage+1})">Next →</button>
    `;
  }

  function goPage(p) { currentPage = p; loadEntries(); }

  function openCreate() {
    document.getElementById('modalTitle').textContent = 'Add Entry';
    document.getElementById('entryId').value = '';
    document.getElementById('entryForm').reset();
    document.getElementById('modal').classList.add('active');
  }

  async function openEdit(id) {
    const res = await fetch(`${API}/${id}`);
    if (!res.ok) { toast('Entry not found', true); return; }
    const e = await res.json();
    document.getElementById('modalTitle').textContent = 'Edit Entry';
    document.getElementById('entryId').value = e.id;
    document.getElementById('fName').value = e.name;
    document.getElementById('fType').value = e.type;
    document.getElementById('fLocation').value = e.location || '';
    document.getElementById('fDescription').value = e.description || '';
    document.getElementById('fContact').value = e.contact || '';
    document.getElementById('fHours').value = e.hours ? JSON.stringify(e.hours) : '';
    document.getElementById('fAdditional').value = (e.additional_info || []).join('\\n');
    document.getElementById('modal').classList.add('active');
  }

  function closeModal() { document.getElementById('modal').classList.remove('active'); }

  async function handleSubmit(ev) {
    ev.preventDefault();
    const id = document.getElementById('entryId').value;
    let hours = {};
    try { const h = document.getElementById('fHours').value.trim(); if (h) hours = JSON.parse(h); }
    catch { toast('Invalid JSON in Hours field', true); return; }

    const additional = document.getElementById('fAdditional').value.split('\\n').filter(l => l.trim());

    const body = {
      name: document.getElementById('fName').value,
      type: document.getElementById('fType').value,
      location: document.getElementById('fLocation').value,
      description: document.getElementById('fDescription').value,
      contact: document.getElementById('fContact').value,
      hours,
      additional_info: additional,
    };

    const method = id ? 'PUT' : 'POST';
    const url = id ? `${API}/${id}` : API;
    const res = await fetch(url, { method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
    if (!res.ok) { const err = await res.json(); toast(err.detail || 'Error saving', true); return; }

    closeModal();
    toast(id ? 'Entry updated' : 'Entry created');
    loadEntries();
  }

  async function confirmDelete(id, name) {
    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;
    const res = await fetch(`${API}/${id}`, { method: 'DELETE' });
    if (!res.ok && res.status !== 204) { toast('Error deleting', true); return; }
    toast('Entry deleted');
    loadEntries();
  }

  function toast(msg, isError) {
    const el = document.getElementById('toast');
    el.textContent = msg;
    el.className = 'toast show' + (isError ? ' error' : '');
    setTimeout(() => el.classList.remove('show'), 3000);
  }

  function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

  loadEntries();
</script>
</body>
</html>"""
