"""
ICRA — FastAPI server.
Exposes the RAG pipeline as a simple POST /ask endpoint,
plus CRUD API and management UI for Q&A entries.

Run:  uvicorn server:app --reload
"""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from database import (
    get_connection,
    init_db,
    seed_from_json,
    get_entries,
    get_entry,
    create_entry,
    update_entry,
    delete_entry,
    get_session_messages,
    delete_session,
    get_knowledge_gaps,
    get_knowledge_gap,
    update_knowledge_gap_text,
    resolve_knowledge_gap,
    dismiss_knowledge_gap,
)
from knowledge_base import get_or_create_collection, add_document, update_document, delete_document
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
    # Initialize SQLite and seed
    logger.info("Initializing SQLite database...")
    init_db()
    seed_from_json()

    logger.info("Loading knowledge base into ChromaDB...")
    collection = get_or_create_collection()
    logger.info(f"Ready — {collection.count()} documents indexed.")
    yield


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ICRA", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


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


class KnowledgeGapUpdate(BaseModel):
    question: str
    context: str = ""


class KnowledgeGapResolve(BaseModel):
    question: str
    answer: str
    type: str
    name: str | None = None
    location: str = ""
    contact: str = ""
    hours: dict = {}
    additional_info: list[str] = []


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if req.session_id:
        conn = get_connection()
        row = conn.execute(
            "SELECT id FROM chat_sessions WHERE id = ?", (req.session_id,)
        ).fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Session not found")

        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (req.session_id, "user", req.question),
        )
        conn.commit()

        row = conn.execute(
            "SELECT title FROM chat_sessions WHERE id = ?", (req.session_id,)
        ).fetchone()
        if row and row["title"] == "New Chat":
            title = req.question[:35] + "..." if len(req.question) > 35 else req.question
            conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ?",
                (title, req.session_id),
            )
            conn.commit()

        conn.close()

    result = generate_answer(req.question, collection, session_id=req.session_id)

    if req.session_id:
        conn = get_connection()
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (req.session_id, "assistant", result.answer),
        )
        conn.commit()
        conn.close()

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


# ── Knowledge gaps ───────────────────────────────────────────────────────────


@app.get("/api/knowledge-gaps")
async def list_knowledge_gaps(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str | None = Query(None, pattern="^(open|resolved|dismissed)$"),
):
    gaps, total = get_knowledge_gaps(page=page, per_page=per_page, status=status)
    return {
        "gaps": gaps,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


@app.get("/api/knowledge-gaps/{gap_id}")
async def get_single_knowledge_gap(gap_id: str):
    gap = get_knowledge_gap(gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")
    return gap


@app.patch("/api/knowledge-gaps/{gap_id}")
async def update_knowledge_gap(gap_id: str, data: KnowledgeGapUpdate):
    gap = update_knowledge_gap_text(gap_id, data.question, data.context)
    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")
    return gap


@app.post("/api/knowledge-gaps/{gap_id}/resolve")
async def resolve_gap(gap_id: str, data: KnowledgeGapResolve):
    gap = get_knowledge_gap(gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")
    if gap["status"] != "open":
        raise HTTPException(status_code=400, detail=f"Gap is already {gap['status']}")

    entry_data = {
        "name": data.name or data.question,
        "type": data.type,
        "location": data.location,
        "hours": data.hours,
        "description": data.answer,
        "contact": data.contact,
        "additional_info": data.additional_info,
    }
    entry = create_entry(entry_data)
    add_document(collection, entry)

    updated_gap = resolve_knowledge_gap(gap_id, entry["id"])
    return {"gap": updated_gap, "entry": entry}


@app.post("/api/knowledge-gaps/{gap_id}/dismiss")
async def dismiss_gap(gap_id: str):
    gap = dismiss_knowledge_gap(gap_id)
    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")
    return gap


# ── Chat sessions ────────────────────────────────────────────────────────────


@app.post("/sessions")
async def create_session():
    session_id = str(uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_sessions (id, title) VALUES (?, ?)",
        (session_id, "New Chat"),
    )
    conn.commit()
    conn.close()
    return {"session_id": session_id}


@app.get("/sessions")
async def list_sessions():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM chat_sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str):
    return get_session_messages(session_id)


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}


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
  .tabs { display: flex; gap: .5rem; margin-bottom: 1rem; }
  .tab-btn { background: #e5e7eb; color: #333; }
  .tab-btn.active { background: #2563eb; color: #fff; }
  .gap-question { font-weight: 500; }
  .gap-context { font-size: .85rem; color: #666; font-style: italic; margin-top: .25rem; }
  .status-badge { display: inline-block; padding: .15rem .5rem; border-radius: 999px; font-size: .8rem; }
  .status-open { background: #fef3c7; color: #92400e; }
  .status-resolved { background: #d1fae5; color: #065f46; }
  .status-dismissed { background: #e5e7eb; color: #4b5563; }
</style>
</head>
<body>
  <h1>ICRA — Manage Entries</h1>
  <p class="subtitle">Add, edit, and delete knowledge base entries.</p>

  <div class="tabs">
    <button class="tab-btn active" id="tabEntriesBtn" onclick="switchTab('entries')">Entries</button>
    <button class="tab-btn" id="tabGapsBtn" onclick="switchTab('gaps')">Knowledge Gaps</button>
  </div>

  <div id="entriesView">
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
  </div>

  <div id="gapsView" style="display:none">
    <div class="toolbar">
      <span id="gapsTotalCount"></span>
    </div>

    <table>
      <thead>
        <tr><th>Question</th><th>Status</th><th>Actions</th></tr>
      </thead>
      <tbody id="gapsTableBody"></tbody>
    </table>
    <div id="gapsEmptyState" class="empty" style="display:none">No open knowledge gaps.</div>

    <div class="pagination" id="gapsPagination"></div>
  </div>

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

  <!-- Resolve Gap Modal -->
  <div class="modal-overlay" id="resolveModal">
    <div class="modal">
      <h2>Resolve Knowledge Gap</h2>
      <p class="gap-context" id="resolveGapContext"></p>
      <form id="resolveForm" onsubmit="handleResolveSubmit(event)">
        <input type="hidden" id="resolveGapId">
        <div class="form-group"><label>Question *</label><input id="rQuestion" required></div>
        <div class="form-group"><label>Answer *</label><textarea id="rAnswer" required></textarea></div>
        <div class="form-group"><label>Type *</label><input id="rType" required></div>
        <div class="form-group"><label>Name</label><input id="rName" placeholder="Defaults to the question above"></div>
        <div class="form-group"><label>Location</label><input id="rLocation"></div>
        <div class="form-group"><label>Contact</label><input id="rContact"></div>
        <div class="form-group"><label>Hours (JSON)</label><textarea id="rHours" placeholder='{"monday_friday": "9am-5pm"}'></textarea></div>
        <div class="form-group"><label>Additional Info (one per line)</label><textarea id="rAdditional" placeholder="One item per line"></textarea></div>
        <div class="form-actions">
          <button type="button" class="btn-secondary" onclick="closeResolveModal()">Cancel</button>
          <button type="submit" class="btn-primary">Resolve &amp; Create Entry</button>
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

  // ── Knowledge gaps ─────────────────────────────────────────────────────
  const GAPS_API = '/api/knowledge-gaps';
  let gapsCurrentPage = 1;
  let gapsLoaded = false;
  const gapsPerPage = 15;

  function switchTab(tab) {
    const entriesActive = tab === 'entries';
    document.getElementById('entriesView').style.display = entriesActive ? '' : 'none';
    document.getElementById('gapsView').style.display = entriesActive ? 'none' : '';
    document.getElementById('tabEntriesBtn').classList.toggle('active', entriesActive);
    document.getElementById('tabGapsBtn').classList.toggle('active', !entriesActive);
    if (!entriesActive && !gapsLoaded) {
      gapsLoaded = true;
      loadGaps();
    }
  }

  async function loadGaps() {
    const res = await fetch(`${GAPS_API}?page=${gapsCurrentPage}&per_page=${gapsPerPage}&status=open`);
    const data = await res.json();
    renderGapsTable(data.gaps);
    renderGapsPagination(data);
    document.getElementById('gapsTotalCount').textContent = `${data.total} open gaps`;
  }

  function renderGapsTable(gaps) {
    const tbody = document.getElementById('gapsTableBody');
    const empty = document.getElementById('gapsEmptyState');
    if (!gaps.length) { tbody.innerHTML = ''; empty.style.display = 'block'; return; }
    empty.style.display = 'none';
    tbody.innerHTML = gaps.map(g => `
      <tr>
        <td>
          <div class="gap-question">${esc(g.question)}</div>
          <div class="gap-context">${esc(g.context)}</div>
        </td>
        <td><span class="status-badge status-${esc(g.status)}">${esc(g.status)}</span></td>
        <td class="actions">
          <button class="btn-primary" onclick="openResolve('${esc(g.id)}')">Resolve</button>
          <button class="btn-danger" onclick="confirmDismiss('${esc(g.id)}')">Dismiss</button>
        </td>
      </tr>
    `).join('');
  }

  function renderGapsPagination(data) {
    const el = document.getElementById('gapsPagination');
    if (data.total_pages <= 1) { el.innerHTML = ''; return; }
    el.innerHTML = `
      <button class="btn-secondary" ${gapsCurrentPage <= 1 ? 'disabled' : ''} onclick="goGapsPage(${gapsCurrentPage-1})">← Prev</button>
      <span>Page ${data.page} of ${data.total_pages}</span>
      <button class="btn-secondary" ${gapsCurrentPage >= data.total_pages ? 'disabled' : ''} onclick="goGapsPage(${gapsCurrentPage+1})">Next →</button>
    `;
  }

  function goGapsPage(p) { gapsCurrentPage = p; loadGaps(); }

  async function openResolve(id) {
    const res = await fetch(`${GAPS_API}/${id}`);
    if (!res.ok) { toast('Knowledge gap not found', true); return; }
    const g = await res.json();
    document.getElementById('resolveGapId').value = g.id;
    document.getElementById('resolveGapContext').textContent = g.context;
    document.getElementById('rQuestion').value = g.question;
    document.getElementById('rAnswer').value = '';
    document.getElementById('rType').value = '';
    document.getElementById('rName').value = '';
    document.getElementById('rLocation').value = '';
    document.getElementById('rContact').value = '';
    document.getElementById('rHours').value = '';
    document.getElementById('rAdditional').value = '';
    document.getElementById('resolveModal').classList.add('active');
  }

  function closeResolveModal() { document.getElementById('resolveModal').classList.remove('active'); }

  async function handleResolveSubmit(ev) {
    ev.preventDefault();
    const id = document.getElementById('resolveGapId').value;
    let hours = {};
    try { const h = document.getElementById('rHours').value.trim(); if (h) hours = JSON.parse(h); }
    catch { toast('Invalid JSON in Hours field', true); return; }

    const additional = document.getElementById('rAdditional').value.split('\\n').filter(l => l.trim());

    const body = {
      question: document.getElementById('rQuestion').value,
      answer: document.getElementById('rAnswer').value,
      type: document.getElementById('rType').value,
      name: document.getElementById('rName').value || null,
      location: document.getElementById('rLocation').value,
      contact: document.getElementById('rContact').value,
      hours,
      additional_info: additional,
    };

    const res = await fetch(`${GAPS_API}/${id}/resolve`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    });
    if (!res.ok) { const err = await res.json(); toast(err.detail || 'Error resolving', true); return; }

    closeResolveModal();
    toast('Gap resolved — entry created');
    loadGaps();
  }

  async function confirmDismiss(id) {
    if (!confirm('Dismiss this knowledge gap without creating an entry?')) return;
    const res = await fetch(`${GAPS_API}/${id}/dismiss`, { method: 'POST' });
    if (!res.ok) { toast('Error dismissing', true); return; }
    toast('Gap dismissed');
    loadGaps();
  }

  loadEntries();
</script>
</body>
</html>"""
