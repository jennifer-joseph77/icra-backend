"""
ICRA Chatbot - Full Integrated System
Run: uvicorn server:app --reload
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

collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    logging.basicConfig(
        level=logging.INFO if config.VERBOSE else logging.WARNING,
    )
    logger.info("Loading knowledge base...")
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

# ---------------- Frontend ----------------

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
