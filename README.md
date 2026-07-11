# ICRA — Intelligent Campus Resource Assistant

A command-line RAG (Retrieval-Augmented Generation) chatbot that answers questions about campus facilities and services by retrieving relevant documents from a local knowledge base and generating natural language responses via Claude or Gemini.

## Architecture

```
User Question
     │
     ▼
┌──────────────┐     ┌──────────────────┐
│  ChromaDB    │────▶│  Top-K Documents │
│  (retrieval) │     │  (with scores)   │
└──────────────┘     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  LLM API         │
                     │  Claude / Gemini │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Answer + Sources│
                     └──────────────────┘
```

**Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`) — runs locally, no API key needed.
**Vector Store:** ChromaDB — persistent, local storage in `./chroma_db/`.
**LLM:** Anthropic Claude or Google Gemini (configurable via environment variable).

## Setup

### 1. Clone and enter the project

```bash
cd icra-backend
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will download the sentence-transformers embedding model (~80 MB). This only happens once.

### 4. Configure your API keys

```bash
cp .env.example .env
```

Edit `.env` to set your preferred provider and API key(s):

```env
# Choose your LLM provider: "anthropic" (default) or "gemini"
LLM_PROVIDER=anthropic

ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

You only need to set the key for the provider you intend to use.

#### Getting an Anthropic API key

Sign up at [console.anthropic.com](https://console.anthropic.com) and create an API key.

#### Getting a Gemini API key

1. Go to [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)
2. Sign in with your Google account
3. Click **Create API key** and copy the key into `GEMINI_API_KEY`

### 5. Run the server

```bash
uvicorn server:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser to use the web interface.

The API is also available directly at `POST /ask` (see interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)).

### Alternative: Run the terminal demo

```bash
python main.py
```

### 6. Run the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The suite runs against isolated temporary SQLite/ChromaDB instances and a small fixture dataset — it never touches your real `icra.db` or `chroma_db`, and mocks the LLM call so it doesn't hit Anthropic/Gemini. The first run still needs network access to download the `all-MiniLM-L6-v2` embedding model if it isn't already cached locally.

## Sample Queries

Try these questions to see the RAG pipeline in action:

| Question | What it tests |
|---|---|
| "Where is the computer science lab?" | Location lookup |
| "What are the library hours on weekdays?" | Hours retrieval |
| "Where can I find the registrar's office?" | Admin office lookup |
| "Which buildings have computer labs?" | Multi-result retrieval |
| "What food options are available on campus?" | Category-wide query |
| "How do I contact financial aid?" | Contact info retrieval |
| "Where is the CS department located?" | Academic building lookup |
| "Is there late-night food on campus?" | Specific-hours query |
| "Where can I get my student ID replaced?" | Service lookup |
| "What free software can I get as a student?" | Detail retrieval |

## What the Demo Shows

When you ask a question, the terminal displays:

1. **Retrieved Documents** — the top-3 documents ChromaDB found most relevant, with relevance scores
2. **Answer** — Claude's generated response based only on the retrieved context
3. **Sources** — the campus facilities cited in the answer

This makes the RAG pipeline transparent: you can see which documents were retrieved and verify the answer is grounded in them.

## Project Structure

```
icra-backend/
├── server.py             # FastAPI server (chat UI, /ask, entries CRUD, chat sessions)
├── main.py               # Terminal interface (alternative)
├── rag_pipeline.py       # Retrieve → Augment → Generate
├── knowledge_base.py     # JSON loader and ChromaDB indexing
├── database.py           # SQLite layer (entries, chat sessions/messages)
├── config.py             # Settings and environment variables
├── templates/
│   └── index.html        # Chat UI (sessions sidebar, theme toggle)
├── static/
│   ├── script.js
│   └── styles.css
├── data/
│   └── campus_data.json  # 38 campus facility entries
├── tests/                # pytest integration test suite
│   ├── conftest.py
│   └── fixtures/campus_data.json
├── requirements.txt      # Python dependencies
├── requirements-dev.txt  # Test dependencies (pytest, httpx)
├── pytest.ini
├── chroma_db/            # ChromaDB persistent storage (gitignored)
├── icra.db               # SQLite database (gitignored)
├── .env.example          # Template for API key
└── .gitignore
```

## Configuration

All settings are driven by environment variables (`.env`) and `config.py`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | LLM provider: `anthropic` or `gemini` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `GEMINI_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model to use |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `TOP_K_RESULTS` | `5` | Documents retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer model |
| `VERBOSE` | `True` | Show retrieval logs in terminal |

## How the RAG Pipeline Works

1. **Indexing (one-time):** Campus data entries are converted to text documents and embedded using sentence-transformers. The embeddings are stored in a local ChromaDB collection.

2. **Retrieval:** When a user asks a question, the question is embedded with the same model, and ChromaDB finds the top-K most similar documents using cosine distance.

3. **Augmentation:** The retrieved documents are formatted into a context block and inserted into a prompt alongside the user's question.

4. **Generation:** The prompt is sent to Claude with a system message instructing it to answer only from the provided context. Claude generates a natural language response.

## Known Limitations

- Knowledge base is static (loaded from JSON, not a live database)
- No conversation memory — each question is independent
- Embedding model is general-purpose, not fine-tuned for campus queries
- Similarity scores are approximate; irrelevant results can appear for vague queries

## Next Steps (Week 2)

- [x] Web interface with FastAPI
- [ ] Conversation history / multi-turn context
- [ ] Admin interface to add/edit knowledge base entries
- [ ] Evaluation metrics (retrieval precision, answer quality)
