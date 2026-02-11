"""
Configuration for the ICRA (Intelligent Campus Resource Assistant) backend.
Loads settings from environment variables and provides defaults.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Claude Model ---
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "campus_resources"

# --- Data ---
CAMPUS_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "campus_data.json")

# --- Retrieval ---
TOP_K_RESULTS = 3  # Number of documents to retrieve per query

# --- Embedding Model ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model (local, free)

# --- Logging ---
VERBOSE = True  # Show retrieval logs during demo
