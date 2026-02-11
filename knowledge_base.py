"""
Knowledge Base Loader for ICRA.
Reads campus data from JSON, converts entries into documents,
and loads them into a ChromaDB collection with sentence-transformer embeddings.
"""

import json
import logging
import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)


def load_campus_data(path: str = config.CAMPUS_DATA_PATH) -> list[dict]:
    """Load campus entries from the JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} entries from {path}")
    return data


def entry_to_document(entry: dict) -> str:
    """
    Convert a single campus data entry into a text document for embedding.
    Combines all fields into a readable block so the embedding captures
    the full semantics of the entry.
    """
    # Format hours — they can be a dict with varying keys
    hours_lines = []
    if isinstance(entry.get("hours"), dict):
        for period, time in entry["hours"].items():
            label = period.replace("_", " ").title()
            hours_lines.append(f"  {label}: {time}")
    hours_text = "\n".join(hours_lines) if hours_lines else "  Not specified"

    # Format additional info
    additional = "\n".join(
        f"  - {item}" for item in entry.get("additional_info", [])
    )

    doc = (
        f"Name: {entry['name']}\n"
        f"Type: {entry['type']}\n"
        f"Location: {entry['location']}\n"
        f"Hours:\n{hours_text}\n"
        f"Description: {entry['description']}\n"
        f"Contact: {entry['contact']}\n"
        f"Additional Info:\n{additional}"
    )
    return doc


def get_or_create_collection(
    reset: bool = False,
) -> chromadb.Collection:
    """
    Return a ChromaDB collection, creating it (and populating it) if needed.

    Args:
        reset: If True, delete existing collection and rebuild from scratch.
    """
    # Use sentence-transformers for local, free embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    if reset:
        try:
            client.delete_collection(config.CHROMA_COLLECTION_NAME)
            logger.info("Deleted existing collection (reset=True).")
        except Exception:
            pass  # Collection didn't exist — that's fine

    # get_or_create is idempotent: returns existing if already populated
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        embedding_function=ef,
    )

    # Only populate if the collection is empty
    if collection.count() == 0:
        logger.info("Collection is empty — loading campus data...")
        data = load_campus_data()
        documents = []
        ids = []
        metadatas = []

        for entry in data:
            doc_text = entry_to_document(entry)
            documents.append(doc_text)
            ids.append(entry["id"])
            metadatas.append({
                "name": entry["name"],
                "type": entry["type"],
                "location": entry["location"],
                "contact": entry["contact"],
            })

        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(documents)} documents to ChromaDB.")
    else:
        logger.info(
            f"Collection already has {collection.count()} documents — skipping load."
        )

    return collection


def query_knowledge_base(
    collection: chromadb.Collection,
    query: str,
    top_k: int = config.TOP_K_RESULTS,
) -> dict:
    """
    Query the ChromaDB collection and return the top-k results.

    Returns a dict with keys: documents, metadatas, distances, ids
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    return results
