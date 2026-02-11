"""
RAG Pipeline for ICRA.
Handles the full Retrieve → Augment → Generate flow:
  1. Takes a user question
  2. Retrieves relevant documents from ChromaDB
  3. Builds a prompt with retrieved context
  4. Sends the prompt to Claude for generation
  5. Returns the answer along with source information
"""

import logging
from dataclasses import dataclass, field

import anthropic
import chromadb

import config
from knowledge_base import query_knowledge_base

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Container for a RAG pipeline result."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)


SYSTEM_PROMPT = """\
You are ICRA, the Intelligent Campus Resource Assistant. Your job is to help \
students, faculty, and visitors find information about campus facilities and \
services.

Rules:
- Answer ONLY based on the provided context documents. Do not make up information.
- If the context does not contain enough information to answer, say so honestly \
and suggest where the user might find help.
- Be concise but helpful. Use bullet points when listing multiple items.
- Always mention the source facility name(s) you used to answer.
- If hours or contact info are in the context, include them in your answer.
"""


def build_context_block(results: dict) -> str:
    """
    Format retrieved documents into a context block for the prompt.
    """
    docs = results["documents"][0]  # ChromaDB returns nested lists
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    blocks = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        blocks.append(
            f"--- Source {i} (similarity distance: {dist:.4f}) ---\n"
            f"{doc}\n"
        )
    return "\n".join(blocks)


def generate_answer(
    question: str,
    collection: chromadb.Collection,
    top_k: int = config.TOP_K_RESULTS,
) -> RAGResult:
    """
    Full RAG pipeline: retrieve relevant docs, then generate an answer with Claude.

    Args:
        question: The user's question.
        collection: The ChromaDB collection to search.
        top_k: Number of documents to retrieve.

    Returns:
        RAGResult with the generated answer and source metadata.
    """
    # ---- Step 1: Retrieve ----
    logger.info(f"Retrieving top-{top_k} documents for: '{question}'")
    results = query_knowledge_base(collection, question, top_k=top_k)

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    if not docs:
        return RAGResult(
            answer="I couldn't find any relevant information in the campus "
                   "knowledge base. Please try rephrasing your question or "
                   "contact Student Services for help.",
            sources=[],
            retrieved_docs=[],
            distances=[],
        )

    # Log retrieval details
    for i, (meta, dist) in enumerate(zip(metadatas, distances), 1):
        logger.info(
            f"  [{i}] {meta['name']} (type={meta['type']}, "
            f"distance={dist:.4f})"
        )

    # ---- Step 2: Augment — build the prompt ----
    context_block = build_context_block(results)

    user_message = (
        f"Context documents:\n\n{context_block}\n\n"
        f"Student question: {question}\n\n"
        f"Provide a helpful answer based on the context above."
    )

    # ---- Step 3: Generate ----
    logger.info("Sending prompt to Claude...")

    if not config.ANTHROPIC_API_KEY:
        return RAGResult(
            answer="ERROR: ANTHROPIC_API_KEY is not set. "
                   "Please add it to your .env file.\n"
                   "See .env.example for the expected format.",
            sources=[],
            retrieved_docs=docs,
            distances=distances,
        )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        answer_text = response.content[0].text
    except anthropic.AuthenticationError:
        answer_text = (
            "ERROR: Invalid Anthropic API key. Please check your .env file."
        )
    except anthropic.RateLimitError:
        answer_text = (
            "ERROR: Rate limit exceeded. Please wait a moment and try again."
        )
    except anthropic.APIError as e:
        answer_text = f"ERROR: Anthropic API error — {e}"

    # ---- Build result ----
    sources = [
        {"id": ids[i], "name": metadatas[i]["name"], "type": metadatas[i]["type"]}
        for i in range(len(ids))
    ]

    return RAGResult(
        answer=answer_text,
        sources=sources,
        retrieved_docs=docs,
        distances=distances,
    )
