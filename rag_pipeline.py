"""
RAG Pipeline for ICRA.
Handles the full Retrieve → Augment → Generate flow:
  1. Takes a user question
  2. Retrieves relevant documents from ChromaDB
  3. Builds a prompt with retrieved context
  4. Sends the prompt to Claude for generation
  5. Returns the answer along with source information
"""

import json
import logging
from dataclasses import dataclass, field

import anthropic
import chromadb
import openai

import config
import database
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
- If the context does not contain enough information to answer the question, call \
the flag_knowledge_gap tool instead of guessing. After calling the tool, still give \
the user a brief, honest response explaining you couldn't fully answer and \
suggesting where they might find help (e.g. Student Services).
- Be concise but helpful. Use bullet points when listing multiple items.
- Always mention the source facility name(s) you used to answer.
- If hours or contact info are in the context, include them in your answer.
"""


# ── flag_knowledge_gap tool ─────────────────────────────────────────────────

FLAG_KNOWLEDGE_GAP_TOOL_NAME = "flag_knowledge_gap"

FLAG_KNOWLEDGE_GAP_DESCRIPTION = (
    "Record a question that could NOT be answered from the retrieved context, "
    "so a human admin can research it and add a proper knowledge base entry later. "
    "Call this whenever the provided context is insufficient, missing, or only "
    "tangentially related to the question — do not guess or fabricate an answer "
    "instead of calling this tool. "
    "IMPORTANT: `context` must NOT contain your answer or any invented facts. It is "
    "purely investigative metadata for a human: what you searched for, which source "
    "documents almost matched but didn't fully answer it, what specific detail was "
    "missing, and any clarifying detail that would help a human write a good answer "
    "later (e.g. 'the retrieved docs cover general library hours but not holiday "
    "hours, which is what was asked')."
)

FLAG_KNOWLEDGE_GAP_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "The user's question, cleaned up / rephrased into a clear, "
                "standalone question (fix typos, resolve pronouns, remove chit-chat)."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "Investigative notes for a human admin — NOT the answer. E.g. what "
                "was searched, which near-miss sources were found, what specific "
                "piece is missing."
            ),
        },
    },
    "required": ["question", "context"],
}

MAX_TOOL_ROUNDS = 1


def _execute_flag_knowledge_gap(tool_input: dict, session_id: str | None) -> str:
    """Execute the flag_knowledge_gap tool call: record it, return a tool-result string."""
    question = (tool_input.get("question") or "").strip()
    context = (tool_input.get("context") or "").strip()
    if not question:
        return "ERROR: question is required and was empty."
    gap = database.create_knowledge_gap(
        question=question, context=context, session_id=session_id
    )
    logger.info(f"Logged knowledge gap {gap['id']}: {question!r}")
    return f"Recorded as knowledge gap {gap['id']}. Continue answering the user as helpfully as you can."


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


def _call_llm(user_message: str) -> str:
    """Dispatch the LLM call to the configured provider."""
    provider = config.LLM_PROVIDER

    if provider == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            return (
                "ERROR: ANTHROPIC_API_KEY is not set. "
                "Please add it to your .env file.\n"
                "See .env.example for the expected format."
            )
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        try:
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except anthropic.AuthenticationError:
            return "ERROR: Invalid Anthropic API key. Please check your .env file."
        except anthropic.RateLimitError:
            return "ERROR: Rate limit exceeded. Please wait a moment and try again."
        except anthropic.APIError as e:
            return f"ERROR: Anthropic API error — {e}"

    elif provider == "gemini":
        if not config.GEMINI_API_KEY:
            return (
                "ERROR: GEMINI_API_KEY is not set. "
                "Please add it to your .env file.\n"
                "See .env.example for the expected format."
            )
        client = openai.OpenAI(
            api_key=config.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        try:
            response = client.chat.completions.create(
                model=config.GEMINI_MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        except openai.AuthenticationError:
            return "ERROR: Invalid Gemini API key. Please check your .env file."
        except openai.RateLimitError:
            return "ERROR: Rate limit exceeded. Please wait a moment and try again."
        except openai.APIError as e:
            return f"ERROR: Gemini API error — {e}"

    else:
        return (
            f"ERROR: Unknown LLM_PROVIDER '{provider}'. "
            "Valid options are 'anthropic' or 'gemini'."
        )


def _call_llm_with_tools(user_message: str, session_id: str | None = None) -> str:
    """
    Like _call_llm, but exposes the flag_knowledge_gap tool to the model and
    runs up to MAX_TOOL_ROUNDS of tool-call round-trips before returning final text.
    """
    provider = config.LLM_PROVIDER

    if provider == "anthropic":
        return _call_anthropic_with_tools(user_message, session_id)
    elif provider == "gemini":
        return _call_gemini_with_tools(user_message, session_id)
    else:
        return (
            f"ERROR: Unknown LLM_PROVIDER '{provider}'. "
            "Valid options are 'anthropic' or 'gemini'."
        )


_INCOMPLETE_RESPONSE_MESSAGE = (
    "I noted this as something I need help with, but couldn't finish generating "
    "a full response. Please try rephrasing your question."
)


def _call_anthropic_with_tools(user_message: str, session_id: str | None) -> str:
    if not config.ANTHROPIC_API_KEY:
        return (
            "ERROR: ANTHROPIC_API_KEY is not set. "
            "Please add it to your .env file.\n"
            "See .env.example for the expected format."
        )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    tools = [{
        "name": FLAG_KNOWLEDGE_GAP_TOOL_NAME,
        "description": FLAG_KNOWLEDGE_GAP_DESCRIPTION,
        "input_schema": FLAG_KNOWLEDGE_GAP_INPUT_SCHEMA,
    }]
    messages: list = [{"role": "user", "content": user_message}]

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        rounds = 0
        while response.stop_reason == "tool_use" and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_use_blocks:
                if block.name == FLAG_KNOWLEDGE_GAP_TOOL_NAME:
                    result_text = _execute_flag_knowledge_gap(block.input, session_id)
                else:
                    result_text = f"ERROR: unknown tool '{block.name}'"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})

            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks) if text_blocks else _INCOMPLETE_RESPONSE_MESSAGE
    except anthropic.AuthenticationError:
        return "ERROR: Invalid Anthropic API key. Please check your .env file."
    except anthropic.RateLimitError:
        return "ERROR: Rate limit exceeded. Please wait a moment and try again."
    except anthropic.APIError as e:
        return f"ERROR: Anthropic API error — {e}"


def _serialize_gemini_tool_call(tc) -> dict:
    """
    Rebuild a tool_call dict to send back to Gemini's OpenAI-compatible endpoint.

    Gemini 3 thinking models attach a `thought_signature` (under
    `extra_content.google`) to each function call, which must be echoed back
    verbatim on the next turn — otherwise the API rejects the request with
    "Function call is missing a thought_signature". Other OpenAI-compatible
    providers won't set this field, so it's carried through only when present.
    """
    serialized = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }
    extra_content = getattr(tc, "extra_content", None)
    if extra_content:
        serialized["extra_content"] = extra_content
    return serialized


def _call_gemini_with_tools(user_message: str, session_id: str | None) -> str:
    if not config.GEMINI_API_KEY:
        return (
            "ERROR: GEMINI_API_KEY is not set. "
            "Please add it to your .env file.\n"
            "See .env.example for the expected format."
        )

    client = openai.OpenAI(
        api_key=config.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": FLAG_KNOWLEDGE_GAP_TOOL_NAME,
            "description": FLAG_KNOWLEDGE_GAP_DESCRIPTION,
            "parameters": FLAG_KNOWLEDGE_GAP_INPUT_SCHEMA,
        },
    }]
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model=config.GEMINI_MODEL,
            max_tokens=1024,
            messages=messages,
            tools=tools,
        )

        rounds = 0
        message = response.choices[0].message
        while message.tool_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [_serialize_gemini_tool_call(tc) for tc in message.tool_calls],
            })

            for tc in message.tool_calls:
                if tc.function.name == FLAG_KNOWLEDGE_GAP_TOOL_NAME:
                    args = json.loads(tc.function.arguments)
                    result_text = _execute_flag_knowledge_gap(args, session_id)
                else:
                    result_text = f"ERROR: unknown tool '{tc.function.name}'"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

            response = client.chat.completions.create(
                model=config.GEMINI_MODEL,
                max_tokens=1024,
                messages=messages,
                tools=tools,
            )
            message = response.choices[0].message

        return message.content or _INCOMPLETE_RESPONSE_MESSAGE
    except openai.AuthenticationError:
        return "ERROR: Invalid Gemini API key. Please check your .env file."
    except openai.RateLimitError:
        return "ERROR: Rate limit exceeded. Please wait a moment and try again."
    except openai.APIError as e:
        return f"ERROR: Gemini API error — {e}"


def generate_answer(
    question: str,
    collection: chromadb.Collection,
    top_k: int = config.TOP_K_RESULTS,
    session_id: str | None = None,
) -> RAGResult:
    """
    Full RAG pipeline: retrieve relevant docs, then generate an answer with Claude.

    Args:
        question: The user's question.
        collection: The ChromaDB collection to search.
        top_k: Number of documents to retrieve.
        session_id: The chat session this question belongs to, if any.

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
        database.create_knowledge_gap(
            question=question,
            context="No documents were retrieved from the knowledge base at all "
                    "(empty ChromaDB result set) for this query.",
            session_id=session_id,
        )
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
    logger.info(f"Sending prompt to {config.LLM_PROVIDER}...")
    answer_text = _call_llm_with_tools(user_message, session_id=session_id)

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
