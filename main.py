#!/usr/bin/env python3
"""
ICRA — Intelligent Campus Resource Assistant
Interactive terminal demo for the RAG pipeline.

Run:  python main.py
"""

import logging
import sys

import config
from knowledge_base import get_or_create_collection
from rag_pipeline import generate_answer

# ── ANSI color helpers ──────────────────────────────────────────────────────

BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ── Logging setup ───────────────────────────────────────────────────────────

def setup_logging():
    level = logging.INFO if config.VERBOSE else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"{DIM}%(levelname)s | %(name)s | %(message)s{RESET}",
    )


# ── Display helpers ─────────────────────────────────────────────────────────

WELCOME = f"""
{BOLD}{'=' * 60}
  ICRA — Intelligent Campus Resource Assistant
{'=' * 60}{RESET}

I can answer questions about campus facilities and services:
  • Libraries, computer labs, and makerspaces
  • Food options and dining hours
  • Administrative offices (registrar, financial aid, etc.)
  • Academic departments and faculty contacts
  • Health, safety, IT help, and more

{DIM}Type your question and press Enter.
Type 'quit' or 'exit' to leave.{RESET}
"""


def print_retrieved_docs(result):
    """Print the retrieved documents section for demo visibility."""
    print(f"\n{color('── Retrieved Documents ──', YELLOW)}")
    for i, (doc_meta, dist) in enumerate(
        zip(result.sources, result.distances), 1
    ):
        score_pct = max(0, (1 - dist / 2)) * 100  # rough relevance %
        print(
            f"  {color(f'[{i}]', YELLOW)} "
            f"{color(doc_meta['name'], BOLD)} "
            f"{DIM}({doc_meta['type']}) — "
            f"relevance ≈ {score_pct:.0f}%{RESET}"
        )
    print()


def print_answer(result):
    """Print the generated answer."""
    print(color("── Answer ──", GREEN))
    print(f"{GREEN}{result.answer}{RESET}")
    print()

    # Sources footer
    if result.sources:
        names = ", ".join(s["name"] for s in result.sources)
        print(f"{DIM}Sources: {names}{RESET}")
    print()


# ── Main loop ───────────────────────────────────────────────────────────────

def main():
    setup_logging()
    print(WELCOME)

    # ── Startup checks ──
    if not config.ANTHROPIC_API_KEY:
        print(color(
            "WARNING: ANTHROPIC_API_KEY is not set.\n"
            "Retrieval will still work, but answer generation requires the key.\n"
            "Add it to a .env file — see .env.example for the format.\n",
            RED,
        ))

    # ── Load / build ChromaDB collection ──
    print(f"{DIM}Loading knowledge base into ChromaDB...{RESET}")
    try:
        collection = get_or_create_collection()
    except Exception as e:
        print(color(f"Failed to initialize knowledge base: {e}", RED))
        sys.exit(1)
    print(f"{DIM}Ready! {collection.count()} documents indexed.{RESET}\n")

    # ── Interactive loop ──
    while True:
        try:
            question = input(color("You: ", BLUE)).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        # Run the RAG pipeline
        result = generate_answer(question, collection)

        # Display results
        print_retrieved_docs(result)
        print_answer(result)


if __name__ == "__main__":
    main()
