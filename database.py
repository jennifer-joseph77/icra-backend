"""
SQLite database layer for ICRA.
Persists Q&A entries and provides CRUD operations.
"""

import json
import sqlite3
import uuid
from pathlib import Path

import config


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            location TEXT NOT NULL DEFAULT '',
            hours TEXT NOT NULL DEFAULT '{}',
            description TEXT NOT NULL DEFAULT '',
            contact TEXT NOT NULL DEFAULT '',
            additional_info TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT '',
            session_id TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            resolved_entry_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE SET NULL,
            FOREIGN KEY(resolved_entry_id) REFERENCES entries(id) ON DELETE SET NULL
        )
    """)
    conn.commit()
    conn.close()


def seed_from_json():
    """Seed the database from campus_data.json if empty."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    if count > 0:
        conn.close()
        return

    data_path = Path(config.CAMPUS_DATA_PATH)
    if not data_path.exists():
        conn.close()
        return

    with open(data_path) as f:
        entries = json.load(f)

    for entry in entries:
        conn.execute(
            """INSERT INTO entries (id, name, type, location, hours, description, contact, additional_info)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry["id"],
                entry["name"],
                entry["type"],
                entry.get("location", ""),
                json.dumps(entry.get("hours", {})),
                entry.get("description", ""),
                entry.get("contact", ""),
                json.dumps(entry.get("additional_info", [])),
            ),
        )

    conn.commit()
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict with parsed JSON fields."""
    d = dict(row)
    d["hours"] = json.loads(d["hours"]) if d["hours"] else {}
    d["additional_info"] = json.loads(d["additional_info"]) if d["additional_info"] else []
    return d


def get_entries(page: int = 1, per_page: int = 20) -> tuple[list[dict], int]:
    """Return a page of entries and the total count."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    offset = (page - 1) * per_page
    rows = conn.execute(
        "SELECT * FROM entries ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (per_page, offset),
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows], total


def get_entry(entry_id: str) -> dict | None:
    """Get a single entry by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def create_entry(data: dict) -> dict:
    """Create a new entry. Returns the created entry."""
    entry_id = data.get("id") or f"user-{uuid.uuid4().hex[:8]}"
    conn = get_connection()
    conn.execute(
        """INSERT INTO entries (id, name, type, location, hours, description, contact, additional_info)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry_id,
            data["name"],
            data["type"],
            data.get("location", ""),
            json.dumps(data.get("hours", {})),
            data.get("description", ""),
            data.get("contact", ""),
            json.dumps(data.get("additional_info", [])),
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


def update_entry(entry_id: str, data: dict) -> dict | None:
    """Update an existing entry. Returns the updated entry or None if not found."""
    conn = get_connection()
    existing = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    if not existing:
        conn.close()
        return None

    conn.execute(
        """UPDATE entries SET name=?, type=?, location=?, hours=?, description=?, contact=?, additional_info=?, updated_at=datetime('now')
           WHERE id=?""",
        (
            data["name"],
            data["type"],
            data.get("location", ""),
            json.dumps(data.get("hours", {})),
            data.get("description", ""),
            data.get("contact", ""),
            json.dumps(data.get("additional_info", [])),
            entry_id,
        ),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    conn.close()
    return _row_to_dict(row)


def delete_entry(entry_id: str) -> bool:
    """Delete an entry. Returns True if deleted, False if not found."""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def get_session_messages(session_id: str) -> list[dict]:
    """Return all messages for a chat session, oldest first."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT role, content, created_at FROM chat_messages
           WHERE session_id = ? ORDER BY id ASC""",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_session(session_id: str) -> bool:
    """Delete a chat session (and its messages via cascade). Returns True if deleted."""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def create_knowledge_gap(
    question: str, context: str = "", session_id: str | None = None
) -> dict:
    """Record a new open knowledge gap. Returns the created row."""
    gap_id = f"gap-{uuid.uuid4().hex[:8]}"
    conn = get_connection()
    conn.execute(
        """INSERT INTO knowledge_gaps (id, question, context, session_id, status)
           VALUES (?, ?, ?, ?, 'open')""",
        (gap_id, question, context, session_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    conn.close()
    return dict(row)


def get_knowledge_gaps(
    page: int = 1, per_page: int = 20, status: str | None = None
) -> tuple[list[dict], int]:
    """Return a page of knowledge gaps (optionally filtered by status) and the total count."""
    conn = get_connection()
    where = "WHERE status = ?" if status else ""
    params: tuple = (status,) if status else ()
    total = conn.execute(
        f"SELECT COUNT(*) FROM knowledge_gaps {where}", params
    ).fetchone()[0]
    offset = (page - 1) * per_page
    rows = conn.execute(
        f"SELECT * FROM knowledge_gaps {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (*params, per_page, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows], total


def get_knowledge_gap(gap_id: str) -> dict | None:
    """Get a single knowledge gap by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_knowledge_gap_text(gap_id: str, question: str, context: str) -> dict | None:
    """Rephrase the question / edit the context of a gap. Returns the updated row or None."""
    conn = get_connection()
    existing = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    if not existing:
        conn.close()
        return None
    conn.execute(
        """UPDATE knowledge_gaps SET question=?, context=?, updated_at=datetime('now')
           WHERE id=?""",
        (question, context, gap_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    conn.close()
    return dict(row)


def resolve_knowledge_gap(gap_id: str, entry_id: str) -> dict | None:
    """Mark a gap resolved and link it to the newly created entry. Returns the updated row or None."""
    conn = get_connection()
    existing = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    if not existing:
        conn.close()
        return None
    conn.execute(
        """UPDATE knowledge_gaps SET status='resolved', resolved_entry_id=?, updated_at=datetime('now')
           WHERE id=?""",
        (entry_id, gap_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    conn.close()
    return dict(row)


def dismiss_knowledge_gap(gap_id: str) -> dict | None:
    """Mark a gap dismissed. Returns the updated row or None if not found."""
    conn = get_connection()
    existing = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    if not existing:
        conn.close()
        return None
    conn.execute(
        "UPDATE knowledge_gaps SET status='dismissed', updated_at=datetime('now') WHERE id=?",
        (gap_id,),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM knowledge_gaps WHERE id = ?", (gap_id,)).fetchone()
    conn.close()
    return dict(row)
