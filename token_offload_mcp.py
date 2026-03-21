"""
token_offload_mcp.py
====================
A lightweight MCP server for mid-run context offloading.
Agents dump heavy context here, get back a key, retrieve when needed.
Like RAM paging but for context windows.

Usage:
    python token_offload_mcp.py

Then point your agent's MCP config at it.
"""

import json
import time
import uuid
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH     = Path.home() / ".token_offload.db"
DEFAULT_TTL = 3600   # seconds — chunks auto-expire after 1 hour
MAX_CONTENT = 50_000 # characters — hard cap per chunk

# ── DB setup ──────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                key       TEXT PRIMARY KEY,
                content   TEXT NOT NULL,
                tags      TEXT DEFAULT '',
                stored_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hits      INTEGER DEFAULT 0
            )
        """)
        _local.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_expires ON chunks(expires_at)"
        )
        _local.conn.commit()
    return _local.conn

def _purge_expired(db):
    db.execute("DELETE FROM chunks WHERE expires_at < ?", (time.time(),))
    db.commit()

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="token-offload",
    instructions=(
        "Mid-run context offloading for AI agents. "
        "Use store() to park heavy context, retrieve() to pull it back, "
        "search() to find relevant chunks by keyword, forget() to drop one early. "
        "All chunks auto-expire after 1 hour unless ttl is specified."
    )
)

@mcp.tool()
def store(
    content: str,
    tags: str = "",
    ttl: int = DEFAULT_TTL
) -> dict:
    """
    Dump a chunk of context into offload storage.
    Returns a key you can use to retrieve it later.

    Args:
        content: The text to offload. Can be reasoning chains, tool results,
                 background info, anything bloating your context window.
        tags:    Comma-separated labels e.g. "user_data,step_3,background"
        ttl:     Seconds until auto-expiry. Default 3600 (1 hour).

    Returns:
        { key, expires_at, size_chars }
    """
    if len(content) > MAX_CONTENT:
        return {
            "error": f"Content too large ({len(content)} chars). Max is {MAX_CONTENT}."
        }

    key        = str(uuid.uuid4())[:8]
    now        = time.time()
    expires_at = now + ttl
    db         = get_db()
    _purge_expired(db)

    db.execute(
        "INSERT INTO chunks (key, content, tags, stored_at, expires_at) VALUES (?,?,?,?,?)",
        (key, content, tags, now, expires_at)
    )
    db.commit()

    return {
        "key":        key,
        "expires_at": int(expires_at),
        "size_chars": len(content),
        "tags":       tags
    }


@mcp.tool()
def retrieve(key: str) -> dict:
    """
    Pull a stored chunk back into context by its key.

    Args:
        key: The key returned by store().

    Returns:
        { key, content, tags, stored_at, hits }
        or { error } if key is missing or expired.
    """
    db  = get_db()
    _purge_expired(db)
    row = db.execute(
        "SELECT * FROM chunks WHERE key = ?", (key,)
    ).fetchone()

    if not row:
        return {"error": f"Key '{key}' not found or expired."}

    db.execute("UPDATE chunks SET hits = hits + 1 WHERE key = ?", (key,))
    db.commit()

    return {
        "key":       row["key"],
        "content":   row["content"],
        "tags":      row["tags"],
        "stored_at": int(row["stored_at"]),
        "hits":      row["hits"] + 1
    }


@mcp.tool()
def search(query: str, limit: int = 5) -> dict:
    """
    Find stored chunks that contain the query string.
    Simple substring match — fast, no embeddings needed.

    Args:
        query: Keyword or phrase to search for.
        limit: Max results to return. Default 5.

    Returns:
        { results: [ { key, tags, preview, stored_at } ] }
    """
    db = get_db()
    _purge_expired(db)

    rows = db.execute(
        """
        SELECT key, content, tags, stored_at
        FROM chunks
        WHERE content LIKE ? OR tags LIKE ?
        ORDER BY stored_at DESC
        LIMIT ?
        """,
        (f"%{query}%", f"%{query}%", limit)
    ).fetchall()

    return {
        "results": [
            {
                "key":       r["key"],
                "tags":      r["tags"],
                "preview":   r["content"][:200] + ("..." if len(r["content"]) > 200 else ""),
                "stored_at": int(r["stored_at"])
            }
            for r in rows
        ],
        "count": len(rows)
    }


@mcp.tool()
def forget(key: str) -> dict:
    """
    Explicitly delete a chunk before it expires.
    Use when you know you're done with something and want to keep storage clean.

    Args:
        key: The key to delete.

    Returns:
        { deleted: true/false }
    """
    db      = get_db()
    cursor  = db.execute("DELETE FROM chunks WHERE key = ?", (key,))
    db.commit()
    return {"deleted": cursor.rowcount > 0, "key": key}


@mcp.tool()
def status() -> dict:
    """
    Check what's currently in offload storage.
    Good for an agent to call at the start of a run to see if there's
    anything useful left over from a previous step.

    Returns:
        { active_chunks, total_size_chars, oldest, newest, all_tags }
    """
    db = get_db()
    _purge_expired(db)

    rows = db.execute(
        "SELECT key, tags, length(content) as sz, stored_at FROM chunks ORDER BY stored_at"
    ).fetchall()

    if not rows:
        return {"active_chunks": 0, "total_size_chars": 0, "all_tags": []}

    all_tags = set()
    for r in rows:
        for t in r["tags"].split(","):
            if t.strip():
                all_tags.add(t.strip())

    return {
        "active_chunks":   len(rows),
        "total_size_chars": sum(r["sz"] for r in rows),
        "oldest":          int(rows[0]["stored_at"]),
        "newest":          int(rows[-1]["stored_at"]),
        "all_tags":        sorted(all_tags),
        "keys":            [r["key"] for r in rows]
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("token-offload MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
