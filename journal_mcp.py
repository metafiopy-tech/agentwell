"""
journal_mcp.py
==============
Structured episodic run journal for AI agents.
Like keeping a diary — logs decisions, surprises, and reasoning chains
per run with enough structure to replay, search, and learn from.

The key difference from raw logging: every entry captures not just
WHAT happened but WHY the agent decided it, and WHAT it was surprised by.
That's what makes it replayable and learnable.

Usage:
    python journal_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "journal": {
      "command": "python",
      "args": ["/path/to/journal_mcp.py"]
    }
  }
}

Pattern:
    # Start a run
    journal.open_run(run_id="r1", goal="...")

    # Log entries throughout
    journal.entry(run_id="r1", type="decision", content="...", reasoning="...", surprise_level=2)

    # Close the run with a summary
    journal.close_run(run_id="r1", outcome="...", lessons=[])

    # Next run — check for relevant prior lessons
    prior = journal.recall(query="similar task", limit=3)
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional
import httpx
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH    = Path.home() / ".journal.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 500

ENTRY_TYPES = ["decision", "observation", "error", "surprise", "milestone", "hypothesis", "correction"]

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id   TEXT PRIMARY KEY,
                ts_open  REAL NOT NULL,
                ts_close REAL,
                goal     TEXT,
                outcome  TEXT,
                lessons  TEXT DEFAULT '[]',
                status   TEXT DEFAULT 'open'
            );
            CREATE TABLE IF NOT EXISTS entries (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                ts             REAL NOT NULL,
                run_id         TEXT NOT NULL,
                type           TEXT NOT NULL,
                content        TEXT NOT NULL,
                reasoning      TEXT DEFAULT '',
                surprise_level INTEGER DEFAULT 0,
                step           INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_run_entries ON entries(run_id);
            CREATE INDEX IF NOT EXISTS idx_type ON entries(type);
        """)
        _local.conn.commit()
    return _local.conn

def _call_claude(prompt: str) -> str:
    resp = httpx.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="journal",
    instructions=(
        "Structured run journal for AI agents. Log decisions, surprises, "
        "and reasoning chains. Replay past runs. Surface lessons before "
        "starting similar tasks. Breaks the cycle of repeating mistakes."
    )
)

@mcp.tool()
def open_run(run_id: str, goal: str = "") -> dict:
    """
    Open a new journal run. Call at the start of each agent session.

    Args:
        run_id: Unique identifier for this run.
        goal:   What you're trying to accomplish.

    Returns:
        { run_id, status: "open" }
    """
    db = get_db()
    existing = db.execute("SELECT run_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if existing:
        return {"run_id": run_id, "status": "already_open", "message": "Run already exists"}

    db.execute(
        "INSERT INTO runs (run_id, ts_open, goal, status) VALUES (?,?,?,?)",
        (run_id, time.time(), goal, "open")
    )
    db.commit()
    return {"run_id": run_id, "status": "open", "goal": goal}


@mcp.tool()
def entry(
    run_id: str,
    content: str,
    type: str = "observation",
    reasoning: str = "",
    surprise_level: int = 0,
    step: int = 0
) -> dict:
    """
    Log a journal entry during a run.

    Args:
        run_id:         The active run ID.
        content:        What happened / what was decided / what was observed.
        type:           Entry type: decision | observation | error | surprise |
                        milestone | hypothesis | correction
        reasoning:      Why this decision was made or why this was surprising.
        surprise_level: 0-10. High = something unexpected happened. Surprises
                        get extra weight in retrospectives.
        step:           Optional step number for ordering.

    Returns:
        { entry_id, run_id, logged }
    """
    if type not in ENTRY_TYPES:
        type = "observation"

    db  = get_db()
    cur = db.execute(
        "INSERT INTO entries (ts, run_id, type, content, reasoning, surprise_level, step) VALUES (?,?,?,?,?,?,?)",
        (time.time(), run_id, type, content, reasoning,
         max(0, min(10, surprise_level)), step)
    )
    db.commit()
    return {"entry_id": cur.lastrowid, "run_id": run_id, "type": type, "logged": True}


@mcp.tool()
def close_run(
    run_id: str,
    outcome: str = "",
    lessons: list[str] = [],
    auto_extract: bool = True
) -> dict:
    """
    Close a run and record the outcome.
    If auto_extract=True, uses Claude to extract lessons from the entries.

    Args:
        run_id:       The run to close.
        outcome:      What happened — success, failure, partial, redirected.
        lessons:      Manual lessons learned. Auto-extracted if auto_extract=True.
        auto_extract: Automatically extract lessons from entries. Default True.

    Returns:
        { run_id, lessons, entry_count, status: "closed" }
    """
    db = get_db()

    run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not run:
        return {"error": f"Run '{run_id}' not found"}

    entries = db.execute(
        "SELECT type, content, reasoning, surprise_level FROM entries WHERE run_id=? ORDER BY ts",
        (run_id,)
    ).fetchall()

    final_lessons = list(lessons)

    if auto_extract and entries:
        entry_text = "\n".join(
            f"[{e['type']}|surprise:{e['surprise_level']}] {e['content']}"
            + (f" (reasoning: {e['reasoning']})" if e['reasoning'] else "")
            for e in entries
        )

        prompt = f"""Extract reusable lessons from this AI agent run journal.
Goal: {run['goal'] or 'unspecified'}
Outcome: {outcome or 'unspecified'}

Entries:
{entry_text[:1500]}

Respond ONLY with valid JSON:
{{
  "lessons": [<list of 2-5 concise, actionable lessons that apply to future similar runs. One sentence each.>],
  "patterns": [<recurring themes or patterns noticed>],
  "do_differently": [<specific things to change next time>]
}}"""

        try:
            raw = _call_claude(prompt)
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
            extracted = json.loads(raw_clean)
            final_lessons = extracted.get("lessons", []) + list(lessons)
        except:
            pass

    db.execute(
        "UPDATE runs SET ts_close=?, outcome=?, lessons=?, status='closed' WHERE run_id=?",
        (time.time(), outcome, json.dumps(final_lessons), run_id)
    )
    db.commit()

    return {
        "run_id":      run_id,
        "status":      "closed",
        "outcome":     outcome,
        "lessons":     final_lessons,
        "entry_count": len(entries)
    }


@mcp.tool()
def recall(
    query: str = "",
    type_filter: str = "",
    min_surprise: int = 0,
    limit: int = 10
) -> dict:
    """
    Search journal history. Use at the start of a run to pull
    relevant prior experiences before starting.

    Args:
        query:        Keyword to search entries and run goals.
        type_filter:  Filter by entry type e.g. "error" or "surprise"
        min_surprise: Only return entries with surprise_level >= this.
        limit:        Max results. Default 10.

    Returns:
        { entries: [...], lessons_from_matching_runs: [...] }
    """
    db = get_db()

    conditions = []
    params     = []

    if query:
        conditions.append("(e.content LIKE ? OR e.reasoning LIKE ? OR r.goal LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%", f"%{query}%"])

    if type_filter:
        conditions.append("e.type = ?")
        params.append(type_filter)

    if min_surprise > 0:
        conditions.append("e.surprise_level >= ?")
        params.append(min_surprise)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)

    rows = db.execute(
        f"""SELECT e.*, r.goal, r.outcome
            FROM entries e
            LEFT JOIN runs r ON e.run_id = r.run_id
            {where}
            ORDER BY e.surprise_level DESC, e.ts DESC
            LIMIT ?""",
        params
    ).fetchall()

    entries = [
        {
            "run_id":        r["run_id"],
            "type":          r["type"],
            "content":       r["content"],
            "reasoning":     r["reasoning"],
            "surprise_level":r["surprise_level"],
            "goal":          r["goal"]
        }
        for r in rows
    ]

    # Pull lessons from runs that appeared in results
    run_ids = list({e["run_id"] for e in entries})
    lessons = []
    if run_ids:
        lesson_rows = db.execute(
            f"SELECT run_id, lessons FROM runs WHERE run_id IN ({','.join('?'*len(run_ids))})",
            run_ids
        ).fetchall()
        for lr in lesson_rows:
            if lr["lessons"]:
                parsed = json.loads(lr["lessons"])
                lessons.extend(parsed)

    return {
        "entries":                  entries,
        "lessons_from_matching_runs": list(set(lessons)),
        "count":                    len(entries)
    }


@mcp.tool()
def replay(run_id: str) -> dict:
    """
    Full replay of a run — every entry in order with its reasoning.
    Good for post-mortems or understanding what a past run actually did.

    Args:
        run_id: The run to replay.

    Returns:
        { run, entries, timeline }
    """
    db = get_db()

    run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not run:
        return {"error": f"Run '{run_id}' not found"}

    entries = db.execute(
        "SELECT * FROM entries WHERE run_id=? ORDER BY ts ASC",
        (run_id,)
    ).fetchall()

    return {
        "run": {
            "run_id":  run["run_id"],
            "goal":    run["goal"],
            "outcome": run["outcome"],
            "lessons": json.loads(run["lessons"]) if run["lessons"] else [],
            "status":  run["status"]
        },
        "entries": [
            {
                "step":          e["step"],
                "type":          e["type"],
                "content":       e["content"],
                "reasoning":     e["reasoning"],
                "surprise_level":e["surprise_level"],
                "ts":            int(e["ts"])
            }
            for e in entries
        ],
        "entry_count": len(entries),
        "surprises":   sum(1 for e in entries if e["surprise_level"] >= 5)
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("journal MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
