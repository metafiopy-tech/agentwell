"""
sleep_mcp.py
============
Memory consolidation for AI agents.
Like sleep — compresses episodic run logs into semantic summaries,
discards noise, and surfaces durable facts the agent can wake up with.

No agent natively does this. Most just accumulate noise across runs
until context bloats or they start contradicting themselves. This
fixes that with a scheduled consolidation pass.

Usage:
    python sleep_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "sleep": {
      "command": "python",
      "args": ["/path/to/sleep_mcp.py"]
    }
  }
}

Typical pattern:
    # During run — log episodes
    sleep.log_episode(run_id="r1", content="tried X, got Y, decided Z")

    # After run — consolidate
    result = sleep.consolidate(run_id="r1")
    # result["semantic_memory"] = clean facts to carry forward

    # Next run — wake up
    context = sleep.wake(tags=["user_prefs", "domain"])
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

DB_PATH    = Path.home() / ".sleep_memory.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 800

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                run_id     TEXT NOT NULL,
                content    TEXT NOT NULL,
                importance INTEGER DEFAULT 5,
                consolidated INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                run_id     TEXT,
                tags       TEXT DEFAULT '',
                content    TEXT NOT NULL,
                source_ids TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_run ON episodes(run_id);
            CREATE INDEX IF NOT EXISTS idx_tags ON semantic_memory(tags);
        """)
        _local.conn.commit()
    return _local.conn

# ── LLM ──────────────────────────────────────────────────────────────────────

def _call_claude(prompt: str) -> str:
    resp = httpx.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=45
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="sleep",
    instructions=(
        "Memory consolidation for AI agents. Log episodes during a run, "
        "consolidate after, wake up clean. Compresses noisy episodic logs "
        "into durable semantic facts. Prevents noise accumulation across runs."
    )
)

@mcp.tool()
def log_episode(
    content: str,
    run_id: str,
    importance: int = 5
) -> dict:
    """
    Log a single episode (observation, decision, result) during a run.
    These accumulate and get consolidated during sleep().

    Args:
        content:    What happened — a decision made, something learned,
                    a result observed. Be specific. 1-3 sentences.
        run_id:     Tag for this run. Use a consistent ID per session.
        importance: 1-10. High importance = more likely to survive consolidation.

    Returns:
        { episode_id, run_id, logged }
    """
    if not content or not run_id:
        return {"error": "content and run_id required"}

    db  = get_db()
    cur = db.execute(
        "INSERT INTO episodes (ts, run_id, content, importance) VALUES (?,?,?,?)",
        (time.time(), run_id, content, max(1, min(10, importance)))
    )
    db.commit()
    return {"episode_id": cur.lastrowid, "run_id": run_id, "logged": True}


@mcp.tool()
def consolidate(
    run_id: str,
    tags: str = "",
    discard_threshold: int = 3
) -> dict:
    """
    Consolidate all logged episodes from a run into semantic memory.
    This is the sleep pass — compresses noisy logs into clean facts,
    discards low-importance noise, surfaces durable knowledge.

    Call this at the end of a run or when context is getting heavy.

    Args:
        run_id:            The run to consolidate.
        tags:              Comma-separated tags for the resulting memory
                           e.g. "user_prefs,domain_facts,errors"
        discard_threshold: Episodes with importance below this get
                           excluded from consolidation. Default 3.

    Returns:
        {
          semantic_memory: str (the consolidated knowledge),
          episodes_processed: int,
          episodes_discarded: int,
          memory_id: int
        }
    """
    db = get_db()

    episodes = db.execute(
        "SELECT * FROM episodes WHERE run_id=? AND consolidated=0 ORDER BY importance DESC, ts ASC",
        (run_id,)
    ).fetchall()

    if not episodes:
        return {"error": f"No unconsolidated episodes found for run_id '{run_id}'"}

    kept      = [e for e in episodes if e["importance"] >= discard_threshold]
    discarded = len(episodes) - len(kept)

    if not kept:
        return {
            "error":              "All episodes below discard threshold",
            "episodes_discarded": discarded
        }

    episode_text = "\n".join(
        f"[importance:{e['importance']}] {e['content']}"
        for e in kept
    )

    prompt = f"""You are consolidating an AI agent's episodic memory into semantic memory.
These are raw episode logs from run '{run_id}':

{episode_text}

Extract the durable, reusable knowledge from these episodes.
Respond ONLY with valid JSON:
{{
  "semantic_memory": <string: clean, concise paragraph of facts/learnings
    that should persist. Exclude noise, transient observations, and anything
    run-specific that won't matter next time. Write in third person as stable facts.
    Max 200 words.>,
  "key_facts": [<list of 3-8 atomic facts, one sentence each>],
  "suggested_tags": [<list of 2-5 relevant tags for retrieval>]
}}"""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        return {"error": f"Consolidation failed: {e}"}

    semantic = result.get("semantic_memory", "")
    key_facts = result.get("key_facts", [])
    auto_tags = result.get("suggested_tags", [])

    final_tags = tags if tags else ",".join(auto_tags)
    source_ids = ",".join(str(e["id"]) for e in kept)

    cur = db.execute(
        "INSERT INTO semantic_memory (ts, run_id, tags, content, source_ids) VALUES (?,?,?,?,?)",
        (time.time(), run_id, final_tags, semantic, source_ids)
    )
    mem_id = cur.lastrowid

    db.execute(
        "UPDATE episodes SET consolidated=1 WHERE run_id=? AND id IN (%s)"
        % ",".join("?" * len(kept)),
        [run_id] + [e["id"] for e in kept]
    )
    db.commit()

    return {
        "semantic_memory":    semantic,
        "key_facts":          key_facts,
        "memory_id":          mem_id,
        "tags":               final_tags,
        "episodes_processed": len(kept),
        "episodes_discarded": discarded
    }


@mcp.tool()
def wake(
    tags: str = "",
    query: str = "",
    limit: int = 5
) -> dict:
    """
    Load relevant semantic memories at the start of a run.
    This is waking up — start with the clean compressed knowledge,
    not raw noisy logs.

    Args:
        tags:  Comma-separated tags to filter by e.g. "user_prefs,domain"
        query: Keyword search across memory content.
        limit: Max memories to return. Default 5.

    Returns:
        { memories: [...], context_block: str }
    """
    db   = get_db()
    rows = []

    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        for tag in tag_list:
            r = db.execute(
                "SELECT * FROM semantic_memory WHERE tags LIKE ? ORDER BY ts DESC LIMIT ?",
                (f"%{tag}%", limit)
            ).fetchall()
            rows.extend(r)
        seen = set()
        rows = [r for r in rows if not (r["id"] in seen or seen.add(r["id"]))]
    elif query:
        rows = db.execute(
            "SELECT * FROM semantic_memory WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM semantic_memory ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()

    memories = [
        {
            "id":      r["id"],
            "ts":      int(r["ts"]),
            "run_id":  r["run_id"],
            "tags":    r["tags"],
            "content": r["content"]
        }
        for r in rows[:limit]
    ]

    context_block = ""
    if memories:
        context_block = "Relevant memory from prior runs:\n" + "\n---\n".join(
            m["content"] for m in memories
        )

    return {
        "memories":      memories,
        "context_block": context_block,
        "count":         len(memories)
    }


@mcp.tool()
def memory_stats() -> dict:
    """
    Overview of what's in memory — episode counts, consolidation
    status, tag distribution. Good for a quick health check.
    """
    db = get_db()

    ep_total  = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    ep_pending = db.execute("SELECT COUNT(*) FROM episodes WHERE consolidated=0").fetchone()[0]
    sem_total  = db.execute("SELECT COUNT(*) FROM semantic_memory").fetchone()[0]

    tag_rows = db.execute("SELECT tags FROM semantic_memory").fetchall()
    all_tags: dict = {}
    for row in tag_rows:
        for t in row["tags"].split(","):
            t = t.strip()
            if t:
                all_tags[t] = all_tags.get(t, 0) + 1

    return {
        "total_episodes":           ep_total,
        "unconsolidated_episodes":  ep_pending,
        "semantic_memories":        sem_total,
        "tag_distribution":         all_tags
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("sleep MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
