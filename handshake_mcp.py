"""
handshake_mcp.py
================
Context synchronization between AI agents.
Like a kiss — a brief, meaningful exchange. Two agents share what they
know and what they're confused about, then split back off independently.
Prevents duplication, resolves drift, coordinates without a heavy bus.

Usage:
    python handshake_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "handshake": {
      "command": "python",
      "args": ["/path/to/handshake_mcp.py"]
    }
  }
}

Pattern:
    # Agent A posts its state
    token = handshake.offer(agent_id="agent_a", context="...", open_questions=["..."])

    # Agent B picks it up and responds
    result = handshake.accept(token=token, agent_id="agent_b", context="...", response_to_questions=["..."])

    # Both agents pull the merged sync
    sync = handshake.pull(token=token, agent_id="agent_a")
"""

import json
import time
import sqlite3
import threading
import hashlib
from pathlib import Path
from typing import Optional
import httpx
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH    = Path.home() / ".handshake.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 500
TOKEN_TTL  = 3600  # handshake tokens expire after 1 hour

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS handshakes (
                token       TEXT PRIMARY KEY,
                ts          REAL NOT NULL,
                expires_at  REAL NOT NULL,
                agent_a     TEXT NOT NULL,
                context_a   TEXT NOT NULL,
                questions_a TEXT DEFAULT '[]',
                agent_b     TEXT,
                context_b   TEXT,
                questions_b TEXT DEFAULT '[]',
                responses_b TEXT DEFAULT '[]',
                merged      TEXT,
                status      TEXT DEFAULT 'open'
            );
        """)
        _local.conn.commit()
    return _local.conn

def _purge(db):
    db.execute("DELETE FROM handshakes WHERE expires_at < ?", (time.time(),))
    db.commit()

def _make_token(agent_id: str) -> str:
    raw = f"{agent_id}{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

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
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="handshake",
    instructions=(
        "Context synchronization between agents. One agent offers, another accepts. "
        "Both pull a merged sync that resolves conflicts and surfaces shared unknowns. "
        "Fast, clean, non-blocking."
    )
)

@mcp.tool()
def offer(
    agent_id: str,
    context: str,
    open_questions: list[str] = [],
    goal: str = ""
) -> dict:
    """
    Initiate a handshake. Post your current context and open questions.
    Returns a token that the other agent uses to accept.

    Args:
        agent_id:       Your identifier.
        context:        Your current state / what you know.
        open_questions: Things you're uncertain about that the other agent might answer.
        goal:           Optional — what you're trying to accomplish.

    Returns:
        { token, expires_at, status: "waiting_for_partner" }
    """
    if not agent_id or not context:
        return {"error": "agent_id and context required"}

    db    = get_db()
    _purge(db)
    token = _make_token(agent_id)
    now   = time.time()

    db.execute(
        """INSERT INTO handshakes
           (token, ts, expires_at, agent_a, context_a, questions_a, status)
           VALUES (?,?,?,?,?,?,?)""",
        (token, now, now + TOKEN_TTL, agent_id, context[:2000],
         json.dumps(open_questions), "open")
    )
    db.commit()

    return {
        "token":       token,
        "expires_at":  int(now + TOKEN_TTL),
        "status":      "waiting_for_partner",
        "share_token": f"Share this token with the other agent: {token}"
    }


@mcp.tool()
def accept(
    token: str,
    agent_id: str,
    context: str,
    open_questions: list[str] = [],
    response_to_questions: list[str] = []
) -> dict:
    """
    Accept a handshake offer. Post your context and answers.
    This triggers the merge — both agents can now pull the sync.

    Args:
        token:                 The token from the offering agent.
        agent_id:              Your identifier.
        context:               Your current state / what you know.
        open_questions:        Your own open questions.
        response_to_questions: Answers to the offering agent's questions (in order).

    Returns:
        { token, merged_summary, status: "complete" }
    """
    db = get_db()
    _purge(db)

    row = db.execute(
        "SELECT * FROM handshakes WHERE token=? AND status='open'", (token,)
    ).fetchone()

    if not row:
        return {"error": f"Token '{token}' not found, expired, or already used."}

    if row["agent_b"]:
        return {"error": "Handshake already accepted by another agent."}

    # Build the merge prompt
    q_a = json.loads(row["questions_a"])
    q_a_str = "\n".join(f"- {q}" for q in q_a) if q_a else "None"
    q_b_str = "\n".join(f"- {q}" for q in open_questions) if open_questions else "None"
    r_b_str = "\n".join(f"- {r}" for r in response_to_questions) if response_to_questions else "None"

    prompt = f"""Two AI agents are doing a context handshake. Synthesize their shared state.

Agent A ({row['agent_a']}) knows:
{row['context_a'][:800]}
Agent A's open questions: {q_a_str}

Agent B ({agent_id}) knows:
{context[:800]}
Agent B's open questions: {q_b_str}
Agent B's answers to A's questions: {r_b_str}

Synthesize a merged context block that:
1. Combines what both agents know without duplication
2. Resolves any contradictions (note them if unresolvable)
3. Lists remaining open questions neither can answer
4. States what each agent should do differently based on this sync

Respond ONLY with valid JSON:
{{
  "merged_context": <string, clean synthesis of shared knowledge, max 200 words>,
  "contradictions": [<any conflicts found>],
  "remaining_questions": [<questions neither agent could answer>],
  "for_agent_a": <string, one sentence — what A should update based on this>,
  "for_agent_b": <string, one sentence — what B should update based on this>
}}"""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        merged = json.loads(raw_clean)
    except Exception as e:
        merged = {
            "merged_context":      f"Merge failed: {e}. Raw context from both agents available.",
            "contradictions":      [],
            "remaining_questions": q_a + list(open_questions),
            "for_agent_a":         "Review Agent B's context manually.",
            "for_agent_b":         "Review Agent A's context manually."
        }

    db.execute(
        """UPDATE handshakes SET
           agent_b=?, context_b=?, questions_b=?, responses_b=?, merged=?, status='complete'
           WHERE token=?""",
        (agent_id, context[:2000], json.dumps(open_questions),
         json.dumps(response_to_questions), json.dumps(merged), token)
    )
    db.commit()

    return {
        "token":              token,
        "status":             "complete",
        "merged_context":     merged.get("merged_context", ""),
        "contradictions":     merged.get("contradictions", []),
        "remaining_questions":merged.get("remaining_questions", []),
        "your_update":        merged.get("for_agent_b", "")
    }


@mcp.tool()
def pull(token: str, agent_id: str) -> dict:
    """
    Pull the merged sync result after a handshake completes.
    Call this after offer() once the other agent has accepted.

    Args:
        token:    The handshake token.
        agent_id: Your agent ID (returns your specific update).

    Returns:
        { merged_context, your_update, contradictions, remaining_questions }
    """
    db  = get_db()
    row = db.execute(
        "SELECT * FROM handshakes WHERE token=?", (token,)
    ).fetchone()

    if not row:
        return {"error": f"Token '{token}' not found or expired."}

    if row["status"] == "open":
        return {"status": "waiting", "message": "Partner hasn't accepted yet."}

    merged = json.loads(row["merged"]) if row["merged"] else {}

    is_a = agent_id == row["agent_a"]
    your_update = merged.get("for_agent_a" if is_a else "for_agent_b", "")

    return {
        "status":               "complete",
        "merged_context":       merged.get("merged_context", ""),
        "your_update":          your_update,
        "contradictions":       merged.get("contradictions", []),
        "remaining_questions":  merged.get("remaining_questions", []),
        "partner_agent":        row["agent_b"] if is_a else row["agent_a"]
    }


@mcp.tool()
def broadcast(
    agent_id: str,
    context: str,
    to_agents: list[str]
) -> dict:
    """
    One-to-many context broadcast. No merge — just pushes your context
    to a named list of agents. They can retrieve it by agent_id.
    Good for a coordinator agent updating its workers.

    Args:
        agent_id:  Your identifier (the sender).
        context:   What you want to share.
        to_agents: List of agent IDs to notify.

    Returns:
        { tokens: {agent_id: token}, broadcast_id }
    """
    db     = get_db()
    tokens = {}
    now    = time.time()

    for target in to_agents:
        token = _make_token(f"{agent_id}→{target}")
        db.execute(
            """INSERT INTO handshakes
               (token, ts, expires_at, agent_a, context_a, questions_a, status)
               VALUES (?,?,?,?,?,?,?)""",
            (token, now, now + TOKEN_TTL, agent_id, context[:2000], "[]", "broadcast")
        )
        tokens[target] = token

    db.commit()

    return {
        "tokens":       tokens,
        "from":         agent_id,
        "to_agents":    to_agents,
        "instructions": "Each target agent calls pull(token, agent_id) to receive."
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("handshake MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
