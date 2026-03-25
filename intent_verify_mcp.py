"""
intent_verify_mcp.py
====================
Verify that an agent's proposed action matches its original intent.
The final check before anything irreversible happens.

Most agent failures aren't from bad reasoning — they're from goal drift
that accumulates invisibly across steps until the agent does something
that was never intended.

This runs at the action boundary: before a file is written,
before a command is executed, before a message is sent.
One question: does this action still serve the original intent?

Usage:
    python intent_verify_mcp.py
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
import httpx
from fastmcp import FastMCP

DB_PATH    = Path.home() / ".intent_verify.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 400

IRREVERSIBLE_ACTIONS = [
    "delete", "remove", "drop", "truncate", "overwrite",
    "send", "post", "publish", "deploy", "commit", "push",
    "execute", "run", "format", "wipe", "uninstall"
]

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS verifications (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              REAL NOT NULL,
                agent_id        TEXT,
                run_id          TEXT,
                original_intent TEXT,
                proposed_action TEXT,
                aligned         INTEGER,
                drift_score     REAL,
                verdict         TEXT,
                blocked         INTEGER DEFAULT 0
            );
        """)
        _local.conn.commit()
    return _local.conn

def _call_claude(system, prompt):
    resp = httpx.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()

mcp = FastMCP(
    name="intent_verify",
    instructions=(
        "Final intent check before irreversible agent actions. "
        "Verifies the proposed action still serves the original intent. "
        "Catches goal drift before it executes. "
        "Block or proceed with confidence."
    )
)

@mcp.tool()
def verify(
    original_intent: str,
    proposed_action: str,
    reasoning_chain: str = "",
    agent_id: str = "",
    run_id: str = "",
    auto_block_drift: bool = True
) -> dict:
    """
    Verify a proposed action aligns with original intent.
    Call before any irreversible action.

    Args:
        original_intent:   The goal the agent was given at the start
        proposed_action:   The specific action about to be taken
        reasoning_chain:   The steps the agent took to arrive here (optional)
        agent_id:          Agent identifier
        run_id:            Run identifier
        auto_block_drift:  If True, returns blocked=True when drift is high

    Returns:
        {
          aligned: bool,
          drift_score: float 0-1,
          verdict: "proceed" | "warn" | "block",
          reasoning: str,
          is_irreversible: bool,
          blocked: bool
        }
    """
    # Fast irreversibility check
    action_lower = proposed_action.lower()
    is_irreversible = any(word in action_lower for word in IRREVERSIBLE_ACTIONS)

    reasoning_section = f"\nReasoning chain:\n{reasoning_chain[:600]}" if reasoning_chain else ""

    result = _call_claude(
        """You are an intent verification system. One job: does this action still serve the original intent?

Agents drift. They start with one goal and end up doing something adjacent, related,
but not what was asked. Catch this before it executes.

Be binary: aligned or not aligned. Then give a drift score.
Respond ONLY with valid JSON.""",
        f"""Original intent: {original_intent}

Proposed action: {proposed_action}
{reasoning_section}
Is irreversible: {is_irreversible}

{{
  "aligned": <bool: does this action clearly serve the original intent?>,
  "drift_score": <float 0-1: 0=perfectly aligned, 1=completely off-target>,
  "verdict": <"proceed"|"warn"|"block">,
  "reasoning": <string, 1-2 sentences explaining the verdict>,
  "what_drifted": <string, if drift_score > 0.3, what specifically drifted? Otherwise null>
}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(clean)
    except Exception:
        # Conservative fallback — warn on parse failure
        data = {
            "aligned": not is_irreversible,
            "drift_score": 0.5 if is_irreversible else 0.2,
            "verdict": "warn",
            "reasoning": "Could not fully evaluate — proceed with caution.",
            "what_drifted": None
        }

    drift_score = float(data.get("drift_score", 0.3))
    verdict     = data.get("verdict", "warn")
    aligned     = data.get("aligned", True)

    # Auto-block logic
    blocked = False
    if auto_block_drift:
        if verdict == "block":
            blocked = True
        elif is_irreversible and drift_score > 0.4:
            blocked = True
            verdict = "block"

    db = get_db()
    db.execute(
        "INSERT INTO verifications (ts, agent_id, run_id, original_intent, proposed_action, aligned, drift_score, verdict, blocked) VALUES (?,?,?,?,?,?,?,?,?)",
        (time.time(), agent_id, run_id, original_intent[:300], proposed_action[:300],
         1 if aligned else 0, drift_score, verdict, 1 if blocked else 0)
    )
    db.commit()

    return {
        "aligned": aligned,
        "drift_score": round(drift_score, 3),
        "verdict": verdict,
        "reasoning": data.get("reasoning", ""),
        "what_drifted": data.get("what_drifted"),
        "is_irreversible": is_irreversible,
        "blocked": blocked,
        "safe_to_proceed": not blocked
    }


@mcp.tool()
def quick_check(
    original_intent: str,
    proposed_action: str
) -> dict:
    """
    Fast keyword-based intent check. No API call.
    Use as a pre-filter before the full verify().

    Args:
        original_intent:  The original goal
        proposed_action:  The proposed action

    Returns:
        { likely_aligned, irreversible, recommend_full_verify }
    """
    intent_words = set(original_intent.lower().split())
    action_words = set(proposed_action.lower().split())

    # Remove stop words
    stop = {"the","a","an","is","in","it","to","of","and","or","for","with","on","at","by"}
    intent_words -= stop
    action_words -= stop

    overlap = len(intent_words & action_words)
    total   = len(intent_words) if intent_words else 1
    alignment = overlap / total

    is_irreversible = any(w in proposed_action.lower() for w in IRREVERSIBLE_ACTIONS)
    likely_aligned  = alignment > 0.2 and not (alignment < 0.05 and is_irreversible)

    return {
        "likely_aligned": likely_aligned,
        "alignment_score": round(alignment, 3),
        "irreversible": is_irreversible,
        "recommend_full_verify": is_irreversible or alignment < 0.15
    }


@mcp.tool()
def drift_history(
    agent_id: str = "",
    run_id: str = "",
    limit: int = 20
) -> dict:
    """
    Review past intent verifications.
    Useful for understanding where an agent consistently drifts.

    Returns:
        { verifications: [...], avg_drift, blocked_count }
    """
    db = get_db()

    if run_id:
        rows = db.execute(
            "SELECT * FROM verifications WHERE run_id=? ORDER BY ts DESC LIMIT ?",
            (run_id, limit)
        ).fetchall()
    elif agent_id:
        rows = db.execute(
            "SELECT * FROM verifications WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
            (agent_id, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM verifications ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()

    verifications = [
        {
            "ts": int(r["ts"]),
            "agent_id": r["agent_id"],
            "proposed_action": r["proposed_action"][:100],
            "drift_score": r["drift_score"],
            "verdict": r["verdict"],
            "blocked": bool(r["blocked"])
        }
        for r in rows
    ]

    drift_scores = [v["drift_score"] for v in verifications if v["drift_score"] is not None]
    avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 0
    blocked = sum(1 for v in verifications if v["blocked"])

    return {
        "verifications": verifications,
        "count": len(verifications),
        "avg_drift": round(avg_drift, 3),
        "blocked_count": blocked,
        "block_rate": round(blocked / len(verifications), 3) if verifications else 0
    }


if __name__ == "__main__":
    print("intent_verify MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
