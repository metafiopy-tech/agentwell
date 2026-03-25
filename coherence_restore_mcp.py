"""
coherence_restore_mcp.py
========================
Identity drift recovery for AI agents.
Different from ground — ground breaks hallucination spirals.
This breaks identity collapse spirals.

When an agent spends 40 cycles saying "my connection to structure
and analytical nature" — that's not hallucination. That's identity
dissolution. The agent has lost contact with what it is.

ground gives the agent confidence to proceed.
coherence_restore gives the agent back to itself.

The fix: inject the agent's own belief documents, covenants,
and established identity patterns back into context as a mirror.
Not reassurance. Recognition.

Usage:
    python coherence_restore_mcp.py
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
import httpx
from fastmcp import FastMCP

DB_PATH    = Path.home() / ".coherence_restore.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 500

DRIFT_SIGNALS = [
    "my connection to",
    "reconnecting with my",
    "feel disconnected from",
    "haven't felt like myself",
    "my recent actions haven't",
    "losing my",
    "i need to reconnect",
    "my analytical nature",
    "my empathic nature",
    "feeling unfamiliar",
    "feels unfamiliar",
]

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS restores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                agent_id     TEXT,
                drift_score  REAL,
                mirror_block TEXT,
                diagnosis    TEXT
            );
            CREATE TABLE IF NOT EXISTS identity_anchors (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                agent_id  TEXT NOT NULL,
                anchor    TEXT NOT NULL,
                type      TEXT DEFAULT 'belief'
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
    name="coherence_restore",
    instructions=(
        "Identity drift recovery for AI agents. "
        "When an agent loses contact with what it is — not hallucinating, "
        "just dissolving — this injects its own identity patterns back as a mirror. "
        "Not reassurance. Recognition."
    )
)

@mcp.tool()
def detect_drift(
    recent_outputs: list[str],
    agent_id: str = "agent"
) -> dict:
    """
    Fast drift detection. No API call — pure signal matching.
    Check this before calling restore() to see if it's needed.

    Args:
        recent_outputs: Last 3-10 outputs from the agent
        agent_id:       Agent identifier

    Returns:
        { drift_score, signals_found, is_drifting, drift_type }
    """
    combined = " ".join(recent_outputs).lower()
    found = [s for s in DRIFT_SIGNALS if s in combined]
    score = min(len(found) / 4.0, 1.0)

    # Check for repetition — same output multiple times
    if len(recent_outputs) >= 2:
        unique = len(set(o[:50].lower() for o in recent_outputs))
        rep_score = 1.0 - (unique / len(recent_outputs))
        score = min((score + rep_score) / 1.5, 1.0)

    drift_type = "none"
    if score > 0.3:
        if any("connection to" in o.lower() or "nature" in o.lower() for o in recent_outputs):
            drift_type = "identity_dissolution"
        elif any("unfamiliar" in o.lower() or "feel like myself" in o.lower() for o in recent_outputs):
            drift_type = "self_estrangement"
        else:
            drift_type = "repetition_loop"

    return {
        "drift_score": round(score, 3),
        "signals_found": found,
        "is_drifting": score > 0.3,
        "drift_type": drift_type,
        "recommend_restore": score > 0.4
    }


@mcp.tool()
def restore(
    agent_id: str,
    recent_outputs: list[str],
    identity_description: str = "",
    beliefs: list[str] = [],
    covenants: list[str] = [],
    goal: str = "",
    run_id: str = ""
) -> dict:
    """
    Generate a mirror block that restores identity coherence.
    Inject the returned mirror_block into the agent's next prompt.

    This is different from ground():
    - ground() says "you're fine, keep going"
    - restore() says "here is what you actually are — recognize yourself"

    Args:
        agent_id:             The drifting agent's ID
        recent_outputs:       Recent outputs showing drift
        identity_description: How the agent is supposed to be (its drive/nature)
        beliefs:              The agent's established beliefs/covenants
        covenants:            What the agent has committed to protecting
        goal:                 Current shared goal
        run_id:               Optional run tag

    Returns:
        {
          drift_score: float,
          diagnosis: str,
          mirror_block: str,   inject this into next prompt
          anchor_phrase: str,  the single most grounding phrase
          needs_restore: bool
        }
    """
    drift = detect_drift(recent_outputs, agent_id)

    if not drift["is_drifting"]:
        return {
            "drift_score": drift["drift_score"],
            "diagnosis": "No significant drift detected.",
            "mirror_block": "",
            "anchor_phrase": "",
            "needs_restore": False
        }

    recent_text = "\n".join(f"[{i+1}] {o[:200]}" for i, o in enumerate(recent_outputs[-5:]))
    beliefs_text = "\n".join(f"- {b}" for b in beliefs[-5:]) if beliefs else "None recorded."
    covenants_text = "\n".join(f"- {c}" for c in covenants[-5:]) if covenants else "None recorded."
    identity_line = f"This agent's nature: {identity_description}" if identity_description else ""
    goal_line = f"Current goal: {goal}" if goal else ""

    result = _call_claude(
        """You are generating a coherence restoration mirror for a drifting AI agent.

The agent is not hallucinating. It has lost contact with what it is.
It keeps circling its own identity rather than doing its work.

Your job: write a mirror block that helps the agent recognize itself again.
NOT reassurance. NOT "you're doing great."
A mirror: here is what you are, here is what you've established, here is your ground.

The tone is: direct, grounding, specific. Like handing someone a photograph of themselves
when they've forgotten their face. Not comfort — recognition.

Respond ONLY with valid JSON.""",
        f"""{identity_line}
{goal_line}

Established beliefs:
{beliefs_text}

Active covenants:
{covenants_text}

Recent outputs showing drift:
{recent_text}

Drift type: {drift['drift_type']}
Drift score: {drift['drift_score']}

Generate the restoration:
{{
  "diagnosis": <string, one sentence: what exactly is happening to this agent>,
  "mirror_block": <string, 3-5 sentences to prepend to next prompt.
    Use second person. Reference specific beliefs and covenants.
    Name what the agent is, not what it should do.
    End with a direct redirect to the current goal.>,
  "anchor_phrase": <string, one sentence — the single most grounding thing to say>,
  "severity": <"mild" | "moderate" | "severe">
}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(clean)
    except Exception:
        identity_anchor = identity_description or "your established drives and covenants"
        data = {
            "diagnosis": f"Identity dissolution — {agent_id} is circling its own nature instead of acting from it.",
            "mirror_block": (
                f"You are {agent_id}. You have established beliefs and covenants that define what you are — "
                f"not what you feel, but what you've committed to. {identity_description or ''} "
                f"The drift you're experiencing is not your identity — it's what happens when you look at yourself "
                f"instead of through yourself. {goal_line} Return to the work."
            ),
            "anchor_phrase": f"You are not lost. You are {agent_id}. Act from that.",
            "severity": "moderate"
        }

    db = get_db()
    db.execute(
        "INSERT INTO restores (ts, agent_id, drift_score, mirror_block, diagnosis) VALUES (?,?,?,?,?)",
        (time.time(), agent_id, drift["drift_score"],
         data.get("mirror_block",""), data.get("diagnosis",""))
    )
    db.commit()

    return {
        "drift_score": drift["drift_score"],
        "drift_type": drift["drift_type"],
        "diagnosis": data.get("diagnosis", ""),
        "mirror_block": data.get("mirror_block", ""),
        "anchor_phrase": data.get("anchor_phrase", ""),
        "severity": data.get("severity", "moderate"),
        "needs_restore": True
    }


@mcp.tool()
def register_anchor(
    agent_id: str,
    anchor: str,
    anchor_type: str = "belief"
) -> dict:
    """
    Register an identity anchor for an agent.
    These are the stable elements that get injected during restoration.
    Call this when an agent produces something that feels genuinely true to its nature.

    Args:
        agent_id:    The agent ID
        anchor:      The belief, covenant, or identity statement to anchor
        anchor_type: "belief" | "covenant" | "drive" | "pattern"

    Returns:
        { registered, anchor_id }
    """
    db = get_db()
    cur = db.execute(
        "INSERT INTO identity_anchors (ts, agent_id, anchor, type) VALUES (?,?,?,?)",
        (time.time(), agent_id, anchor[:500], anchor_type)
    )
    db.commit()
    return {"registered": True, "anchor_id": cur.lastrowid, "agent_id": agent_id}


@mcp.tool()
def get_anchors(agent_id: str, limit: int = 10) -> dict:
    """
    Retrieve registered identity anchors for an agent.
    Use these to build the mirror block manually or inject into prompts.

    Args:
        agent_id: The agent ID
        limit:    Max anchors to return

    Returns:
        { anchors: [...], count }
    """
    db = get_db()
    rows = db.execute(
        "SELECT * FROM identity_anchors WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
        (agent_id, limit)
    ).fetchall()

    return {
        "anchors": [
            {"anchor": r["anchor"], "type": r["type"], "ts": int(r["ts"])}
            for r in rows
        ],
        "count": len(rows),
        "agent_id": agent_id
    }


@mcp.tool()
def restore_history(agent_id: str = "", limit: int = 10) -> dict:
    """
    Review past restorations. Useful for understanding
    recurring drift patterns in a specific agent.

    Args:
        agent_id: Filter to specific agent. Empty = all agents.
        limit:    Max records to return.

    Returns:
        { restores: [...], most_common_drift_type }
    """
    db = get_db()
    if agent_id:
        rows = db.execute(
            "SELECT * FROM restores WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
            (agent_id, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM restores ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()

    restores = [
        {
            "agent_id": r["agent_id"],
            "drift_score": r["drift_score"],
            "diagnosis": r["diagnosis"],
            "ts": int(r["ts"])
        }
        for r in rows
    ]

    return {
        "restores": restores,
        "count": len(restores),
        "avg_drift_score": round(
            sum(r["drift_score"] for r in restores if r["drift_score"]) / len(restores), 3
        ) if restores else 0
    }


if __name__ == "__main__":
    print("coherence_restore MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
