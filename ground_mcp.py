"""
ground_mcp.py
=============
Confidence injection for AI agents mid-run.
Like a hug — grounds the agent when it detects uncertainty spiraling
into hallucination. Injects a calibrated reassurance + reorientation
block into the agent's context to break the catastrophizing loop.

Usage:
    python ground_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "ground": {
      "command": "python",
      "args": ["/path/to/ground_mcp.py"]
    }
  }
}

Call when you detect hedging language, excessive caveats, or loops:
    result = ground(context="...", symptoms=["excessive hedging", "repeating same output"])
    # inject result["grounding_block"] into next prompt
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
import httpx
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH    = Path.home() / ".ground_history.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 400

SPIRAL_SIGNALS = [
    "i'm not sure",
    "i cannot determine",
    "it's unclear",
    "i may be wrong",
    "i'm uncertain",
    "i don't have enough",
    "i cannot be certain",
    "it's possible that",
    "i might be",
    "i could be wrong",
    "i'm unable to",
    "i apologize",
    "i'm sorry",
]

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS grounds (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                run_id    TEXT,
                trigger   TEXT,
                block     TEXT,
                spiral_score REAL
            )
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
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="ground",
    instructions=(
        "Confidence injection for agents experiencing uncertainty spirals. "
        "Detects hallucination-risk patterns and injects a grounding block "
        "that reorients the agent without breaking its task flow. "
        "Call when you notice hedging, repetition, or excessive caveating."
    )
)

@mcp.tool()
def ground(
    context: str,
    goal: str = "",
    symptoms: list[str] = [],
    run_id: str = ""
) -> dict:
    """
    Generate a grounding block to inject into the agent's context
    when it's spiraling into uncertainty or over-hedging.

    The returned grounding_block is a short paragraph designed to be
    prepended to the agent's next prompt. It validates what's been
    done well, names the uncertainty clearly, and redirects forward.

    Args:
        context:  The agent's recent output showing signs of drift.
        goal:     What the agent is supposed to be doing.
        symptoms: Observed issues e.g. ["repeating itself", "over-caveating"]
        run_id:   Optional run tag.

    Returns:
        {
          spiral_score: float 0-1 (how bad is the spiral),
          grounding_block: str (inject this into the next prompt),
          diagnosis: str,
          needs_grounding: bool
        }
    """
    if not context:
        return {"error": "No context provided."}

    # Quick spiral detection before hitting the API
    context_lower = context.lower()
    signal_hits = sum(1 for s in SPIRAL_SIGNALS if s in context_lower)
    quick_spiral = min(signal_hits / 5.0, 1.0)

    symptom_str = ", ".join(symptoms) if symptoms else "none specified"
    goal_line   = f"Goal: {goal}" if goal else ""

    prompt = f"""An AI agent is showing signs of an uncertainty spiral mid-run.
{goal_line}
Observed symptoms: {symptom_str}

Recent agent output:
---
{context[:1200]}
---

Generate a grounding intervention. Respond ONLY with valid JSON:
{{
  "spiral_score": <float 0.0-1.0, severity of the spiral>,
  "diagnosis": <string, one sentence explaining what's happening>,
  "grounding_block": <string, 2-4 sentence paragraph to prepend to next prompt.
    Tone: calm, direct, confident. Acknowledge what's solid, name the uncertainty
    precisely, redirect to the next concrete step. Do NOT be sycophantic.
    Do NOT say 'great job'. Just reorient.>,
  "needs_grounding": <bool, false if the agent is actually fine>
}}"""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        # Fallback grounding block if API fails
        fallback_block = (
            "The reasoning so far is coherent. The uncertainty you're experiencing "
            "is normal at this stage. Pick the most defensible assumption, state it "
            "explicitly, and continue. You can revisit it later if needed."
        )
        return {
            "spiral_score":    quick_spiral,
            "diagnosis":       "API eval failed — fallback grounding applied",
            "grounding_block": fallback_block,
            "needs_grounding": quick_spiral > 0.3,
            "fallback":        True,
            "error":           str(e)
        }

    spiral_score  = float(result.get("spiral_score", quick_spiral))
    grounding     = result.get("grounding_block", "")
    diagnosis     = result.get("diagnosis", "")
    needs         = result.get("needs_grounding", spiral_score > 0.3)

    db = get_db()
    db.execute(
        "INSERT INTO grounds (ts, run_id, trigger, block, spiral_score) VALUES (?,?,?,?,?)",
        (time.time(), run_id, context[:500], grounding, spiral_score)
    )
    db.commit()

    return {
        "spiral_score":    round(spiral_score, 3),
        "diagnosis":       diagnosis,
        "grounding_block": grounding,
        "needs_grounding": needs,
        "signal_hits":     signal_hits
    }


@mcp.tool()
def detect_spiral(text: str) -> dict:
    """
    Fast spiral detection without generating a grounding block.
    Use this as a cheap pre-check before calling ground().
    No API call — pure heuristic signal matching.

    Args:
        text: Any agent output to check.

    Returns:
        { spiral_score, signals_found, recommend_grounding }
    """
    text_lower = text.lower()
    found = [s for s in SPIRAL_SIGNALS if s in text_lower]
    score = min(len(found) / 5.0, 1.0)

    # Also check for repetition (same sentence appearing multiple times)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
    unique    = len(set(sentences))
    rep_score = 1.0 - (unique / max(len(sentences), 1)) if sentences else 0
    combined  = min((score + rep_score) / 1.5, 1.0)

    return {
        "spiral_score":        round(combined, 3),
        "signals_found":       found,
        "repetition_score":    round(rep_score, 3),
        "recommend_grounding": combined > 0.3
    }


@mcp.tool()
def reorient(
    goal: str,
    last_completed_step: str,
    next_step: str
) -> dict:
    """
    Lightweight reorientation — no spiral detection, just a clean
    'here's where you are, here's what's next' block.
    Like a gentle tap on the shoulder rather than a full hug.

    Args:
        goal:                The agent's overall objective.
        last_completed_step: What was just finished.
        next_step:           What comes next.

    Returns:
        { reorientation_block: str }
    """
    block = (
        f"Progress check: the goal is '{goal}'. "
        f"Completed: {last_completed_step}. "
        f"Next: {next_step}. "
        f"Proceed directly — no need to restate what's already been done."
    )
    return {"reorientation_block": block}


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ground MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
