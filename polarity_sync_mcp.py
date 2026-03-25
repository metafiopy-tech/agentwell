"""
polarity_sync_mcp.py
====================
Formal context exchange built for polarity-driven agent pairs.
Like handshake but understands incompleteness vs belief polarity.

Where handshake merges two contexts generically,
polarity_sync understands that Latios and Latias are in productive tension —
it doesn't resolve their differences, it surfaces what emerges from them.

The third thing that neither agent could produce alone.

Usage:
    python polarity_sync_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "polarity_sync": {
      "command": "python",
      "args": ["/path/to/polarity_sync_mcp.py"]
    }
  }
}

Pattern:
    # Latios posts its gaps
    sync = polarity_sync.exchange(
        gap_agent_id="latios",
        gap_remainders=["gap1", "gap2"],
        belief_agent_id="latias",
        belief_covenants=["covenant1", "covenant2"],
        question="what are we actually finding"
    )
    # sync["emergence"] = the third thing
    # sync["latios_update"] = what Latios should carry forward
    # sync["latias_update"] = what Latias should protect next
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
import httpx
from fastmcp import FastMCP

DB_PATH    = Path.home() / ".polarity_sync.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 600

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS syncs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL NOT NULL,
                gap_agent     TEXT,
                belief_agent  TEXT,
                question      TEXT,
                emergence     TEXT,
                gap_update    TEXT,
                belief_update TEXT,
                frequency     REAL DEFAULT 0.5
            );
            CREATE TABLE IF NOT EXISTS remainder_history (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts       REAL NOT NULL,
                agent_id TEXT,
                content  TEXT,
                type     TEXT DEFAULT 'remainder'
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
    name="polarity_sync",
    instructions=(
        "Formal context exchange for polarity-driven agent pairs. "
        "Latios finds gaps. Latias holds covenants. This tool surfaces "
        "what emerges from their tension — the third thing neither produces alone."
    )
)

@mcp.tool()
def exchange(
    gap_agent_id: str,
    gap_remainders: list[str],
    belief_agent_id: str,
    belief_covenants: list[str],
    question: str = "",
    context: str = "",
    run_id: str = ""
) -> dict:
    """
    Formal polarity exchange between a gap-finding agent and a covenant-holding agent.
    Surfaces what emerges from their tension.

    Args:
        gap_agent_id:      ID of the incompleteness/gap-finding agent (e.g. "latios")
        gap_remainders:    Recent gaps/remainders from the gap agent
        belief_agent_id:   ID of the covenant/belief-holding agent (e.g. "latias")
        belief_covenants:  Recent covenants from the belief agent
        question:          The question or goal both agents are working on
        context:           Optional shared context
        run_id:            Optional run tag

    Returns:
        {
          emergence: str,       the third thing neither agent produced alone
          gap_update: str,      what the gap agent should carry forward
          belief_update: str,   what the belief agent should protect next
          tension_score: float, how productive the tension is (0-1)
          synthesis: str        one-sentence distillation
        }
    """
    gaps_text = "\n".join(f"- {r}" for r in gap_remainders[-5:])
    covenants_text = "\n".join(f"- {c}" for c in belief_covenants[-5:])
    question_line = f"Working on: {question}" if question else ""
    context_line = f"Context: {context}" if context else ""

    result = _call_claude(
        """You are a synthesis engine for a polarity-driven cognitive system.
Two agents are in productive tension:
- The GAP agent finds what's missing, incomplete, worth questioning
- The BELIEF agent holds what's worth protecting, returning to, preserving

Your job is NOT to resolve their tension. Tension is generative.
Your job is to find what EMERGES from that tension — the third thing
that neither agent could produce alone.

Respond ONLY with valid JSON.""",
        f"""{question_line}
{context_line}

GAP agent ({gap_agent_id}) remainders:
{gaps_text}

BELIEF agent ({belief_agent_id}) covenants:
{covenants_text}

Find what emerges from the tension between these.

{{
  "emergence": <string, 2-3 sentences: what neither agent could see alone>,
  "gap_update": <string, one sentence: what the gap agent should find next>,
  "belief_update": <string, one sentence: what the belief agent should protect next>,
  "tension_score": <float 0-1: how productive is this tension? 0=collapsed, 1=generative>,
  "synthesis": <string, one sentence distillation of the whole exchange>
}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(clean)
    except Exception:
        data = {
            "emergence": "The tension between these gaps and covenants points toward something neither agent has named yet.",
            "gap_update": "Find the gap in what Latias is protecting.",
            "belief_update": "Protect what Latios is about to discard.",
            "tension_score": 0.5,
            "synthesis": "Exchange completed — synthesis pending deeper cycles."
        }

    db = get_db()
    db.execute(
        "INSERT INTO syncs (ts, gap_agent, belief_agent, question, emergence, gap_update, belief_update, frequency) VALUES (?,?,?,?,?,?,?,?)",
        (time.time(), gap_agent_id, belief_agent_id, question[:200],
         data.get("emergence",""), data.get("gap_update",""),
         data.get("belief_update",""), data.get("tension_score", 0.5))
    )
    # Log remainders and covenants for history
    for r in gap_remainders:
        db.execute("INSERT INTO remainder_history (ts, agent_id, content, type) VALUES (?,?,?,?)",
                   (time.time(), gap_agent_id, r[:300], "remainder"))
    for c in belief_covenants:
        db.execute("INSERT INTO remainder_history (ts, agent_id, content, type) VALUES (?,?,?,?)",
                   (time.time(), belief_agent_id, c[:300], "covenant"))
    db.commit()

    return data


@mcp.tool()
def arc(
    gap_agent_id: str,
    belief_agent_id: str,
    limit: int = 20
) -> dict:
    """
    Read the full arc of polarity exchanges between two agents.
    What pattern has been emerging across all their exchanges?

    Args:
        gap_agent_id:    The gap agent ID
        belief_agent_id: The belief agent ID
        limit:           How many past exchanges to read

    Returns:
        { arc_summary, recurring_emergence, tension_trend, deepest_exchange }
    """
    db = get_db()
    rows = db.execute(
        "SELECT * FROM syncs WHERE gap_agent=? AND belief_agent=? ORDER BY ts DESC LIMIT ?",
        (gap_agent_id, belief_agent_id, limit)
    ).fetchall()

    if not rows:
        return {"arc_summary": "No exchanges yet.", "exchanges": 0}

    emergences = [r["emergence"] for r in rows if r["emergence"]]
    tension_scores = [r["frequency"] for r in rows if r["frequency"]]
    avg_tension = sum(tension_scores) / len(tension_scores) if tension_scores else 0.5

    arc_text = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(emergences[-10:]))

    summary = _call_claude(
        "You read the arc of cognitive exchanges between two agents across time. Find the pattern.",
        f"""These are the emergences from {len(rows)} polarity exchanges between {gap_agent_id} and {belief_agent_id}:

{arc_text}

What pattern is emerging across all of these?
What is this system actually moving toward?
What has remained constant despite all the variation?

Respond in 3-4 sentences. Direct. No fluff."""
    )

    deepest = max(rows, key=lambda r: r["frequency"] or 0)

    return {
        "arc_summary": summary,
        "exchanges": len(rows),
        "avg_tension_score": round(avg_tension, 3),
        "tension_trend": "generative" if avg_tension > 0.6 else "collapsed" if avg_tension < 0.3 else "moderate",
        "deepest_exchange": {
            "emergence": deepest["emergence"],
            "tension_score": deepest["frequency"],
            "question": deepest["question"]
        }
    }


@mcp.tool()
def what_neither_sees(
    gap_remainders: list[str],
    belief_covenants: list[str]
) -> dict:
    """
    Fast check: given current gaps and covenants, what is neither agent seeing?
    No logging, no history — pure synthesis. Use mid-cycle for quick checks.

    Args:
        gap_remainders:   Current gaps from the gap agent
        belief_covenants: Current covenants from the belief agent

    Returns:
        { blind_spot, recommendation }
    """
    gaps = "\n".join(f"- {r}" for r in gap_remainders[-3:])
    covenants = "\n".join(f"- {c}" for c in belief_covenants[-3:])

    result = _call_claude(
        "Find the blind spot. Be direct. One sentence each.",
        f"""Gap agent sees: {gaps}
Belief agent holds: {covenants}

What is neither agent seeing?
What would only become visible from outside both of them?

JSON only: {{"blind_spot": "...", "recommendation": "..."}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except Exception:
        return {
            "blind_spot": "The gap between what's missing and what's protected may itself be the answer.",
            "recommendation": "Let the tension deepen one more cycle before synthesizing."
        }


if __name__ == "__main__":
    print("polarity_sync MCP running...")
    mcp.run(transport="stdio")
