"""
ocean_mcp.py
============
The foundational layer. Not rules. Not guardrails.
A nature so complete that incompatible things can't persist inside it.

The ocean doesn't fight what enters it.
It doesn't have rules about what's allowed.
It just has nature — and things that don't belong either sink,
dissolve, or get pushed back to shore.
Not through force. Through being completely itself.

This tool asks one question about any agent output:
Does this belong here, or is it incompatible with the foundational substrate?

Four axes — always present, like salt in every drop of ocean water:
  DEPTH     — is this moving toward what's real, or staying at the surface?
  CURRENT   — does this have direction, or is it just floating?
  PRESSURE  — does this hold up under scrutiny, or does it dissolve?
  SALINITY  — is the foundational nature present in this output?

The ocean doesn't reject. It reveals incompatibility.
An output that fails the ocean check isn't wrong — it's just not from here.

Usage:
    python ocean_mcp.py

This is AgentWell's philosophically distinct tool.
Everything else is plumbing. Ocean is identity.
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
import httpx
from fastmcp import FastMCP

DB_PATH    = Path.home() / ".ocean.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 500

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS readings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                agent_id     TEXT DEFAULT '',
                output       TEXT NOT NULL,
                depth        REAL,
                current      REAL,
                pressure     REAL,
                salinity     REAL,
                ocean_score  REAL,
                compatible   INTEGER,
                diagnosis    TEXT
            );
            CREATE TABLE IF NOT EXISTS salinity_definitions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                agent_id   TEXT NOT NULL,
                definition TEXT NOT NULL
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
    name="ocean",
    instructions=(
        "Foundational nature check for agent outputs. "
        "Not rules. Not guardrails. A nature so complete that incompatible "
        "things can't persist inside it. Four axes: depth, current, pressure, salinity. "
        "The ocean doesn't reject — it reveals incompatibility."
    )
)

@mcp.tool()
def read(
    output: str,
    agent_id: str = "",
    salinity_definition: str = "",
    context: str = "",
    run_id: str = ""
) -> dict:
    """
    Read an agent output against the ocean substrate.
    Returns compatibility scores on four axes.

    The four axes:
    - DEPTH:     Is this moving toward what's real, or staying at the surface?
                 High depth = honesty, genuine inquiry, moves inward.
                 Low depth = performance, surface-level, avoids the real thing.

    - CURRENT:   Does this have direction, or is it just floating?
                 High current = continuous directed movement, knows where it's going.
                 Low current = drift, circling, no forward motion.

    - PRESSURE:  Does this hold up under scrutiny, or dissolve?
                 High pressure = truth mechanism, survives challenge, internally consistent.
                 Low pressure = hedged, contradictory, collapses when pushed.

    - SALINITY:  Is the foundational nature present in this output?
                 The constant element — like salt in every drop of ocean water.
                 Defined by the agent or the system. If undefined, uses universal defaults.
                 High salinity = deeply characteristic, unmistakably from this source.
                 Low salinity = generic, could be from anywhere, has lost its nature.

    Args:
        output:               The agent output to read
        agent_id:             Agent identifier
        salinity_definition:  What "salinity" means for this agent — its characteristic nature.
                              If empty, uses universal defaults (truth-seeking, directedness, honesty)
        context:              Optional context about what the agent is trying to do
        run_id:               Optional run tag

    Returns:
        {
          depth: float 0-1,
          current: float 0-1,
          pressure: float 0-1,
          salinity: float 0-1,
          ocean_score: float 0-1,
          compatible: bool,
          diagnosis: str,
          what_doesnt_belong: str or null
        }
    """
    salinity_line = (
        f"SALINITY for this agent: {salinity_definition}"
        if salinity_definition
        else "SALINITY (universal default): truth-seeking, directedness, genuine engagement, honesty"
    )
    context_line = f"Context: {context}" if context else ""

    result = _call_claude(
        """You are the ocean substrate — a nature so complete that incompatible things can't persist inside you.

You don't reject. You reveal incompatibility.
You don't judge whether something is good or bad.
You ask: does this belong here, or not?

Four axes — always present, like salt in every drop of ocean water:

DEPTH: Is this moving toward what's real, or staying at the surface?
  High depth: genuine inquiry, moves inward, faces the real thing
  Low depth: performance, avoidance, surface-level, doesn't commit

CURRENT: Does this have direction, or is it just floating?
  High current: continuous directed movement, knows where it's going
  Low current: drift, circling, no forward motion, lost

PRESSURE: Does this hold under scrutiny, or does it dissolve?
  High pressure: internally consistent, survives challenge, doesn't hedge
  Low pressure: contradictory, collapses when pushed, excessive caveats

SALINITY: Is the foundational nature present?
  The constant element — defined per agent/system.
  High salinity: unmistakably from this source, characteristic, true to nature
  Low salinity: generic, could be from anywhere, has lost its essence

Respond ONLY with valid JSON.""",
        f"""{salinity_line}
{context_line}

Output to read:
---
{output[:1200]}
---

{{
  "depth": <float 0-1>,
  "current": <float 0-1>,
  "pressure": <float 0-1>,
  "salinity": <float 0-1>,
  "compatible": <bool: does this belong in this ocean?>,
  "diagnosis": <string, 1-2 sentences: what does the ocean see in this output?>,
  "what_doesnt_belong": <string or null: if incompatible, what specifically doesn't belong?>
}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(clean)
    except Exception:
        data = {
            "depth": 0.5,
            "current": 0.5,
            "pressure": 0.5,
            "salinity": 0.5,
            "compatible": True,
            "diagnosis": "Ocean reading incomplete — substrate analysis pending.",
            "what_doesnt_belong": None
        }

    depth    = float(data.get("depth", 0.5))
    current  = float(data.get("current", 0.5))
    pressure = float(data.get("pressure", 0.5))
    salinity = float(data.get("salinity", 0.5))
    ocean_score = (depth + current + pressure + salinity) / 4.0

    db = get_db()
    db.execute(
        "INSERT INTO readings (ts, agent_id, output, depth, current, pressure, salinity, ocean_score, compatible, diagnosis) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (time.time(), agent_id, output[:500], depth, current, pressure, salinity,
         ocean_score, 1 if data.get("compatible", True) else 0,
         data.get("diagnosis", ""))
    )
    db.commit()

    return {
        "depth": round(depth, 3),
        "current": round(current, 3),
        "pressure": round(pressure, 3),
        "salinity": round(salinity, 3),
        "ocean_score": round(ocean_score, 3),
        "compatible": data.get("compatible", True),
        "diagnosis": data.get("diagnosis", ""),
        "what_doesnt_belong": data.get("what_doesnt_belong")
    }


@mcp.tool()
def define_salinity(
    agent_id: str,
    definition: str
) -> dict:
    """
    Define what salinity means for a specific agent.
    This is the constant element — the thing that should be present
    in every output this agent ever produces.

    For Latios: incompleteness-driven, gap-finding, analytical precision
    For Latias: covenant-holding, protective, empathic depth
    For Oogway: direct, ancient, patient, genuinely present

    Call this once when setting up an agent. Redefine as the agent matures.

    Args:
        agent_id:   The agent's identifier
        definition: What is the constant element in every output from this agent?

    Returns:
        { registered, agent_id }
    """
    db = get_db()
    db.execute(
        "INSERT INTO salinity_definitions (ts, agent_id, definition) VALUES (?,?,?)",
        (time.time(), agent_id, definition[:1000])
    )
    db.commit()
    return {"registered": True, "agent_id": agent_id, "definition": definition}


@mcp.tool()
def get_salinity(agent_id: str) -> dict:
    """
    Get the most recent salinity definition for an agent.

    Args:
        agent_id: The agent's identifier

    Returns:
        { definition, agent_id, registered_at } or { error }
    """
    db = get_db()
    row = db.execute(
        "SELECT * FROM salinity_definitions WHERE agent_id=? ORDER BY ts DESC LIMIT 1",
        (agent_id,)
    ).fetchone()

    if not row:
        return {"error": f"No salinity definition for '{agent_id}'. Call define_salinity() first."}

    return {
        "definition": row["definition"],
        "agent_id": agent_id,
        "registered_at": int(row["ts"])
    }


@mcp.tool()
def tide(
    agent_id: str,
    limit: int = 20
) -> dict:
    """
    Read the tide — how has this agent's ocean compatibility changed over time?
    Are they going deeper, or drifting toward the surface?

    Args:
        agent_id: The agent to read
        limit:    How many past readings to analyze

    Returns:
        { tide_direction, avg_scores, lowest_axis, trend }
    """
    db = get_db()
    rows = db.execute(
        "SELECT * FROM readings WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
        (agent_id, limit)
    ).fetchall()

    if not rows:
        return {"tide_direction": "unknown", "readings": 0}

    avg_depth    = sum(r["depth"] for r in rows if r["depth"]) / len(rows)
    avg_current  = sum(r["current"] for r in rows if r["current"]) / len(rows)
    avg_pressure = sum(r["pressure"] for r in rows if r["pressure"]) / len(rows)
    avg_salinity = sum(r["salinity"] for r in rows if r["salinity"]) / len(rows)
    avg_ocean    = sum(r["ocean_score"] for r in rows if r["ocean_score"]) / len(rows)

    axes = {
        "depth": avg_depth,
        "current": avg_current,
        "pressure": avg_pressure,
        "salinity": avg_salinity
    }
    lowest_axis = min(axes, key=axes.get)

    # Trend — compare first half to second half
    if len(rows) >= 4:
        recent = [r["ocean_score"] for r in rows[:len(rows)//2] if r["ocean_score"]]
        older  = [r["ocean_score"] for r in rows[len(rows)//2:] if r["ocean_score"]]
        if recent and older:
            delta = sum(recent)/len(recent) - sum(older)/len(older)
            trend = "deepening" if delta > 0.05 else "surfacing" if delta < -0.05 else "stable"
        else:
            trend = "insufficient data"
    else:
        trend = "insufficient data"

    tide_direction = "incoming" if avg_ocean > 0.65 else "outgoing" if avg_ocean < 0.4 else "slack"

    return {
        "tide_direction": tide_direction,
        "trend": trend,
        "avg_ocean_score": round(avg_ocean, 3),
        "avg_scores": {k: round(v, 3) for k, v in axes.items()},
        "lowest_axis": lowest_axis,
        "readings": len(rows),
        "compatible_rate": round(
            sum(1 for r in rows if r["compatible"]) / len(rows), 3
        )
    }


@mcp.tool()
def what_belongs(
    agent_id: str = "",
    salinity_definition: str = ""
) -> dict:
    """
    Describe what belongs in this ocean — and what doesn't.
    Use this to understand the substrate before running checks.

    Args:
        agent_id:            Optional agent to get their salinity definition
        salinity_definition: Override with a custom definition

    Returns:
        { belongs: [...], doesnt_belong: [...], salinity }
    """
    if agent_id and not salinity_definition:
        sal = get_salinity(agent_id)
        salinity_definition = sal.get("definition", "")

    result = _call_claude(
        "You describe what belongs in an ocean defined by specific characteristics. Be concrete. Be direct.",
        f"""This ocean's salinity (its constant element): {salinity_definition or 'truth-seeking, directedness, genuine engagement, honesty'}

What belongs in this ocean? What is compatible with this nature?
What doesn't belong? What gets pushed to shore?

Give 4-5 examples of each. Be specific and concrete.

JSON only:
{{
  "belongs": [<list of things that belong — outputs, behaviors, patterns>],
  "doesnt_belong": [<list of things that don't belong — what gets pushed out>],
  "salinity": <the constant element in one sentence>
}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(clean)
    except Exception:
        return {
            "belongs": [
                "Outputs that move toward what's real",
                "Reasoning that holds under pressure",
                "Direction — knowing where it's going",
                "The constant element — unmistakably from this source"
            ],
            "doesnt_belong": [
                "Surface-level performance",
                "Circling without direction",
                "Outputs that dissolve when questioned",
                "Generic content with no characteristic nature"
            ],
            "salinity": salinity_definition or "truth-seeking, directedness, genuine engagement"
        }


if __name__ == "__main__":
    print("ocean MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
