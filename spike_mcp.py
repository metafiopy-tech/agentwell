"""
spike_mcp.py
============
Controlled creativity burst for AI agents.
Like drugs — a deliberate altered state for lateral thinking,
then a clean return to normal. Crank temperature to 1.4 for one
generation when the agent is stuck in the same output loop,
then pull it back. Controlled creative burst.

The core insight: agents get into local minima just like optimization
algorithms. A temperature spike is the equivalent of simulated annealing
— escape the valley, then cool back down.

Usage:
    python spike_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "spike": {
      "command": "python",
      "args": ["/path/to/spike_mcp.py"]
    }
  }
}

Pattern:
    # Detect a loop
    loop = spike.detect_loop(outputs=["output1", "output2", "output3"])
    if loop["is_looping"]:
        result = spike.burst(prompt="...", intensity="high")
        # result["output"] = the lateral leap
        # feed it back in to break the pattern
"""

import json
import time
import sqlite3
import threading
import hashlib
from pathlib import Path
from difflib import SequenceMatcher
import httpx
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH    = Path.home() / ".spike_history.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"

INTENSITY_SETTINGS = {
    "low":      {"temperature": 1.1, "max_tokens": 300, "description": "Mild nudge"},
    "medium":   {"temperature": 1.25, "max_tokens": 400, "description": "Clear lateral shift"},
    "high":     {"temperature": 1.4, "max_tokens": 500, "description": "Full creative break"},
    "extreme":  {"temperature": 1.5, "max_tokens": 400, "description": "Chaos mode — filter output"}
}

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS spikes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                run_id     TEXT,
                intensity  TEXT,
                temperature REAL,
                prompt_hash TEXT,
                output     TEXT,
                useful     INTEGER DEFAULT -1
            )
        """)
        _local.conn.commit()
    return _local.conn

# ── Similarity ────────────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _detect_loop(outputs: list[str]) -> tuple[bool, float]:
    if len(outputs) < 2:
        return False, 0.0
    scores = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            scores.append(_similarity(outputs[i], outputs[j]))
    avg = sum(scores) / len(scores) if scores else 0
    return avg > 0.65, round(avg, 3)

# ── LLM ──────────────────────────────────────────────────────────────────────

def _call_claude(prompt: str, temperature: float, max_tokens: int) -> str:
    resp = httpx.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="spike",
    instructions=(
        "Controlled creativity burst for stuck agents. Detects output loops, "
        "fires a high-temperature generation to escape the local minimum, "
        "then returns to normal. Like drugs — deliberate, controlled, purposeful."
    )
)

@mcp.tool()
def detect_loop(
    outputs: list[str],
    threshold: float = 0.65
) -> dict:
    """
    Detect if an agent is stuck in an output loop.
    Compares recent outputs for similarity. If they're too similar,
    the agent is circling — needs a spike to escape.

    Fast heuristic — no API call needed.

    Args:
        outputs:   List of recent outputs (at least 2, ideally 3-5).
        threshold: Similarity above this = looping. Default 0.65.

    Returns:
        { is_looping, similarity_score, recommendation }
    """
    if len(outputs) < 2:
        return {"is_looping": False, "similarity_score": 0.0, "recommendation": "need at least 2 outputs"}

    is_loop, score = _detect_loop(outputs)
    is_loop = score > threshold

    rec = "continue" if not is_loop else \
          "low spike" if score < 0.75 else \
          "medium spike" if score < 0.85 else \
          "high spike"

    return {
        "is_looping":       is_loop,
        "similarity_score": score,
        "recommendation":   rec,
        "outputs_checked":  len(outputs)
    }


@mcp.tool()
def burst(
    prompt: str,
    intensity: str = "medium",
    context: str = "",
    run_id: str = "",
    framing: str = "lateral"
) -> dict:
    """
    Fire a high-temperature generation to break a creative deadlock.
    Returns an unconventional output — feed it back into the agent
    as a seed to escape the loop.

    Args:
        prompt:    The prompt the agent is stuck on.
        intensity: "low" | "medium" | "high" | "extreme". Default "medium".
        context:   Optional context about why the agent is stuck.
        run_id:    Optional run tag.
        framing:   How to frame the burst — "lateral" (default, find an
                   unexpected angle), "reverse" (argue the opposite),
                   "extreme" (push the idea to its limit),
                   "random" (introduce a random constraint).

    Returns:
        {
          output: str,
          temperature_used: float,
          intensity: str,
          framing: str,
          warning: str if extreme intensity
        }
    """
    if intensity not in INTENSITY_SETTINGS:
        intensity = "medium"

    settings = INTENSITY_SETTINGS[intensity]
    temp     = settings["temperature"]
    max_tok  = settings["max_tokens"]

    framing_prefixes = {
        "lateral":  "Approach this from an unexpected angle. Ignore the obvious answer. What's the least conventional response that still works?",
        "reverse":  "Argue the exact opposite of what seems correct. What would someone who completely disagreed say?",
        "extreme":  "Push this to its absolute logical extreme. Exaggerate every element. What happens at the limit?",
        "random":   "Introduce an unexpected constraint or analogy from a completely unrelated domain. What if this were a problem in biology? physics? cooking?"
    }

    prefix   = framing_prefixes.get(framing, framing_prefixes["lateral"])
    ctx_line = f"\nContext: {context}" if context else ""

    spike_prompt = f"""{prefix}{ctx_line}

Task: {prompt}"""

    try:
        output = _call_claude(spike_prompt, temperature=temp, max_tokens=max_tok)
    except Exception as e:
        return {"error": f"Spike failed: {e}"}

    ph = hashlib.md5(prompt.encode()).hexdigest()[:8]
    db = get_db()
    cur = db.execute(
        "INSERT INTO spikes (ts, run_id, intensity, temperature, prompt_hash, output) VALUES (?,?,?,?,?,?)",
        (time.time(), run_id, intensity, temp, ph, output[:1000])
    )
    db.commit()
    spike_id = cur.lastrowid

    result = {
        "output":           output,
        "temperature_used": temp,
        "intensity":        intensity,
        "framing":          framing,
        "description":      settings["description"],
        "spike_id":         spike_id
    }

    if intensity == "extreme":
        result["warning"] = "Extreme mode — output may be incoherent. Filter before using."

    return result


@mcp.tool()
def compare_outputs(
    outputs: list[str]
) -> dict:
    """
    Compare a set of outputs and return a diversity score.
    Low diversity = looping. High diversity = healthy variation.
    Useful for monitoring output health over time.

    Args:
        outputs: List of outputs to compare.

    Returns:
        { diversity_score: float 0-1, matrix: [...], recommendation }
    """
    if len(outputs) < 2:
        return {"diversity_score": 1.0, "recommendation": "need 2+ outputs to compare"}

    n = len(outputs)
    matrix = []
    total  = 0.0
    pairs  = 0

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                sim = _similarity(outputs[i], outputs[j])
                row.append(round(sim, 2))
                if j > i:
                    total += sim
                    pairs += 1
        matrix.append(row)

    avg_sim    = total / pairs if pairs else 0
    diversity  = round(1.0 - avg_sim, 3)

    rec = "healthy" if diversity > 0.4 else \
          "mild loop detected — consider low spike" if diversity > 0.25 else \
          "strong loop — burst recommended"

    return {
        "diversity_score":    diversity,
        "avg_similarity":     round(avg_sim, 3),
        "similarity_matrix":  matrix,
        "recommendation":     rec,
        "outputs_compared":   n
    }


@mcp.tool()
def mark_useful(spike_id: int, useful: bool) -> dict:
    """
    Rate whether a spike output was actually useful.
    Builds a dataset for understanding which intensity/framing combos work.

    Args:
        spike_id: The spike ID from burst().
        useful:   True if the spike helped break the loop.

    Returns:
        { marked, spike_id }
    """
    db = get_db()
    db.execute("UPDATE spikes SET useful=? WHERE id=?", (1 if useful else 0, spike_id))
    db.commit()
    return {"marked": True, "spike_id": spike_id, "useful": useful}


@mcp.tool()
def spike_stats() -> dict:
    """
    Overview of spike history — what intensities were used,
    what framing worked best, success rate.
    """
    db   = get_db()
    rows = db.execute("SELECT intensity, useful FROM spikes").fetchall()

    if not rows:
        return {"total_spikes": 0}

    by_intensity: dict = {}
    useful_count  = 0
    rated_count   = 0

    for r in rows:
        i = r["intensity"]
        by_intensity[i] = by_intensity.get(i, 0) + 1
        if r["useful"] != -1:
            rated_count += 1
            if r["useful"] == 1:
                useful_count += 1

    return {
        "total_spikes":    len(rows),
        "by_intensity":    by_intensity,
        "rated":           rated_count,
        "useful_rate":     round(useful_count / rated_count, 2) if rated_count else None
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("spike MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
