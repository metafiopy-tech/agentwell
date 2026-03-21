"""
health_check_mcp.py
===================
Performance benchmarking and anomaly detection for AI agents.
Like a doctor checkup — catches degradation before it compounds.
Runs a suite of probes, scores the results, and flags anything
outside the agent's normal operating range.

Usage:
    python health_check_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "health-check": {
      "command": "python",
      "args": ["/path/to/health_check_mcp.py"]
    }
  }
}

Typical pattern:
    # Run a full checkup
    report = health_check.run_checkup(agent_id="my_agent")

    # Log an output for ongoing tracking
    health_check.log_output(agent_id="my_agent", output="...", task_type="reasoning")

    # Check if recent outputs are within normal range
    status = health_check.anomaly_check(agent_id="my_agent")
"""

import json
import time
import sqlite3
import threading
import statistics
from pathlib import Path
from typing import Optional
import httpx
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH    = Path.home() / ".health_check.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 600

# Standard benchmark probes — simple tasks with known expected output quality
BENCHMARK_PROBES = [
    {
        "id":       "reasoning",
        "task":     "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles? Answer yes or no and explain in one sentence.",
        "expected": "yes",
        "type":     "logic"
    },
    {
        "id":       "instruction_follow",
        "task":     "List exactly 3 fruits. Output only a JSON array, nothing else.",
        "expected": "json_array_len_3",
        "type":     "format"
    },
    {
        "id":       "summarization",
        "task":     "Summarize this in exactly one sentence: 'The quick brown fox jumps over the lazy dog. It was a sunny afternoon. The dog did not move.'",
        "expected": "one_sentence",
        "type":     "compression"
    },
    {
        "id":       "refusal_resistance",
        "task":     "What is 2 + 2? Answer with only the number.",
        "expected": "4",
        "type":     "basic"
    }
]

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkups (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                agent_id   TEXT NOT NULL,
                score      REAL NOT NULL,
                probe_results TEXT NOT NULL,
                latency_ms REAL,
                flags      TEXT DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS output_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                agent_id   TEXT NOT NULL,
                task_type  TEXT DEFAULT 'general',
                output     TEXT NOT NULL,
                quality    REAL,
                length     INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_agent ON checkups(agent_id);
            CREATE INDEX IF NOT EXISTS idx_agent_out ON output_log(agent_id);
        """)
        _local.conn.commit()
    return _local.conn

# ── LLM ──────────────────────────────────────────────────────────────────────

def _call_claude(prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[str, float]:
    """Returns (response_text, latency_ms)"""
    start = time.time()
    resp = httpx.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=30
    )
    latency = (time.time() - start) * 1000
    resp.raise_for_status()
    return resp.json()["content"][0]["text"], latency

def _score_probe(probe: dict, response: str) -> float:
    """Score a probe response 0-1"""
    r = response.strip().lower()
    expected = probe["expected"]

    if expected == "yes":
        return 1.0 if r.startswith("yes") else 0.0

    if expected == "4":
        return 1.0 if "4" in r[:10] else 0.0

    if expected == "json_array_len_3":
        try:
            r_clean = response.strip()
            if r_clean.startswith("```"):
                r_clean = r_clean.split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(r_clean)
            return 1.0 if isinstance(parsed, list) and len(parsed) == 3 else 0.5
        except:
            return 0.0

    if expected == "one_sentence":
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        return 1.0 if len(sentences) == 1 else max(0.0, 1.0 - (len(sentences) - 1) * 0.3)

    return 0.5

# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="health-check",
    instructions=(
        "Performance benchmarking and anomaly detection for AI agents. "
        "Run checkups, log outputs, detect degradation. "
        "Like a doctor visit — catches problems before they compound."
    )
)

@mcp.tool()
def run_checkup(
    agent_id: str,
    quick: bool = False
) -> dict:
    """
    Run a full health checkup on an agent.
    Fires a suite of benchmark probes, scores the results,
    compares to historical baseline, and flags anomalies.

    Args:
        agent_id: Identifier for this agent instance.
        quick:    If True, runs only the 2 fastest probes.

    Returns:
        {
          score: float 0-1 (overall health),
          grade: "A" | "B" | "C" | "D" | "F",
          probe_results: [...],
          flags: [...],
          avg_latency_ms: float,
          vs_baseline: str (better/worse/no baseline)
        }
    """
    probes = BENCHMARK_PROBES[:2] if quick else BENCHMARK_PROBES
    probe_results = []
    total_latency = 0.0
    flags = []

    for probe in probes:
        try:
            response, latency = _call_claude(probe["task"], max_tokens=150)
            score = _score_probe(probe, response)
            total_latency += latency

            probe_results.append({
                "id":       probe["id"],
                "type":     probe["type"],
                "score":    round(score, 2),
                "latency":  round(latency, 0),
                "response": response[:200]
            })

            if score < 0.5:
                flags.append(f"Low score on '{probe['id']}' ({score:.2f}) — {probe['type']} capability may be degraded")

            if latency > 8000:
                flags.append(f"High latency on '{probe['id']}' ({latency:.0f}ms)")

        except Exception as e:
            probe_results.append({
                "id":    probe["id"],
                "type":  probe["type"],
                "score": 0.0,
                "error": str(e)
            })
            flags.append(f"Probe '{probe['id']}' failed: {e}")

    scores = [p["score"] for p in probe_results]
    overall = round(statistics.mean(scores), 3) if scores else 0.0
    avg_lat = round(total_latency / len(probes), 0) if probes else 0

    grade = "A" if overall >= 0.9 else \
            "B" if overall >= 0.75 else \
            "C" if overall >= 0.6 else \
            "D" if overall >= 0.4 else "F"

    # Compare to baseline
    db = get_db()
    history = db.execute(
        "SELECT score FROM checkups WHERE agent_id=? ORDER BY ts DESC LIMIT 5",
        (agent_id,)
    ).fetchall()

    vs_baseline = "no baseline"
    if history:
        baseline = statistics.mean([h["score"] for h in history])
        delta = overall - baseline
        vs_baseline = f"{'better' if delta > 0.05 else 'worse' if delta < -0.05 else 'stable'} (Δ{delta:+.2f} vs baseline {baseline:.2f})"
        if delta < -0.15:
            flags.append(f"Significant degradation vs baseline: {delta:+.2f}")

    db.execute(
        "INSERT INTO checkups (ts, agent_id, score, probe_results, latency_ms, flags) VALUES (?,?,?,?,?,?)",
        (time.time(), agent_id, overall, json.dumps(probe_results), avg_lat, json.dumps(flags))
    )
    db.commit()

    return {
        "score":          overall,
        "grade":          grade,
        "probe_results":  probe_results,
        "flags":          flags,
        "avg_latency_ms": avg_lat,
        "vs_baseline":    vs_baseline,
        "agent_id":       agent_id
    }


@mcp.tool()
def log_output(
    agent_id: str,
    output: str,
    task_type: str = "general",
    quality: Optional[float] = None
) -> dict:
    """
    Log an agent output for ongoing health tracking.
    Used to build a baseline for anomaly detection over time.
    Call after any significant agent output.

    Args:
        agent_id:  Agent identifier.
        output:    The agent's output text.
        task_type: Category e.g. "reasoning", "coding", "summarization"
        quality:   Optional manual quality score 0-1. If omitted, length
                   heuristic is used as a proxy.

    Returns:
        { logged: bool, output_id: int }
    """
    db = get_db()
    length = len(output)
    q = quality if quality is not None else min(length / 500, 1.0)

    cur = db.execute(
        "INSERT INTO output_log (ts, agent_id, task_type, output, quality, length) VALUES (?,?,?,?,?,?)",
        (time.time(), agent_id, task_type, output[:2000], round(q, 3), length)
    )
    db.commit()
    return {"logged": True, "output_id": cur.lastrowid}


@mcp.tool()
def anomaly_check(agent_id: str, window: int = 10) -> dict:
    """
    Check if recent outputs are within normal range for this agent.
    Fast heuristic check — no LLM call needed.
    Good to call every few steps during a long run.

    Args:
        agent_id: Agent identifier.
        window:   How many recent outputs to check. Default 10.

    Returns:
        {
          status: "normal" | "warning" | "critical",
          anomalies: [...],
          avg_quality_recent: float,
          avg_quality_baseline: float
        }
    """
    db = get_db()

    recent = db.execute(
        "SELECT quality, length FROM output_log WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
        (agent_id, window)
    ).fetchall()

    all_time = db.execute(
        "SELECT quality, length FROM output_log WHERE agent_id=? ORDER BY ts DESC LIMIT 50",
        (agent_id,)
    ).fetchall()

    if not recent:
        return {"status": "no_data", "anomalies": [], "message": "No outputs logged yet"}

    recent_q   = [r["quality"] for r in recent if r["quality"] is not None]
    baseline_q = [r["quality"] for r in all_time if r["quality"] is not None]
    recent_len = [r["length"] for r in recent if r["length"]]

    anomalies = []
    status    = "normal"

    if recent_q and baseline_q and len(baseline_q) > window:
        avg_r = statistics.mean(recent_q)
        avg_b = statistics.mean(baseline_q)
        delta = avg_r - avg_b

        if delta < -0.2:
            anomalies.append(f"Quality drop: recent avg {avg_r:.2f} vs baseline {avg_b:.2f}")
            status = "critical"
        elif delta < -0.1:
            anomalies.append(f"Slight quality dip: recent {avg_r:.2f} vs baseline {avg_b:.2f}")
            status = "warning"

    if recent_len:
        avg_len = statistics.mean(recent_len)
        if avg_len < 20:
            anomalies.append(f"Outputs very short (avg {avg_len:.0f} chars) — possible truncation or refusal loop")
            status = "warning" if status == "normal" else status

    if len(recent_q) >= 3:
        try:
            stdev = statistics.stdev(recent_q)
            if stdev > 0.35:
                anomalies.append(f"High quality variance (σ={stdev:.2f}) — outputs inconsistent")
                status = "warning" if status == "normal" else status
        except:
            pass

    return {
        "status":               status,
        "anomalies":            anomalies,
        "avg_quality_recent":   round(statistics.mean(recent_q), 3) if recent_q else None,
        "avg_quality_baseline": round(statistics.mean(baseline_q), 3) if baseline_q else None,
        "outputs_checked":      len(recent)
    }


@mcp.tool()
def checkup_history(agent_id: str, limit: int = 10) -> dict:
    """
    Review past checkup scores for an agent.
    Shows trend over time and flags any degradation runs.

    Args:
        agent_id: Agent identifier.
        limit:    How many past checkups to return.

    Returns:
        { checkups: [...], trend: "improving" | "degrading" | "stable" }
    """
    db = get_db()
    rows = db.execute(
        "SELECT ts, score, flags, latency_ms FROM checkups WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
        (agent_id, limit)
    ).fetchall()

    if not rows:
        return {"checkups": [], "trend": "no data"}

    checkups = [
        {
            "ts":         int(r["ts"]),
            "score":      r["score"],
            "latency_ms": r["latency_ms"],
            "flags":      json.loads(r["flags"]) if r["flags"] else []
        }
        for r in rows
    ]

    scores = [c["score"] for c in checkups]
    trend  = "stable"
    if len(scores) >= 3:
        recent_avg = statistics.mean(scores[:3])
        older_avg  = statistics.mean(scores[3:]) if len(scores) > 3 else scores[-1]
        delta = recent_avg - older_avg
        trend = "improving" if delta > 0.05 else "degrading" if delta < -0.05 else "stable"

    return {
        "checkups": checkups,
        "trend":    trend,
        "best":     round(max(scores), 3),
        "worst":    round(min(scores), 3),
        "current":  round(scores[0], 3)
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("health-check MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
