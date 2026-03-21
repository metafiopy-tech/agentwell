"""
self_eval_mcp.py
================
Mid-run self-evaluation for AI agents.
Like eating a snack — small caloric hit, subtle recalibration, doesn't
interrupt the workflow. Agent rates its last N outputs, flags weak
assumptions, and gets a confidence score back before continuing.

Usage:
    python self_eval_mcp.py

Wire into your agent's MCP config:
{
  "mcpServers": {
    "self-eval": {
      "command": "python",
      "args": ["/path/to/self_eval_mcp.py"]
    }
  }
}

Then call mid-run:
    eval = self_eval(outputs=["step 1 result", "step 2 result"])
    if eval["confidence"] < 0.6:
        # recalibrate before continuing
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

DB_PATH    = Path.home() / ".self_eval_history.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 512

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS evals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                run_id      TEXT,
                confidence  REAL,
                weakest     TEXT,
                flags       TEXT,
                raw         TEXT
            )
        """)
        _local.conn.commit()
    return _local.conn

# ── LLM call ─────────────────────────────────────────────────────────────────

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
    name="self-eval",
    instructions=(
        "Mid-run self-evaluation. Call between steps to catch drift, "
        "weak assumptions, and low-confidence reasoning before it compounds. "
        "Like a snack — small, fast, keeps you calibrated."
    )
)

@mcp.tool()
def self_eval(
    outputs: list[str],
    goal: str = "",
    run_id: str = "",
    threshold: float = 0.6
) -> dict:
    """
    Rate the quality of the last N reasoning steps or outputs.
    Returns a confidence score, the weakest step, and any flags.
    Call this between major steps in a long agent run.

    Args:
        outputs:   List of recent outputs/reasoning steps to evaluate.
                   Pass the last 2-5 for best results.
        goal:      Optional — the agent's current objective. Helps
                   the evaluator check for goal drift.
        run_id:    Optional tag to group evals from the same run.
        threshold: Confidence below this triggers a warning. Default 0.6.

    Returns:
        {
          confidence: float 0-1,
          weakest: str (which output was weakest and why),
          flags: list of specific concerns,
          recommendation: "continue" | "recalibrate" | "stop",
          alert: bool (true if confidence < threshold)
        }
    """
    if not outputs:
        return {"error": "No outputs provided to evaluate."}

    numbered = "\n".join(
        f"[{i+1}] {o}" for i, o in enumerate(outputs)
    )
    goal_line = f"\nCurrent goal: {goal}" if goal else ""

    prompt = f"""You are a ruthless quality evaluator for an AI agent mid-run.
{goal_line}

The agent's last {len(outputs)} output(s):
{numbered}

Evaluate these outputs. Respond ONLY with valid JSON, no other text:
{{
  "confidence": <float 0.0-1.0, overall confidence in the reasoning quality>,
  "weakest_index": <int, 1-based index of the weakest output>,
  "weakest_reason": <string, one sentence explaining why it's weakest>,
  "flags": [<list of specific concerns, empty if none>],
  "recommendation": <"continue" | "recalibrate" | "stop">
}}

Be harsh. A confidence of 0.8+ means genuinely solid reasoning.
Flag: hallucination risk, assumption leaps, goal drift, vague conclusions."""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        return {"error": f"Eval failed: {e}", "raw": raw if 'raw' in dir() else ""}

    confidence  = float(result.get("confidence", 0.5))
    weakest_idx = result.get("weakest_index", 1)
    weakest     = f"Output [{weakest_idx}]: {result.get('weakest_reason', '')}"
    flags       = result.get("flags", [])
    rec         = result.get("recommendation", "continue")
    alert       = confidence < threshold

    # Persist
    db = get_db()
    db.execute(
        "INSERT INTO evals (ts, run_id, confidence, weakest, flags, raw) VALUES (?,?,?,?,?,?)",
        (time.time(), run_id, confidence, weakest, json.dumps(flags), raw_clean)
    )
    db.commit()

    return {
        "confidence":      round(confidence, 3),
        "weakest":         weakest,
        "flags":           flags,
        "recommendation":  rec,
        "alert":           alert,
        "alert_message":   f"Confidence {confidence:.2f} below threshold {threshold}" if alert else None,
        "outputs_checked": len(outputs)
    }


@mcp.tool()
def eval_history(
    run_id: str = "",
    limit: int = 10
) -> dict:
    """
    Review past self-evals. Useful at the start of a run to see
    how previous runs performed, or to spot recurring weak points.

    Args:
        run_id: Filter to a specific run. Empty = all runs.
        limit:  Max records to return. Default 10.

    Returns:
        { evals: [...], avg_confidence, low_confidence_count }
    """
    db = get_db()
    if run_id:
        rows = db.execute(
            "SELECT * FROM evals WHERE run_id=? ORDER BY ts DESC LIMIT ?",
            (run_id, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM evals ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()

    evals = [
        {
            "id":         r["id"],
            "ts":         int(r["ts"]),
            "run_id":     r["run_id"],
            "confidence": r["confidence"],
            "weakest":    r["weakest"],
            "flags":      json.loads(r["flags"]) if r["flags"] else []
        }
        for r in rows
    ]

    confidences = [e["confidence"] for e in evals if e["confidence"] is not None]
    avg = round(sum(confidences) / len(confidences), 3) if confidences else None

    return {
        "evals":               evals,
        "avg_confidence":      avg,
        "low_confidence_count": sum(1 for c in confidences if c < 0.6),
        "count":               len(evals)
    }


@mcp.tool()
def quick_check(statement: str) -> dict:
    """
    Fast single-statement confidence check.
    Pass one claim or conclusion — get back whether it's solid.
    The one-liner version of self_eval for casual use mid-step.

    Args:
        statement: A single claim, conclusion, or reasoning step.

    Returns:
        { confidence, concern, safe_to_proceed }
    """
    prompt = f"""Rate the reliability of this single statement from an AI agent:

"{statement}"

Respond ONLY with valid JSON:
{{
  "confidence": <float 0.0-1.0>,
  "concern": <string, one sentence — main risk or null if none>,
  "safe_to_proceed": <bool>
}}"""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        return {"error": f"Check failed: {e}"}

    return {
        "confidence":      round(float(result.get("confidence", 0.5)), 3),
        "concern":         result.get("concern"),
        "safe_to_proceed": result.get("safe_to_proceed", True)
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("self-eval MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
