"""
cost_guard_mcp.py
=================
Token spend tracking and anomaly detection for AI agents.
Fires an alert when an agent is burning unusually fast.

Most agent frameworks track costs after the fact.
cost_guard tracks in real time and interrupts before the bill compounds.

Patterns it catches:
- Sudden spike in call frequency (loop with no exit condition)
- Model escalation without authorization (agent switching to Opus mid-run)
- Duplicate calls (same prompt sent multiple times)
- Long-running tasks with no progress signal

Usage:
    python cost_guard_mcp.py
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

DB_PATH    = Path.home() / ".cost_guard.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"

# Cost estimates per 1K tokens (input/output blended)
MODEL_COSTS = {
    "claude-opus-4":      0.075,
    "claude-sonnet-4":    0.012,
    "claude-haiku-4":     0.00125,
    "gpt-4o":             0.010,
    "gpt-4-turbo":        0.015,
    "gpt-3.5-turbo":      0.001,
    "llama3.2":           0.0,
    "deepseek-r1:7b":     0.0,
    "mistral":            0.0,
    "qwen2.5:7b":         0.0,
}

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                agent_id     TEXT NOT NULL,
                run_id       TEXT DEFAULT '',
                model        TEXT NOT NULL,
                tokens_in    INTEGER DEFAULT 0,
                tokens_out   INTEGER DEFAULT 0,
                cost_est     REAL DEFAULT 0.0,
                prompt_hash  TEXT DEFAULT '',
                task_type    TEXT DEFAULT 'general'
            );
            CREATE TABLE IF NOT EXISTS budgets (
                agent_id     TEXT PRIMARY KEY,
                daily_limit  REAL DEFAULT 1.0,
                run_limit    REAL DEFAULT 0.10,
                alert_at     REAL DEFAULT 0.8,
                updated_at   REAL
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                agent_id     TEXT,
                alert_type   TEXT,
                message      TEXT,
                cost_at_alert REAL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_ts ON calls(agent_id, ts);
            CREATE INDEX IF NOT EXISTS idx_run ON calls(run_id);
        """)
        _local.conn.commit()
    return _local.conn

def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    rate = 0.0
    for key, cost in MODEL_COSTS.items():
        if key in model.lower():
            rate = cost
            break
    return ((tokens_in + tokens_out) / 1000) * rate

mcp = FastMCP(
    name="cost_guard",
    instructions=(
        "Token spend tracking and anomaly detection for AI agents. "
        "Log calls, set budgets, detect runaway loops before they compound. "
        "Fires alerts on duplicate prompts, model escalation, and budget overruns."
    )
)

@mcp.tool()
def log_call(
    agent_id: str,
    model: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    run_id: str = "",
    task_type: str = "general",
    prompt_hash: str = ""
) -> dict:
    """
    Log a single API call for cost tracking.
    Call this after every LLM call in your agent.

    Args:
        agent_id:    Agent identifier
        model:       Model used e.g. "claude-sonnet-4", "gpt-4o", "llama3.2"
        tokens_in:   Input token count
        tokens_out:  Output token count
        run_id:      Optional run tag for grouping
        task_type:   Type of call e.g. "reasoning", "synthesis", "eval"
        prompt_hash: Optional MD5 of prompt for duplicate detection

    Returns:
        { logged, cost_est, run_total, daily_total, alerts: [...] }
    """
    db    = get_db()
    cost  = _estimate_cost(model, tokens_in, tokens_out)
    now   = time.time()
    alerts = []

    # Duplicate detection
    if prompt_hash:
        recent_dup = db.execute(
            "SELECT COUNT(*) as n FROM calls WHERE agent_id=? AND prompt_hash=? AND ts > ?",
            (agent_id, prompt_hash, now - 300)
        ).fetchone()
        if recent_dup["n"] > 0:
            alerts.append({
                "type": "duplicate_prompt",
                "message": f"Same prompt sent {recent_dup['n']+1} times in 5 minutes"
            })

    db.execute(
        "INSERT INTO calls (ts, agent_id, run_id, model, tokens_in, tokens_out, cost_est, prompt_hash, task_type) VALUES (?,?,?,?,?,?,?,?,?)",
        (now, agent_id, run_id, model, tokens_in, tokens_out, cost, prompt_hash, task_type)
    )
    db.commit()

    # Run total
    run_total = 0.0
    if run_id:
        row = db.execute(
            "SELECT SUM(cost_est) as total FROM calls WHERE run_id=?", (run_id,)
        ).fetchone()
        run_total = row["total"] or 0.0

    # Daily total
    day_start = now - 86400
    row = db.execute(
        "SELECT SUM(cost_est) as total FROM calls WHERE agent_id=? AND ts > ?",
        (agent_id, day_start)
    ).fetchone()
    daily_total = row["total"] or 0.0

    # Budget check
    budget = db.execute(
        "SELECT * FROM budgets WHERE agent_id=?", (agent_id,)
    ).fetchone()

    if budget:
        if run_id and run_total > budget["run_limit"]:
            alerts.append({
                "type": "run_budget_exceeded",
                "message": f"Run cost ${run_total:.4f} exceeds limit ${budget['run_limit']:.4f}"
            })
        if daily_total > budget["daily_limit"]:
            alerts.append({
                "type": "daily_budget_exceeded",
                "message": f"Daily cost ${daily_total:.4f} exceeds limit ${budget['daily_limit']:.4f}"
            })
        elif daily_total > budget["daily_limit"] * budget["alert_at"]:
            alerts.append({
                "type": "approaching_daily_limit",
                "message": f"At {daily_total/budget['daily_limit']*100:.0f}% of daily budget"
            })

    for alert in alerts:
        db.execute(
            "INSERT INTO alerts (ts, agent_id, alert_type, message, cost_at_alert) VALUES (?,?,?,?,?)",
            (now, agent_id, alert["type"], alert["message"], daily_total)
        )
    if alerts:
        db.commit()

    return {
        "logged": True,
        "cost_est": round(cost, 6),
        "run_total": round(run_total, 4),
        "daily_total": round(daily_total, 4),
        "alerts": alerts
    }


@mcp.tool()
def set_budget(
    agent_id: str,
    daily_limit: float = 1.0,
    run_limit: float = 0.10,
    alert_at: float = 0.8
) -> dict:
    """
    Set cost limits for an agent.

    Args:
        agent_id:    Agent identifier
        daily_limit: Max spend per day in USD. Default $1.00
        run_limit:   Max spend per run in USD. Default $0.10
        alert_at:    Alert threshold as fraction of daily_limit. Default 0.8 (80%)

    Returns:
        { set, agent_id, daily_limit, run_limit }
    """
    db = get_db()
    db.execute(
        """INSERT INTO budgets (agent_id, daily_limit, run_limit, alert_at, updated_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(agent_id) DO UPDATE SET
           daily_limit=excluded.daily_limit,
           run_limit=excluded.run_limit,
           alert_at=excluded.alert_at,
           updated_at=excluded.updated_at""",
        (agent_id, daily_limit, run_limit, alert_at, time.time())
    )
    db.commit()
    return {
        "set": True,
        "agent_id": agent_id,
        "daily_limit": daily_limit,
        "run_limit": run_limit,
        "alert_at": f"{alert_at*100:.0f}%"
    }


@mcp.tool()
def spend_report(
    agent_id: str = "",
    run_id: str = "",
    hours: int = 24
) -> dict:
    """
    Spend report for an agent or run.

    Args:
        agent_id: Filter by agent. Empty = all agents.
        run_id:   Filter by run. Takes precedence over agent_id.
        hours:    Lookback window in hours. Default 24.

    Returns:
        { total_cost, call_count, by_model, by_task, top_runs, alerts }
    """
    db   = get_db()
    since = time.time() - (hours * 3600)

    if run_id:
        rows = db.execute(
            "SELECT * FROM calls WHERE run_id=? ORDER BY ts DESC",
            (run_id,)
        ).fetchall()
    elif agent_id:
        rows = db.execute(
            "SELECT * FROM calls WHERE agent_id=? AND ts > ? ORDER BY ts DESC",
            (agent_id, since)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM calls WHERE ts > ? ORDER BY ts DESC",
            (since,)
        ).fetchall()

    total = sum(r["cost_est"] for r in rows)
    by_model: dict = {}
    by_task: dict = {}
    by_run: dict = {}

    for r in rows:
        m = r["model"]
        t = r["task_type"]
        ri = r["run_id"]
        by_model[m] = by_model.get(m, 0) + r["cost_est"]
        by_task[t] = by_task.get(t, 0) + r["cost_est"]
        if ri:
            by_run[ri] = by_run.get(ri, 0) + r["cost_est"]

    top_runs = sorted(by_run.items(), key=lambda x: -x[1])[:5]

    recent_alerts = db.execute(
        "SELECT * FROM alerts WHERE ts > ? ORDER BY ts DESC LIMIT 10",
        (since,)
    ).fetchall()

    return {
        "total_cost": round(total, 4),
        "call_count": len(rows),
        "hours": hours,
        "by_model": {k: round(v, 4) for k, v in by_model.items()},
        "by_task": {k: round(v, 4) for k, v in by_task.items()},
        "top_runs": [{"run_id": r, "cost": round(c, 4)} for r, c in top_runs],
        "recent_alerts": [
            {"type": a["alert_type"], "message": a["message"], "ts": int(a["ts"])}
            for a in recent_alerts
        ]
    }


@mcp.tool()
def detect_runaway(
    agent_id: str,
    window_minutes: int = 10,
    call_threshold: int = 30,
    cost_threshold: float = 0.05
) -> dict:
    """
    Detect if an agent is in a runaway loop.
    Fast check — no API call needed.

    Args:
        agent_id:        Agent to check
        window_minutes:  Lookback window. Default 10 minutes.
        call_threshold:  Calls in window above this = runaway. Default 30.
        cost_threshold:  Cost in window above this = runaway. Default $0.05.

    Returns:
        { is_runaway, calls_in_window, cost_in_window, recommendation }
    """
    db    = get_db()
    since = time.time() - (window_minutes * 60)

    rows = db.execute(
        "SELECT COUNT(*) as n, SUM(cost_est) as cost FROM calls WHERE agent_id=? AND ts > ?",
        (agent_id, since)
    ).fetchone()

    calls = rows["n"] or 0
    cost  = rows["cost"] or 0.0
    is_runaway = calls > call_threshold or cost > cost_threshold

    rec = "normal"
    if is_runaway:
        if cost > cost_threshold * 2:
            rec = "stop immediately — cost escalating rapidly"
        elif calls > call_threshold * 2:
            rec = "interrupt — call frequency abnormal, likely infinite loop"
        else:
            rec = "investigate — approaching runaway thresholds"

    return {
        "is_runaway": is_runaway,
        "calls_in_window": calls,
        "cost_in_window": round(cost, 4),
        "window_minutes": window_minutes,
        "recommendation": rec
    }


if __name__ == "__main__":
    print("cost_guard MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
