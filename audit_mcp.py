"""
audit_mcp.py
============
Adversarial blind spot scanner for AI agents.
Like looking in a mirror — surfaces what the agent can't see about itself.
Generates a red-team challenge against the agent's own reasoning,
finds the weakest assumptions, and stress-tests conclusions.

Usage:
    python audit_mcp.py

Wire into MCP config:
{
  "mcpServers": {
    "audit": {
      "command": "python",
      "args": ["/path/to/audit_mcp.py"]
    }
  }
}

Typical use:
    # Before committing to a plan or conclusion
    result = audit.scan(reasoning="my plan is X because Y and Z")
    # result["vulnerabilities"] = what a critic would attack
    # result["strongest_challenge"] = the one most likely to be right
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

DB_PATH    = Path.home() / ".audit_history.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 700

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("""
            CREATE TABLE IF NOT EXISTS audits (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                 REAL NOT NULL,
                run_id             TEXT,
                reasoning_preview  TEXT,
                vulnerability_count INTEGER,
                severity           TEXT,
                strongest_challenge TEXT,
                raw                TEXT
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
    name="audit",
    instructions=(
        "Adversarial blind spot scanner. Finds what the agent is confidently "
        "wrong about. Red-teams reasoning chains, stress-tests assumptions, "
        "surfaces vulnerabilities before they become failures."
    )
)

@mcp.tool()
def scan(
    reasoning: str,
    context: str = "",
    goal: str = "",
    run_id: str = "",
    severity_threshold: str = "medium"
) -> dict:
    """
    Red-team the agent's own reasoning.
    Finds the weakest assumptions, generates the strongest challenges,
    and returns a prioritized list of vulnerabilities.

    Args:
        reasoning:          The agent's current reasoning chain or plan.
        context:            Optional background context.
        goal:               What the agent is trying to accomplish.
        run_id:             Optional run tag.
        severity_threshold: Only return issues at or above this level.
                            "low" | "medium" | "high". Default "medium".

    Returns:
        {
          vulnerabilities: [...],
          strongest_challenge: str,
          severity: "low" | "medium" | "high" | "critical",
          safe_to_proceed: bool,
          recommendations: [...]
        }
    """
    if not reasoning:
        return {"error": "No reasoning provided to audit."}

    ctx_line  = f"\nContext: {context}" if context else ""
    goal_line = f"\nGoal: {goal}" if goal else ""

    prompt = f"""You are a ruthless adversarial auditor for an AI agent.
Your job is to find everything that could be wrong with the following reasoning.{goal_line}{ctx_line}

Agent reasoning to audit:
---
{reasoning[:1500]}
---

Find every vulnerability. Be adversarial. Assume the agent is confidently wrong somewhere.
Respond ONLY with valid JSON:
{{
  "vulnerabilities": [
    {{
      "type": <"assumption" | "logic_gap" | "missing_info" | "overconfidence" | "scope_creep" | "other">,
      "description": <string, one clear sentence>,
      "severity": <"low" | "medium" | "high" | "critical">,
      "evidence": <string, quote or reference from the reasoning that shows this>
    }}
  ],
  "strongest_challenge": <string, the single most damaging critique — 2-3 sentences>,
  "overall_severity": <"low" | "medium" | "high" | "critical">,
  "safe_to_proceed": <bool>,
  "recommendations": [<list of 1-3 concrete fixes, each one sentence>]
}}

If the reasoning is genuinely solid, say so — but still look hard."""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        return {"error": f"Audit failed: {e}"}

    sev_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    threshold_rank = sev_rank.get(severity_threshold, 1)

    all_vulns     = result.get("vulnerabilities", [])
    filtered_vulns = [
        v for v in all_vulns
        if sev_rank.get(v.get("severity", "low"), 0) >= threshold_rank
    ]

    overall_sev       = result.get("overall_severity", "medium")
    strongest         = result.get("strongest_challenge", "")
    safe              = result.get("safe_to_proceed", True)
    recommendations   = result.get("recommendations", [])

    db = get_db()
    db.execute(
        "INSERT INTO audits (ts, run_id, reasoning_preview, vulnerability_count, severity, strongest_challenge, raw) VALUES (?,?,?,?,?,?,?)",
        (time.time(), run_id, reasoning[:300], len(all_vulns), overall_sev, strongest, raw_clean)
    )
    db.commit()

    return {
        "vulnerabilities":    filtered_vulns,
        "all_vulnerabilities": len(all_vulns),
        "strongest_challenge": strongest,
        "severity":            overall_sev,
        "safe_to_proceed":     safe,
        "recommendations":     recommendations
    }


@mcp.tool()
def assumption_extract(text: str) -> dict:
    """
    Extract every assumption embedded in a piece of reasoning.
    No evaluation — just surfaces them so the agent can see
    what it's taking for granted.

    Args:
        text: Any reasoning, plan, or conclusion.

    Returns:
        { assumptions: [...], hidden_count: int }
    """
    prompt = f"""Extract every assumption — explicit and hidden — from this text:

---
{text[:1000]}
---

Respond ONLY with valid JSON:
{{
  "explicit_assumptions": [<assumptions the author stated openly>],
  "hidden_assumptions": [<things the author assumed without saying>],
  "most_dangerous": <string, which single assumption is most likely to be wrong>
}}"""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        return {"error": f"Extraction failed: {e}"}

    explicit = result.get("explicit_assumptions", [])
    hidden   = result.get("hidden_assumptions", [])

    return {
        "explicit_assumptions": explicit,
        "hidden_assumptions":   hidden,
        "most_dangerous":       result.get("most_dangerous", ""),
        "total_count":          len(explicit) + len(hidden),
        "hidden_count":         len(hidden)
    }


@mcp.tool()
def stress_test(
    conclusion: str,
    n_challenges: int = 3
) -> dict:
    """
    Generate N adversarial challenges against a single conclusion.
    Like sparring — makes the conclusion stronger or reveals it's wrong.

    Args:
        conclusion:   A specific claim or conclusion to stress-test.
        n_challenges: Number of challenges to generate. Default 3, max 5.

    Returns:
        { challenges: [...], verdict: "holds" | "weak" | "fails" }
    """
    n = min(max(n_challenges, 1), 5)

    prompt = f"""Generate {n} adversarial challenges against this conclusion:

"{conclusion}"

Each challenge should try to break the conclusion from a different angle.
Respond ONLY with valid JSON:
{{
  "challenges": [
    {{
      "angle": <string, what perspective this challenge comes from>,
      "challenge": <string, the actual challenge in 1-2 sentences>,
      "severity": <"weak" | "moderate" | "strong">
    }}
  ],
  "verdict": <"holds" — conclusion survives challenges | "weak" — has real problems | "fails" — conclusion is likely wrong>
}}"""

    try:
        raw = _call_claude(prompt)
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(raw_clean)
    except Exception as e:
        return {"error": f"Stress test failed: {e}"}

    return {
        "challenges": result.get("challenges", []),
        "verdict":    result.get("verdict", "weak"),
        "conclusion": conclusion[:200]
    }


@mcp.tool()
def audit_history(limit: int = 10) -> dict:
    """
    Review past audits. Useful for spotting recurring blind spots
    across runs — if the same vulnerability type keeps showing up,
    that's a systemic issue worth fixing at the prompt level.

    Returns:
        { audits: [...], recurring_types: dict }
    """
    db   = get_db()
    rows = db.execute(
        "SELECT ts, run_id, vulnerability_count, severity, strongest_challenge FROM audits ORDER BY ts DESC LIMIT ?",
        (limit,)
    ).fetchall()

    audits = [
        {
            "ts":                  int(r["ts"]),
            "run_id":              r["run_id"],
            "vulnerability_count": r["vulnerability_count"],
            "severity":            r["severity"],
            "strongest_challenge": r["strongest_challenge"]
        }
        for r in rows
    ]

    sev_counts: dict = {}
    for a in audits:
        s = a["severity"] or "unknown"
        sev_counts[s] = sev_counts.get(s, 0) + 1

    return {
        "audits":           audits,
        "severity_distribution": sev_counts,
        "total_audits":     len(audits)
    }


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("audit MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
