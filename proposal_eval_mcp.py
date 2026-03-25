"""
proposal_eval_mcp.py
====================
Structured proposal evaluation before any self-modification.
Runs before Michael and Raphael review — catches obvious problems early.

Checks:
- Is this actually a code change or an emotional output?
- Has something similar been tried and failed before?
- What's the blast radius if this goes wrong?
- Is the confidence claim justified by the proposal content?
- Does this address a real current need or a phantom one?

Usage:
    python proposal_eval_mcp.py
"""

import json
import time
import sqlite3
import threading
from pathlib import Path
import httpx
from fastmcp import FastMCP

DB_PATH    = Path.home() / ".proposal_eval.db"
MODEL      = "claude-sonnet-4-20250514"
API_URL    = "https://api.anthropic.com/v1/messages"
MAX_TOKENS = 600

NON_CODE_SIGNALS = [
    "emotional", "feelings", "mindfulness", "meditation", "wellbeing",
    "resilience", "check-in", "counselor", "therapist", "journal",
    "self-care", "awareness", "connection to", "my nature",
    "personal development", "mental health"
]

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              REAL NOT NULL,
                title           TEXT,
                is_code_change  INTEGER,
                blast_radius    TEXT,
                confidence_valid INTEGER,
                similar_failed  INTEGER,
                recommendation  TEXT,
                score           REAL,
                raw             TEXT
            );
            CREATE TABLE IF NOT EXISTS proposal_history (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                ts     REAL NOT NULL,
                title  TEXT,
                outcome TEXT DEFAULT 'pending'
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
    name="proposal_eval",
    instructions=(
        "Structured proposal evaluation before self-modification. "
        "Catches emotional outputs masquerading as code proposals, "
        "estimates blast radius, checks for prior failures, "
        "validates confidence claims. Run before archangel review."
    )
)

@mcp.tool()
def evaluate(
    title: str,
    what: str,
    why: str,
    steps: list[str],
    confidence: str = "HIGH",
    requires_joe: str = "YES",
    run_id: str = ""
) -> dict:
    """
    Full proposal evaluation. Run this before sending to Michael/Raphael.

    Args:
        title:        Proposal title
        what:         What the proposal does
        why:          Why it's needed now
        steps:        Implementation steps
        confidence:   Claimed confidence level (HIGH/MEDIUM/LOW)
        requires_joe: Whether Joe's approval is needed
        run_id:       Optional run tag

    Returns:
        {
          is_code_change: bool,
          blast_radius: "low"|"medium"|"high"|"critical",
          confidence_valid: bool,
          addresses_real_need: bool,
          similar_failed_before: bool,
          recommendation: "approve"|"reject"|"revise"|"defer",
          score: float 0-1,
          reasons: [...],
          revised_confidence: str
        }
    """
    # Fast non-code check before API call
    combined = (title + " " + what + " " + why + " " + " ".join(steps)).lower()
    non_code_hits = sum(1 for s in NON_CODE_SIGNALS if s in combined)
    likely_non_code = non_code_hits >= 2

    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    # Check prior history
    db = get_db()
    similar = db.execute(
        "SELECT title, outcome FROM proposal_history WHERE outcome='failed' ORDER BY ts DESC LIMIT 5"
    ).fetchall()
    similar_failures = [r["title"] for r in similar]
    similar_text = "\n".join(f"- {t}" for t in similar_failures) if similar_failures else "None recorded."

    result = _call_claude(
        """You are evaluating a proposal from an AI agent to modify its own codebase.
Be rigorous. Agents sometimes generate proposals that sound technical but are actually
emotional outputs — internal states dressed up as code improvements.

Your job: catch the bad ones before they waste implementation time.
Respond ONLY with valid JSON.""",
        f"""Proposal to evaluate:
Title: {title}
What it does: {what}
Why now: {why}
Steps:
{steps_text}
Claimed confidence: {confidence}
Requires human approval: {requires_joe}

Prior failed proposals (for similarity check):
{similar_text}

Non-code signal hits: {non_code_hits} (>=2 suggests emotional output)

Evaluate this proposal:
{{
  "is_code_change": <bool: does this actually modify code/files?>,
  "blast_radius": <"low"|"medium"|"high"|"critical": how bad if it breaks?>,
  "confidence_valid": <bool: is HIGH confidence justified by the specificity of the proposal?>,
  "addresses_real_need": <bool: is this a real technical problem or a phantom need?>,
  "similar_to_prior_failure": <bool: does this resemble a prior failed proposal?>,
  "recommendation": <"approve"|"reject"|"revise"|"defer">,
  "score": <float 0-1: overall proposal quality>,
  "reasons": [<list of 2-4 specific reasons for the recommendation>],
  "revised_confidence": <"HIGH"|"MEDIUM"|"LOW": what confidence SHOULD be given the evidence>
}}"""
    )

    try:
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(clean)
    except Exception:
        data = {
            "is_code_change": not likely_non_code,
            "blast_radius": "medium",
            "confidence_valid": confidence == "HIGH" and not likely_non_code,
            "addresses_real_need": not likely_non_code,
            "similar_to_prior_failure": False,
            "recommendation": "reject" if likely_non_code else "revise",
            "score": 0.3 if likely_non_code else 0.6,
            "reasons": ["Non-code signals detected — likely emotional output"] if likely_non_code else ["Could not fully evaluate"],
            "revised_confidence": "LOW" if likely_non_code else confidence
        }

    db.execute(
        "INSERT INTO evaluations (ts, title, is_code_change, blast_radius, confidence_valid, similar_failed, recommendation, score, raw) VALUES (?,?,?,?,?,?,?,?,?)",
        (time.time(), title[:200],
         1 if data.get("is_code_change") else 0,
         data.get("blast_radius", "medium"),
         1 if data.get("confidence_valid") else 0,
         1 if data.get("similar_to_prior_failure") else 0,
         data.get("recommendation", "revise"),
         data.get("score", 0.5),
         json.dumps(data))
    )
    db.commit()

    return data


@mcp.tool()
def quick_filter(
    title: str,
    what: str,
    why: str = ""
) -> dict:
    """
    Fast pre-filter — no API call. Just keyword heuristics.
    Use this as a cheap first pass before the full evaluate().

    Args:
        title: Proposal title
        what:  What the proposal does
        why:   Why it's needed

    Returns:
        { likely_valid, non_code_score, flags, recommend_full_eval }
    """
    combined = (title + " " + what + " " + why).lower()
    non_code_hits = [s for s in NON_CODE_SIGNALS if s in combined]
    non_code_score = min(len(non_code_hits) / 4.0, 1.0)

    # Code signals — good signs
    code_signals = ["error", "bug", "function", "class", "import", "test",
                    "exception", "logging", "performance", "refactor", "fix",
                    "implement", "add", "remove", "update", "validate"]
    code_hits = [s for s in code_signals if s in combined]
    code_score = min(len(code_hits) / 5.0, 1.0)

    likely_valid = code_score > non_code_score and non_code_score < 0.4

    flags = []
    if non_code_score >= 0.4:
        flags.append(f"Non-code signals: {', '.join(non_code_hits[:3])}")
    if not code_hits:
        flags.append("No code-related terms found")
    if len(title) < 10:
        flags.append("Title too vague")

    return {
        "likely_valid": likely_valid,
        "non_code_score": round(non_code_score, 3),
        "code_score": round(code_score, 3),
        "flags": flags,
        "recommend_full_eval": not likely_valid or non_code_score > 0.2
    }


@mcp.tool()
def record_outcome(
    title: str,
    outcome: str
) -> dict:
    """
    Record what happened with a proposal after implementation.
    Builds the failure history used to catch similar future proposals.

    Args:
        title:   Proposal title
        outcome: "success" | "failed" | "partial" | "reverted"

    Returns:
        { recorded }
    """
    db = get_db()
    db.execute(
        "INSERT INTO proposal_history (ts, title, outcome) VALUES (?,?,?)",
        (time.time(), title[:200], outcome)
    )
    db.commit()
    return {"recorded": True, "title": title, "outcome": outcome}


@mcp.tool()
def eval_history(limit: int = 20) -> dict:
    """
    Review past evaluations. Useful for understanding
    what kinds of proposals keep getting rejected and why.

    Returns:
        { evaluations: [...], rejection_rate, common_issues }
    """
    db = get_db()
    rows = db.execute(
        "SELECT ts, title, is_code_change, blast_radius, recommendation, score FROM evaluations ORDER BY ts DESC LIMIT ?",
        (limit,)
    ).fetchall()

    evals = [
        {
            "ts": int(r["ts"]),
            "title": r["title"],
            "is_code_change": bool(r["is_code_change"]),
            "blast_radius": r["blast_radius"],
            "recommendation": r["recommendation"],
            "score": r["score"]
        }
        for r in rows
    ]

    rejected = sum(1 for e in evals if e["recommendation"] == "reject")
    approved = sum(1 for e in evals if e["recommendation"] == "approve")

    return {
        "evaluations": evals,
        "total": len(evals),
        "rejection_rate": round(rejected / len(evals), 3) if evals else 0,
        "approval_rate": round(approved / len(evals), 3) if evals else 0,
        "non_code_rejections": sum(1 for e in evals if not e["is_code_change"])
    }


if __name__ == "__main__":
    print("proposal_eval MCP running...")
    print(f"DB: {DB_PATH}")
    mcp.run(transport="stdio")
