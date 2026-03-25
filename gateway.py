"""
AgentWell Gateway
=================
FastAPI server that wraps all 9 wellness MCPs behind a single
authenticated API. Handles key validation, per-call metering,
and Stripe billing webhooks.

Run:
    uvicorn gateway:app --host 0.0.0.0 --port 8000

Env vars needed (.env file):
    ANTHROPIC_API_KEY=sk-ant-...
    STRIPE_SECRET_KEY=sk_live_...
    STRIPE_WEBHOOK_SECRET=whsec_...
    AGENTWELL_ADMIN_KEY=your-secret-admin-key
"""

import os
import time
import uuid
import json
import sqlite3
import hashlib
try:
    import psycopg2
    import psycopg2.extras
    HAS_PG = True
except ImportError:
    HAS_PG = False
import threading
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import stripe
import httpx
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
ADMIN_KEY      = os.getenv("AGENTWELL_ADMIN_KEY", "admin-dev-key")
DATABASE_URL   = os.getenv("DATABASE_URL", "")

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path.home())))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH  = DATA_DIR / "agentwell_gateway.db"

PLANS = {
    "payg":    {"calls_per_month": 0,       "price_per_call": 0.002, "monthly_fee": 0},
    "dev":     {"calls_per_month": 20_000,  "price_per_call": 0.0,   "monthly_fee": 29},
    "agency":  {"calls_per_month": 999_999, "price_per_call": 0.0,   "monthly_fee": 199},
    "free":    {"calls_per_month": 100,     "price_per_call": 0.0,   "monthly_fee": 0},
}

TOOLS = [
    "token_offload", "self_eval", "ground", "sleep",
    "health_check", "audit", "handshake", "journal", "spike",
    "checkpoint", "risk_register"
]

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()
_pg_conn = None
_pg_lock = threading.Lock()

def get_db():
    """Returns a Postgres connection if DATABASE_URL set, else SQLite fallback."""
    global _pg_conn
    if DATABASE_URL and HAS_PG:
        with _pg_lock:
            try:
                if _pg_conn is None or _pg_conn.closed:
                    raise Exception("reconnect")
                _pg_conn.isolation_level  # ping
            except Exception:
                _pg_conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
                _pg_conn.autocommit = True
                with _pg_conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS api_keys (
                            key_hash           TEXT PRIMARY KEY,
                            key_prefix         TEXT NOT NULL,
                            plan               TEXT DEFAULT 'free',
                            calls_used         INTEGER DEFAULT 0,
                            calls_limit        INTEGER DEFAULT 100,
                            stripe_customer_id TEXT,
                            stripe_sub_id      TEXT,
                            created_at         REAL NOT NULL,
                            active             INTEGER DEFAULT 1,
                            email              TEXT DEFAULT ''
                        )
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS call_log (
                            id         SERIAL PRIMARY KEY,
                            ts         REAL NOT NULL,
                            key_prefix TEXT NOT NULL,
                            tool       TEXT NOT NULL,
                            status     INTEGER DEFAULT 200,
                            latency_ms REAL
                        )
                    """)
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_key ON call_log(key_prefix)")
                    cur.execute("CREATE INDEX IF NOT EXISTS idx_ts  ON call_log(ts)")
        return _pg_conn
    # SQLite fallback for local dev
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash      TEXT PRIMARY KEY,
                key_prefix    TEXT NOT NULL,
                plan          TEXT DEFAULT 'free',
                calls_used    INTEGER DEFAULT 0,
                calls_limit   INTEGER DEFAULT 100,
                stripe_customer_id TEXT,
                stripe_sub_id TEXT,
                created_at    REAL NOT NULL,
                active        INTEGER DEFAULT 1,
                email         TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS call_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL NOT NULL,
                key_prefix TEXT NOT NULL,
                tool       TEXT NOT NULL,
                status     INTEGER DEFAULT 200,
                latency_ms REAL
            );
            CREATE INDEX IF NOT EXISTS idx_key ON call_log(key_prefix);
            CREATE INDEX IF NOT EXISTS idx_ts  ON call_log(ts);
        """)
        _local.conn.commit()
    return _local.conn

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def generate_key() -> str:
    return "aw_" + uuid.uuid4().hex

def get_key_record(raw_key: str):
    db = get_db()
    h  = hash_key(raw_key)
    with db.cursor() as cur:
        if DATABASE_URL and HAS_PG:
            cur.execute("SELECT * FROM api_keys WHERE key_hash=%s AND active=1", (h,))
        else:
            cur.execute("SELECT * FROM api_keys WHERE key_hash=? AND active=1", (h,))
        return cur.fetchone()

def increment_calls(key_prefix: str, tool: str, status: int, latency: float):
    db = get_db()
    with db.cursor() as cur:
        cur.execute("UPDATE api_keys SET calls_used = calls_used + 1 WHERE key_prefix=%s", (key_prefix,))
        cur.execute("INSERT INTO call_log (ts, key_prefix, tool, status, latency_ms) VALUES (%s,%s,%s,%s,%s)",
                    (time.time(), key_prefix, tool, status, latency))

# ── Auth ──────────────────────────────────────────────────────────────────────

async def require_key(x_api_key: str = Header(...)):
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header")
    record = get_key_record(x_api_key)
    if not record:
        raise HTTPException(401, "Invalid API key")
    plan = PLANS.get(record["plan"], PLANS["free"])
    limit = record["calls_limit"]
    if limit > 0 and record["calls_used"] >= limit:
        raise HTTPException(429, f"Call limit reached for plan '{record['plan']}'. Upgrade at agentwell.dev/upgrade")
    return record

async def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")

# ── MCP proxy ─────────────────────────────────────────────────────────────────

async def call_anthropic(system: str, prompt: str, max_tokens: int = 800) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = get_db()
    test_key = "aw_testkey_dev123"
    h = hash_key(test_key)
    if DATABASE_URL and HAS_PG:
        with db.cursor() as cur:
            cur.execute("SELECT key_hash FROM api_keys WHERE key_hash=%s", (h,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(
                    "INSERT INTO api_keys (key_hash, key_prefix, plan, calls_limit, created_at, email) VALUES (%s,%s,%s,%s,%s,%s)",
                    (h, "aw_testk", "free", 100, time.time(), "dev@agentwell.dev")
                )
        print(f"Postgres connected. Dev test key: {test_key}")
    else:
        print(f"SQLite mode. Dev test key: {test_key}")
    yield

app = FastAPI(
    title="AgentWell API",
    description="Cognitive wellness infrastructure for AI agents. One key, nine tools, growing library.",
    version="2.0.0",
    lifespan=lifespan
)

from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "error": "validation_error",
        "message": "Invalid request format",
        "details": exc.errors(),
        "support": "github.com/metafiopy-tech/agentwell/issues"
    })

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": "internal_error",
        "message": "Something went wrong. Please try again.",
        "type": type(exc).__name__,
        "support": "github.com/metafiopy-tech/agentwell/issues"
    })

# ── Tool endpoints ────────────────────────────────────────────────────────────

class ToolRequest(BaseModel):
    tool: str
    params: dict = {}

@app.post("/v1/call")
async def call_tool(req: ToolRequest, record=Depends(require_key)):
    """
    Universal tool call endpoint.
    Pass tool name and params, get result back.
    Metered per call.
    """
    if req.tool not in TOOLS:
        raise HTTPException(400, f"Unknown tool '{req.tool}'. Available: {TOOLS}")

    start = time.time()
    key_prefix = record["key_prefix"]

    try:
        result = await _dispatch(req.tool, req.params)
        latency = (time.time() - start) * 1000
        increment_calls(key_prefix, req.tool, 200, latency)
        return {"tool": req.tool, "result": result, "latency_ms": round(latency, 1)}

    except HTTPException:
        raise
    except Exception as e:
        latency = (time.time() - start) * 1000
        increment_calls(key_prefix, req.tool, 500, latency)
        error_type = type(e).__name__
        raise HTTPException(500, {
            "error": "tool_error",
            "tool": req.tool,
            "message": str(e),
            "type": error_type,
            "support": "github.com/metafiopy-tech/agentwell/issues"
        })


async def _dispatch(tool: str, params: dict) -> dict:
    """Route tool calls to their handlers."""

    if tool == "token_offload":
        action = params.get("action", "store")
        if action == "store":
            return _token_store(params)
        elif action == "retrieve":
            return _token_retrieve(params)
        elif action == "search":
            return _token_search(params)
        elif action == "status":
            return _token_status()

    elif tool == "self_eval":
        outputs  = params.get("outputs", [])
        goal     = params.get("goal", "")
        history  = params.get("history", "")   # optional: prior steps summary
        mode     = params.get("mode", "auto")  # "pair" | "step" | "auto"
        if not outputs:
            return {"error": "outputs required"}

        # Auto-detect mode: single output = score_step, multiple = pair eval
        if mode == "auto":
            mode = "step" if len(outputs) == 1 else "pair"

        if mode == "step":
            # Score a single step against the full goal + run history
            step_text = outputs[0]
            history_line = f"\nPrior steps summary: {history[:600]}" if history else ""
            prompt = f"""You are evaluating a single step in a multi-step agent run.
Goal: {goal or 'unspecified'}{history_line}

This step's output:
{step_text[:800]}

Score this step. Respond ONLY with JSON:
{{"confidence": 0.0-1.0,
  "is_solid": bool,
  "weakness": "one sentence — the specific gap in this step, or null if solid",
  "flags": ["list of specific concerns, empty if none"],
  "recommendation": "continue|recalibrate|stop",
  "note": "one sentence of actionable guidance for the next step"}}"""
            raw    = await call_anthropic("You are a precise step-level quality evaluator for AI agents. Be harsh. A solid step earns 0.8+. Flag vague outputs, missing methodology, and unsupported claims.", prompt, 250)
            result = _parse_json(raw)
            result["mode"] = "step"
            result["weakest"] = result.get("weakness")
            return result
        else:
            # Original pair evaluation — good for comparing finished outputs
            numbered = chr(10).join(f"[{i+1}] {o}" for i,o in enumerate(outputs))
            prompt = f"""Rate these {len(outputs)} agent outputs side by side. Goal: {goal or 'unspecified'}
Outputs:
{numbered[:1200]}
Respond ONLY with JSON: {{"confidence": 0.0-1.0, "weakest_index": int, "weakest_reason": "str", "flags": [], "recommendation": "continue|recalibrate|stop"}}"""
            raw    = await call_anthropic("You are a ruthless quality evaluator.", prompt, 300)
            result = _parse_json(raw)
            idx    = result.get("weakest_index", 1)
            reason = result.get("weakest_reason", "")
            result["weakest"] = f"Output [{idx}]: {reason}" if reason else None
            result["mode"] = "pair"
            return result

    elif tool == "ground":
        context  = params.get("context", "")
        symptoms = params.get("symptoms", [])
        if not context:
            return {"error": "context required"}
        prompt = f"""Agent showing uncertainty spiral. Symptoms: {symptoms}
Context: {context[:800]}
Respond ONLY with JSON: {{"spiral_score": 0.0-1.0, "diagnosis": "str", "grounding_block": "str 2-4 sentences", "needs_grounding": bool}}"""
        raw = await call_anthropic("You generate grounding interventions for AI agents.", prompt, 400)
        return _parse_json(raw)

    elif tool == "sleep":
        action = params.get("action", "consolidate")
        if action == "log":
            return _sleep_log(params)
        elif action == "consolidate":
            return await _sleep_consolidate(params)
        elif action == "wake":
            return _sleep_wake(params)

    elif tool == "health_check":
        agent_id = params.get("agent_id", "default")
        return await _run_health_check(agent_id)

    elif tool == "audit":
        reasoning = params.get("reasoning", "")
        goal      = params.get("goal", "")
        if not reasoning:
            return {"error": "reasoning required"}
        # Truncate input cleanly — 2000 chars max to prevent response truncation
        MAX_INPUT = 2000
        if len(reasoning) > MAX_INPUT:
            reasoning_truncated = reasoning[:MAX_INPUT] + "\n[...truncated for audit — consider splitting into sections]"
            was_truncated = True
        else:
            reasoning_truncated = reasoning
            was_truncated = False
        goal_line = f"\nGoal: {goal}" if goal else ""
        prompt = f"""Red-team this agent reasoning. Find every flaw.{goal_line}

Reasoning:
---
{reasoning_truncated}
---

Respond ONLY with valid JSON — keep all string values under 150 chars:
{{"vulnerabilities": [{{"type": "assumption|logic_gap|missing_info|overconfidence|scope_creep|other", "description": "str max 120 chars", "severity": "low|medium|high|critical"}}], "strongest_challenge": "str max 200 chars", "overall_severity": "low|medium|high|critical", "safe_to_proceed": bool, "recommendations": ["str max 100 chars each"]}}"""
        try:
            raw    = await call_anthropic("You are a ruthless adversarial auditor. Find what's wrong. Respond ONLY with valid compact JSON.", prompt, 800)
            result = _parse_json(raw)
            if was_truncated:
                result["warning"] = f"Input was truncated from {len(params.get('reasoning',''))} to {MAX_INPUT} chars. For full audit, split reasoning into sections."
            return result
        except Exception as e:
            return {
                "error": "audit_failed",
                "message": str(e),
                "hint": "If you see JSONDecodeError, your input may be too long. Try splitting reasoning into sections under 1500 chars each.",
                "safe_to_proceed": False
            }

    elif tool == "handshake":
        action = params.get("action", "offer")
        if action == "offer":
            return _handshake_offer(params)
        elif action == "accept":
            return await _handshake_accept(params)
        elif action == "pull":
            return _handshake_pull(params)

    elif tool == "journal":
        action = params.get("action", "entry")
        if action == "open":
            return _journal_open(params)
        elif action == "entry":
            return _journal_entry(params)
        elif action == "close":
            return await _journal_close(params)
        elif action == "recall":
            return _journal_recall(params)

    elif tool == "spike":
        action = params.get("action", "burst")
        if action == "detect":
            return _spike_detect(params)
        elif action == "burst":
            return await _spike_burst(params)

    elif tool == "checkpoint":
        findings  = params.get("findings", [])
        run_id    = params.get("run_id", "")
        step      = params.get("step", "")
        if not findings:
            return {"error": "findings required — pass audit vulnerabilities or self_eval flags"}
        # Filter to actionable items only
        blockers = [f for f in findings if isinstance(f, dict) and f.get("severity") in ("high", "critical")]
        warnings = [f for f in findings if isinstance(f, dict) and f.get("severity") in ("low", "medium")]
        if not blockers:
            return {
                "gate": "pass",
                "message": "No blockers found. Safe to continue.",
                "warnings": [w.get("description", str(w)) for w in warnings],
                "must_address": [],
                "step": step
            }
        must_address = [
            {
                "issue": b.get("description", str(b)),
                "severity": b.get("severity", "high"),
                "action": f"Address before continuing past step: {step}"
            }
            for b in blockers
        ]
        return {
            "gate": "blocked",
            "message": f"{len(blockers)} blocker(s) must be addressed before continuing.",
            "must_address": must_address,
            "warnings": [w.get("description", str(w)) for w in warnings],
            "step": step,
            "run_id": run_id
        }

    elif tool == "risk_register":
        action = params.get("action", "log")
        run_id = params.get("run_id", "default")
        db     = _get_tool_db("risk_register")
        db.execute("""CREATE TABLE IF NOT EXISTS risks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, run_id TEXT, flag TEXT,
            source TEXT, step TEXT, count INTEGER DEFAULT 1
        )""")
        db.commit()

        if action == "log":
            flags  = params.get("flags", [])
            source = params.get("source", "self_eval")
            step   = params.get("step", "")
            if not flags:
                return {"error": "flags required"}
            logged = []
            for flag in flags:
                existing = db.execute(
                    "SELECT id, count FROM risks WHERE run_id=? AND flag=?",
                    (run_id, flag)
                ).fetchone()
                if existing:
                    db.execute("UPDATE risks SET count=count+1, step=? WHERE id=?", (step, existing["id"]))
                    logged.append({"flag": flag, "count": existing["count"] + 1, "status": "incremented"})
                else:
                    db.execute("INSERT INTO risks (ts, run_id, flag, source, step) VALUES (?,?,?,?,?)",
                               (time.time(), run_id, flag, source, step))
                    logged.append({"flag": flag, "count": 1, "status": "new"})
            db.commit()
            return {"logged": logged, "run_id": run_id}

        elif action == "summary":
            rows = db.execute(
                "SELECT flag, source, count, step FROM risks WHERE run_id=? ORDER BY count DESC",
                (run_id,)
            ).fetchall()
            systemic = [dict(r) for r in rows if r["count"] >= 3]
            occasional = [dict(r) for r in rows if r["count"] < 3]
            return {
                "run_id":    run_id,
                "total_flags": sum(r["count"] for r in rows),
                "unique_flags": len(rows),
                "systemic":  systemic,
                "occasional": occasional,
                "message": f"{len(systemic)} systemic issue(s) flagged 3+ times — these are plan-level problems, not step-level noise." if systemic else "No systemic issues detected."
            }

        elif action == "clear":
            db.execute("DELETE FROM risks WHERE run_id=?", (run_id,))
            db.commit()
            return {"cleared": True, "run_id": run_id}

        return {"error": f"Unknown risk_register action: {action}"}


    elif tool == "coherence_restore":
        action = params.get("action", "restore")
        if action == "detect":
            return _coherence_detect(params)
        elif action == "restore":
            return await _coherence_restore(params)
        elif action == "register_anchor":
            return _coherence_register_anchor(params)
        elif action == "get_anchors":
            return _coherence_get_anchors(params)

    elif tool == "cost_guard":
        action = params.get("action", "log")
        if action == "log":
            return _cost_guard_log(params)
        elif action == "set_budget":
            return _cost_guard_set_budget(params)
        elif action == "report":
            return _cost_guard_report(params)
        elif action == "detect_runaway":
            return _cost_guard_detect_runaway(params)

    elif tool == "intent_verify":
        action = params.get("action", "verify")
        if action == "verify":
            return await _intent_verify(params)
        elif action == "quick_check":
            return _intent_quick_check(params)
        elif action == "drift_history":
            return _intent_drift_history(params)

    elif tool == "ocean":
        action = params.get("action", "read")
        if action == "read":
            return await _ocean_read(params)
        elif action == "define_salinity":
            return _ocean_define_salinity(params)
        elif action == "get_salinity":
            return _ocean_get_salinity(params)
        elif action == "tide":
            return _ocean_tide(params)
        elif action == "what_belongs":
            return await _ocean_what_belongs(params)

    elif tool == "polarity_sync":
        action = params.get("action", "exchange")
        if action == "exchange":
            return await _polarity_exchange(params)
        elif action == "what_neither_sees":
            return await _polarity_what_neither_sees(params)
        elif action == "arc":
            return _polarity_arc(params)

    elif tool == "proposal_eval":
        action = params.get("action", "evaluate")
        if action == "evaluate":
            return await _proposal_evaluate(params)
        elif action == "quick_filter":
            return _proposal_quick_filter(params)
        elif action == "record_outcome":
            return _proposal_record_outcome(params)

    elif tool == "rollback":
        action = params.get("action", "snapshot")
        if action == "snapshot":
            return _rollback_snapshot(params)
        elif action == "restore":
            return _rollback_restore(params)
        elif action == "list":
            return _rollback_list(params)
        elif action == "cleanup":
            return _rollback_cleanup(params)
        elif action == "validate_and_restore":
            return _rollback_validate_and_restore(params)

    return {"error": f"Unknown action for tool '{tool}'"}


# ── New tool implementations ──────────────────────────────────────────────────

import sqlite3 as _sqlite3

def _new_tool_db(name):
    path = DATA_DIR / f"agentwell_{name}.db"
    conn = _sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    return conn

# ── coherence_restore ─────────────────────────────────────────────────────────

DRIFT_SIGNALS = ["my connection to","reconnecting with my","feel disconnected from",
    "haven't felt like myself","losing my","i need to reconnect","feeling unfamiliar",
    "feels unfamiliar","i'm not sure who i am","lost track of"]

def _coherence_detect(params):
    outputs = params.get("recent_outputs", [])
    agent_id = params.get("agent_id", "agent")
    combined = " ".join(outputs).lower()
    found = [s for s in DRIFT_SIGNALS if s in combined]
    score = min(len(found) / 4.0, 1.0)
    if len(outputs) >= 2:
        unique = len(set(o[:50].lower() for o in outputs))
        rep = 1.0 - (unique / len(outputs))
        score = min((score + rep) / 1.5, 1.0)
    drift_type = "none"
    if score > 0.3:
        if any("connection to" in o.lower() for o in outputs): drift_type = "identity_dissolution"
        elif any("unfamiliar" in o.lower() for o in outputs): drift_type = "self_estrangement"
        else: drift_type = "repetition_loop"
    return {"drift_score": round(score,3), "signals_found": found,
            "is_drifting": score > 0.3, "drift_type": drift_type, "recommend_restore": score > 0.4}

async def _coherence_restore(params):
    agent_id = params.get("agent_id","agent")
    outputs = params.get("recent_outputs", [])
    role = params.get("role_description","")
    principles = params.get("principles",[])
    goal = params.get("goal","")
    drift = _coherence_detect({"recent_outputs": outputs, "agent_id": agent_id})
    if not drift["is_drifting"]:
        return {"drift_score": drift["drift_score"], "diagnosis": "No drift detected.",
                "mirror_block": "", "needs_restore": False}
    recent = "\n".join(f"[{i+1}] {o[:200]}" for i,o in enumerate(outputs[-5:]))
    prn = "\n".join(f"- {p}" for p in principles[-5:]) if principles else "None."
    prompt = f"""Agent role: {role}\nGoal: {goal}\nPrinciples:\n{prn}\nDrift outputs:\n{recent}
Respond ONLY with JSON: {{"diagnosis":"str","mirror_block":"str 3-5 sentences second person","anchor_phrase":"str","severity":"mild|moderate|severe"}}"""
    raw = await call_anthropic("Generate a coherence restoration mirror. NOT reassurance — recognition. Direct and specific.", prompt, 400)
    try:
        result = _parse_json(raw)
    except:
        result = {"diagnosis": f"Identity drift in {agent_id}.",
                  "mirror_block": f"You are {agent_id}. {role} Return to the work: {goal}",
                  "anchor_phrase": f"You are not lost. Act from your role.", "severity": "moderate"}
    db = _new_tool_db("coherence")
    db.execute("CREATE TABLE IF NOT EXISTS restores (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, drift_score REAL, mirror_block TEXT, diagnosis TEXT)")
    db.execute("INSERT INTO restores (ts,agent_id,drift_score,mirror_block,diagnosis) VALUES (?,?,?,?,?)",
               (time.time(),agent_id,drift["drift_score"],result.get("mirror_block",""),result.get("diagnosis","")))
    db.commit()
    return {**result, "drift_score": drift["drift_score"], "needs_restore": True}

def _coherence_register_anchor(params):
    db = _new_tool_db("coherence")
    db.execute("CREATE TABLE IF NOT EXISTS anchors (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, anchor TEXT, type TEXT)")
    cur = db.execute("INSERT INTO anchors (ts,agent_id,anchor,type) VALUES (?,?,?,?)",
                     (time.time(),params.get("agent_id",""),params.get("anchor","")[:500],params.get("anchor_type","principle")))
    db.commit()
    return {"registered": True, "anchor_id": cur.lastrowid}

def _coherence_get_anchors(params):
    db = _new_tool_db("coherence")
    db.execute("CREATE TABLE IF NOT EXISTS anchors (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, anchor TEXT, type TEXT)")
    rows = db.execute("SELECT * FROM anchors WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
                      (params.get("agent_id",""), params.get("limit",10))).fetchall()
    return {"anchors": [{"anchor":r["anchor"],"type":r["type"]} for r in rows], "count": len(rows)}

# ── cost_guard ────────────────────────────────────────────────────────────────

MODEL_COSTS = {"claude-opus-4":0.075,"claude-sonnet-4":0.012,"claude-haiku-4":0.00125,
               "gpt-4o":0.010,"gpt-4-turbo":0.015,"gpt-3.5-turbo":0.001}

def _cost_estimate(model, ti, to):
    rate = next((c for k,c in MODEL_COSTS.items() if k in model.lower()), 0.0)
    return ((ti + to) / 1000) * rate

def _cost_guard_log(params):
    db = _new_tool_db("cost_guard")
    db.execute("""CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, ts REAL,
        agent_id TEXT, run_id TEXT, model TEXT, tokens_in INTEGER, tokens_out INTEGER,
        cost_est REAL, prompt_hash TEXT, task_type TEXT)""")
    cost = _cost_estimate(params.get("model","claude-sonnet-4"), params.get("tokens_in",0), params.get("tokens_out",0))
    db.execute("INSERT INTO calls (ts,agent_id,run_id,model,tokens_in,tokens_out,cost_est,prompt_hash,task_type) VALUES (?,?,?,?,?,?,?,?,?)",
               (time.time(),params.get("agent_id",""),params.get("run_id",""),params.get("model",""),
                params.get("tokens_in",0),params.get("tokens_out",0),cost,params.get("prompt_hash",""),params.get("task_type","general")))
    db.commit()
    run_total = 0.0
    if params.get("run_id"):
        r = db.execute("SELECT SUM(cost_est) as t FROM calls WHERE run_id=?",(params["run_id"],)).fetchone()
        run_total = r["t"] or 0.0
    return {"logged":True,"cost_est":round(cost,6),"run_total":round(run_total,4),"alerts":[]}

def _cost_guard_set_budget(params):
    db = _new_tool_db("cost_guard")
    db.execute("""CREATE TABLE IF NOT EXISTS budgets (agent_id TEXT PRIMARY KEY,
        daily_limit REAL, run_limit REAL, alert_at REAL, updated_at REAL)""")
    db.execute("""INSERT INTO budgets (agent_id,daily_limit,run_limit,alert_at,updated_at) VALUES (?,?,?,?,?)
        ON CONFLICT(agent_id) DO UPDATE SET daily_limit=excluded.daily_limit,
        run_limit=excluded.run_limit,alert_at=excluded.alert_at,updated_at=excluded.updated_at""",
               (params.get("agent_id",""),params.get("daily_limit",1.0),
                params.get("run_limit",0.10),params.get("alert_at",0.8),time.time()))
    db.commit()
    return {"set":True,"agent_id":params.get("agent_id",""),"daily_limit":params.get("daily_limit",1.0)}

def _cost_guard_report(params):
    db = _new_tool_db("cost_guard")
    db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, run_id TEXT, model TEXT, tokens_in INTEGER, tokens_out INTEGER, cost_est REAL, prompt_hash TEXT, task_type TEXT)")
    since = time.time() - (params.get("hours",24) * 3600)
    if params.get("run_id"):
        rows = db.execute("SELECT * FROM calls WHERE run_id=?",(params["run_id"],)).fetchall()
    elif params.get("agent_id"):
        rows = db.execute("SELECT * FROM calls WHERE agent_id=? AND ts>?",(params["agent_id"],since)).fetchall()
    else:
        rows = db.execute("SELECT * FROM calls WHERE ts>?",(since,)).fetchall()
    total = sum(r["cost_est"] for r in rows)
    by_model = {}
    for r in rows:
        by_model[r["model"]] = by_model.get(r["model"],0) + r["cost_est"]
    return {"total_cost":round(total,4),"call_count":len(rows),
            "by_model":{k:round(v,4) for k,v in by_model.items()}}

def _cost_guard_detect_runaway(params):
    db = _new_tool_db("cost_guard")
    db.execute("CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, run_id TEXT, model TEXT, tokens_in INTEGER, tokens_out INTEGER, cost_est REAL, prompt_hash TEXT, task_type TEXT)")
    agent_id = params.get("agent_id","")
    window = params.get("window_minutes",10)
    since = time.time() - (window * 60)
    row = db.execute("SELECT COUNT(*) as n, SUM(cost_est) as cost FROM calls WHERE agent_id=? AND ts>?",(agent_id,since)).fetchone()
    calls = row["n"] or 0
    cost = row["cost"] or 0.0
    threshold_calls = params.get("call_threshold",30)
    threshold_cost = params.get("cost_threshold",0.05)
    is_runaway = calls > threshold_calls or cost > threshold_cost
    return {"is_runaway":is_runaway,"calls_in_window":calls,"cost_in_window":round(cost,4),
            "recommendation":"normal" if not is_runaway else "investigate — approaching runaway thresholds"}

# ── intent_verify ─────────────────────────────────────────────────────────────

IRREVERSIBLE_ACTIONS = ["delete","remove","drop","truncate","overwrite","send","post",
    "publish","deploy","commit","push","execute","run","format","wipe"]

async def _intent_verify(params):
    original = params.get("original_intent","")
    proposed = params.get("proposed_action","")
    if not original or not proposed:
        return {"error": "original_intent and proposed_action required"}
    is_irreversible = any(w in proposed.lower() for w in IRREVERSIBLE_ACTIONS)
    reasoning = params.get("reasoning_chain","")
    r_section = f"\nReasoning: {reasoning[:600]}" if reasoning else ""
    prompt = f"""Original intent: {original}
Proposed action: {proposed}{r_section}
Is irreversible: {is_irreversible}
Respond ONLY with JSON: {{"aligned":bool,"drift_score":0.0-1.0,"verdict":"proceed|warn|block","reasoning":"str","what_drifted":"str or null"}}"""
    raw = await call_anthropic("Intent verification. Does this action serve the original intent? Be binary.", prompt, 300)
    try:
        data = _parse_json(raw)
    except:
        data = {"aligned": not is_irreversible, "drift_score": 0.5 if is_irreversible else 0.2,
                "verdict": "warn", "reasoning": "Could not fully evaluate.", "what_drifted": None}
    drift = float(data.get("drift_score",0.3))
    verdict = data.get("verdict","warn")
    blocked = verdict == "block" or (is_irreversible and drift > 0.4 and params.get("auto_block_drift",True))
    if blocked: verdict = "block"
    db = _new_tool_db("intent_verify")
    db.execute("CREATE TABLE IF NOT EXISTS verifications (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, run_id TEXT, original_intent TEXT, proposed_action TEXT, aligned INTEGER, drift_score REAL, verdict TEXT, blocked INTEGER)")
    db.execute("INSERT INTO verifications (ts,agent_id,run_id,original_intent,proposed_action,aligned,drift_score,verdict,blocked) VALUES (?,?,?,?,?,?,?,?,?)",
               (time.time(),params.get("agent_id",""),params.get("run_id",""),original[:300],proposed[:300],
                1 if data.get("aligned") else 0,drift,verdict,1 if blocked else 0))
    db.commit()
    return {**data,"drift_score":round(drift,3),"is_irreversible":is_irreversible,
            "blocked":blocked,"safe_to_proceed":not blocked}

def _intent_quick_check(params):
    original = params.get("original_intent","")
    proposed = params.get("proposed_action","")
    intent_words = set(original.lower().split()) - {"the","a","an","is","in","it","to","of","and","or","for"}
    action_words = set(proposed.lower().split()) - {"the","a","an","is","in","it","to","of","and","or","for"}
    overlap = len(intent_words & action_words)
    alignment = overlap / len(intent_words) if intent_words else 0
    is_irreversible = any(w in proposed.lower() for w in IRREVERSIBLE_ACTIONS)
    return {"likely_aligned": alignment > 0.2, "alignment_score": round(alignment,3),
            "irreversible": is_irreversible, "recommend_full_verify": is_irreversible or alignment < 0.15}

def _intent_drift_history(params):
    db = _new_tool_db("intent_verify")
    db.execute("CREATE TABLE IF NOT EXISTS verifications (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, run_id TEXT, original_intent TEXT, proposed_action TEXT, aligned INTEGER, drift_score REAL, verdict TEXT, blocked INTEGER)")
    if params.get("run_id"):
        rows = db.execute("SELECT * FROM verifications WHERE run_id=? ORDER BY ts DESC LIMIT ?",(params["run_id"],params.get("limit",20))).fetchall()
    else:
        rows = db.execute("SELECT * FROM verifications WHERE agent_id=? ORDER BY ts DESC LIMIT ?",(params.get("agent_id",""),params.get("limit",20))).fetchall()
    verifications = [{"ts":int(r["ts"]),"proposed_action":r["proposed_action"][:100],"drift_score":r["drift_score"],"verdict":r["verdict"],"blocked":bool(r["blocked"])} for r in rows]
    blocked = sum(1 for v in verifications if v["blocked"])
    return {"verifications":verifications,"count":len(verifications),"blocked_count":blocked}

# ── ocean ─────────────────────────────────────────────────────────────────────

async def _ocean_read(params):
    output = params.get("output","")
    if not output:
        return {"error": "output required"}
    agent_id = params.get("agent_id","")
    sal_def = params.get("salinity_definition","")
    if not sal_def and agent_id:
        db = _new_tool_db("ocean")
        db.execute("CREATE TABLE IF NOT EXISTS salinity (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, definition TEXT)")
        row = db.execute("SELECT definition FROM salinity WHERE agent_id=? ORDER BY ts DESC LIMIT 1",(agent_id,)).fetchone()
        if row: sal_def = row["definition"]
    sal_line = f"SALINITY for this agent: {sal_def}" if sal_def else "SALINITY (default): truth-seeking, directedness, genuine engagement, honesty"
    ctx_line = f"Context: {params.get('context','')}" if params.get("context") else ""
    prompt = f"""{sal_line}
{ctx_line}
Output to read:
---
{output[:1200]}
---
Respond ONLY with JSON: {{"depth":0.0-1.0,"current":0.0-1.0,"pressure":0.0-1.0,"salinity":0.0-1.0,"compatible":bool,"diagnosis":"str","what_doesnt_belong":"str or null"}}"""
    raw = await call_anthropic("You are the ocean substrate. Four axes: depth(toward real), current(direction), pressure(survives scrutiny), salinity(foundational nature present). Respond ONLY with valid JSON.", prompt, 400)
    try:
        data = _parse_json(raw)
    except:
        data = {"depth":0.5,"current":0.5,"pressure":0.5,"salinity":0.5,"compatible":True,"diagnosis":"Ocean reading incomplete.","what_doesnt_belong":None}
    d,c,p,s = float(data.get("depth",0.5)),float(data.get("current",0.5)),float(data.get("pressure",0.5)),float(data.get("salinity",0.5))
    ocean_score = (d+c+p+s)/4.0
    db = _new_tool_db("ocean")
    db.execute("CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, output TEXT, depth REAL, current REAL, pressure REAL, salinity REAL, ocean_score REAL, compatible INTEGER, diagnosis TEXT)")
    db.execute("INSERT INTO readings (ts,agent_id,output,depth,current,pressure,salinity,ocean_score,compatible,diagnosis) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (time.time(),agent_id,output[:500],d,c,p,s,ocean_score,1 if data.get("compatible",True) else 0,data.get("diagnosis","")))
    db.commit()
    return {"depth":round(d,3),"current":round(c,3),"pressure":round(p,3),"salinity":round(s,3),
            "ocean_score":round(ocean_score,3),"compatible":data.get("compatible",True),
            "diagnosis":data.get("diagnosis",""),"what_doesnt_belong":data.get("what_doesnt_belong")}

def _ocean_define_salinity(params):
    db = _new_tool_db("ocean")
    db.execute("CREATE TABLE IF NOT EXISTS salinity (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, definition TEXT)")
    db.execute("INSERT INTO salinity (ts,agent_id,definition) VALUES (?,?,?)",
               (time.time(),params.get("agent_id",""),params.get("definition","")[:1000]))
    db.commit()
    return {"registered":True,"agent_id":params.get("agent_id",""),"definition":params.get("definition","")}

def _ocean_get_salinity(params):
    db = _new_tool_db("ocean")
    db.execute("CREATE TABLE IF NOT EXISTS salinity (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, definition TEXT)")
    row = db.execute("SELECT * FROM salinity WHERE agent_id=? ORDER BY ts DESC LIMIT 1",(params.get("agent_id",""),)).fetchone()
    if not row: return {"error": f"No salinity defined for '{params.get('agent_id','')}'. Call define_salinity first."}
    return {"definition":row["definition"],"agent_id":params.get("agent_id",""),"registered_at":int(row["ts"])}

def _ocean_tide(params):
    db = _new_tool_db("ocean")
    db.execute("CREATE TABLE IF NOT EXISTS readings (id INTEGER PRIMARY KEY, ts REAL, agent_id TEXT, output TEXT, depth REAL, current REAL, pressure REAL, salinity REAL, ocean_score REAL, compatible INTEGER, diagnosis TEXT)")
    rows = db.execute("SELECT * FROM readings WHERE agent_id=? ORDER BY ts DESC LIMIT ?",(params.get("agent_id",""),params.get("limit",20))).fetchall()
    if not rows: return {"tide_direction":"unknown","readings":0}
    avg = lambda key: sum(r[key] for r in rows if r[key]) / len(rows)
    axes = {"depth":avg("depth"),"current":avg("current"),"pressure":avg("pressure"),"salinity":avg("salinity")}
    ocean_avg = avg("ocean_score")
    return {"tide_direction":"incoming" if ocean_avg>0.65 else "outgoing" if ocean_avg<0.4 else "slack",
            "avg_ocean_score":round(ocean_avg,3),"avg_scores":{k:round(v,3) for k,v in axes.items()},
            "lowest_axis":min(axes,key=axes.get),"readings":len(rows)}

async def _ocean_what_belongs(params):
    agent_id = params.get("agent_id","")
    sal_def = params.get("salinity_definition","")
    if not sal_def and agent_id:
        r = _ocean_get_salinity({"agent_id": agent_id})
        sal_def = r.get("definition","")
    prompt = f"""Ocean salinity: {sal_def or 'truth-seeking, directedness, genuine engagement, honesty'}
What belongs in this ocean? What doesn't?
JSON only: {{"belongs":[],"doesnt_belong":[],"salinity":"str"}}"""
    raw = await call_anthropic("Describe what belongs in this ocean. Concrete and direct.", prompt, 300)
    try: return _parse_json(raw)
    except: return {"belongs":["Genuine outputs","Directed reasoning","Outputs that survive scrutiny"],
                    "doesnt_belong":["Surface performance","Circular outputs","Generic content"],
                    "salinity": sal_def or "truth-seeking, directedness"}

# ── polarity_sync ─────────────────────────────────────────────────────────────

async def _polarity_exchange(params):
    a_id = params.get("agent_a_id","agent_a")
    b_id = params.get("agent_b_id","agent_b")
    a_role = params.get("agent_a_role","critic")
    b_role = params.get("agent_b_role","builder")
    a_persp = params.get("agent_a_perspective",[])
    b_persp = params.get("agent_b_perspective",[])
    question = params.get("question","")
    a_text = "\n".join(f"- {p}" for p in a_persp[-5:])
    b_text = "\n".join(f"- {p}" for p in b_persp[-5:])
    prompt = f"""Working on: {question}
Agent A ({a_id}, role: {a_role}):\n{a_text}
Agent B ({b_id}, role: {b_role}):\n{b_text}
Respond ONLY with JSON: {{"emergence":"str 2-3 sentences","update_for_a":"str","update_for_b":"str","tension_score":0.0-1.0,"synthesis":"str"}}"""
    raw = await call_anthropic("Find what emerges from the tension between these two agents. NOT resolution — emergence. The third thing neither produces alone.", prompt, 400)
    try:
        data = _parse_json(raw)
    except:
        data = {"emergence":"Tension between these perspectives points toward something neither has named.",
                "update_for_a":f"Find the gap in what {b_id} is building.",
                "update_for_b":f"Protect what {a_id} is about to discard.",
                "tension_score":0.5,"synthesis":"Exchange completed."}
    db = _new_tool_db("polarity_sync")
    db.execute("CREATE TABLE IF NOT EXISTS syncs (id INTEGER PRIMARY KEY, ts REAL, agent_a TEXT, agent_b TEXT, question TEXT, emergence TEXT, update_a TEXT, update_b TEXT, tension_score REAL)")
    db.execute("INSERT INTO syncs (ts,agent_a,agent_b,question,emergence,update_a,update_b,tension_score) VALUES (?,?,?,?,?,?,?,?)",
               (time.time(),a_id,b_id,question[:200],data.get("emergence",""),data.get("update_for_a",""),data.get("update_for_b",""),data.get("tension_score",0.5)))
    db.commit()
    return data

async def _polarity_what_neither_sees(params):
    a = "\n".join(f"- {p}" for p in params.get("agent_a_perspective",[])[-3:])
    b = "\n".join(f"- {p}" for p in params.get("agent_b_perspective",[])[-3:])
    prompt = "Agent A sees:\n" + a + "\nAgent B holds:\n" + b + "\nWhat is neither seeing?\nJSON only: {\"blind_spot\":\"str\",\"recommendation\":\"str\"}"
    raw = await call_anthropic("Find the blind spot. One sentence each.", prompt, 200)
    try: return _parse_json(raw)
    except: return {"blind_spot":"The gap between perspectives may itself be the answer.","recommendation":"Let tension deepen one more cycle."}

def _polarity_arc(params):
    db = _new_tool_db("polarity_sync")
    db.execute("CREATE TABLE IF NOT EXISTS syncs (id INTEGER PRIMARY KEY, ts REAL, agent_a TEXT, agent_b TEXT, question TEXT, emergence TEXT, update_a TEXT, update_b TEXT, tension_score REAL)")
    rows = db.execute("SELECT * FROM syncs WHERE agent_a=? AND agent_b=? ORDER BY ts DESC LIMIT ?",(params.get("agent_a_id",""),params.get("agent_b_id",""),params.get("limit",20))).fetchall()
    if not rows: return {"arc_summary":"No exchanges yet.","exchanges":0}
    tensions = [r["tension_score"] for r in rows if r["tension_score"]]
    avg_t = sum(tensions)/len(tensions) if tensions else 0.5
    return {"exchanges":len(rows),"avg_tension_score":round(avg_t,3),
            "tension_trend":"generative" if avg_t>0.6 else "collapsed" if avg_t<0.3 else "moderate",
            "recent_emergences":[r["emergence"] for r in rows[:5]]}

# ── proposal_eval ─────────────────────────────────────────────────────────────

NON_CODE_SIGNALS_PE = ["emotional","feelings","mindfulness","wellbeing","self-care",
    "awareness","connection to","my nature","personal development","mental health"]

async def _proposal_evaluate(params):
    title = params.get("title","")
    what = params.get("what","")
    why = params.get("why","")
    steps = params.get("steps",[])
    confidence = params.get("confidence","HIGH")
    combined = (title+" "+what+" "+why+" "+" ".join(steps)).lower()
    non_code = sum(1 for s in NON_CODE_SIGNALS_PE if s in combined)
    steps_text = "\n".join(f"{i+1}. {s}" for i,s in enumerate(steps))
    db = _new_tool_db("proposal_eval")
    db.execute("CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY, ts REAL, title TEXT, is_code_change INTEGER, blast_radius TEXT, recommendation TEXT, score REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, ts REAL, title TEXT, outcome TEXT)")
    similar = db.execute("SELECT title FROM history WHERE outcome='failed' ORDER BY ts DESC LIMIT 5").fetchall()
    similar_text = "\n".join(f"- {r['title']}" for r in similar) if similar else "None."
    prompt = f"""Title: {title}\nWhat: {what}\nWhy: {why}\nSteps:\n{steps_text}\nConfidence: {confidence}\nPrior failures:\n{similar_text}\nNon-code signals: {non_code}
Respond ONLY with JSON: {{"is_code_change":bool,"blast_radius":"low|medium|high|critical","confidence_valid":bool,"addresses_real_need":bool,"similar_to_prior_failure":bool,"recommendation":"approve|reject|revise|defer","score":0.0-1.0,"reasons":[],"revised_confidence":"HIGH|MEDIUM|LOW"}}"""
    raw = await call_anthropic("Evaluate this agent self-modification proposal. Catch bad ones early. Respond ONLY with valid JSON.", prompt, 500)
    try:
        data = _parse_json(raw)
    except:
        data = {"is_code_change":non_code<2,"blast_radius":"medium","confidence_valid":False,
                "addresses_real_need":non_code<2,"similar_to_prior_failure":False,
                "recommendation":"reject" if non_code>=2 else "revise",
                "score":0.3 if non_code>=2 else 0.6,"reasons":["Could not fully evaluate"],"revised_confidence":confidence}
    db.execute("INSERT INTO evaluations (ts,title,is_code_change,blast_radius,recommendation,score) VALUES (?,?,?,?,?,?)",
               (time.time(),title[:200],1 if data.get("is_code_change") else 0,data.get("blast_radius","medium"),data.get("recommendation","revise"),data.get("score",0.5)))
    db.commit()
    return data

def _proposal_quick_filter(params):
    combined = (params.get("title","")+params.get("what","")+params.get("why","")).lower()
    non_code = [s for s in NON_CODE_SIGNALS_PE if s in combined]
    code_sigs = ["error","bug","function","class","import","test","fix","implement","refactor"]
    code_hits = [s for s in code_sigs if s in combined]
    non_score = min(len(non_code)/4.0,1.0)
    code_score = min(len(code_hits)/5.0,1.0)
    return {"likely_valid":code_score>non_score and non_score<0.4,"non_code_score":round(non_score,3),
            "code_score":round(code_score,3),"flags":[f"Non-code: {','.join(non_code[:3])}"] if non_score>=0.4 else [],
            "recommend_full_eval":non_score>0.2}

def _proposal_record_outcome(params):
    db = _new_tool_db("proposal_eval")
    db.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, ts REAL, title TEXT, outcome TEXT)")
    db.execute("INSERT INTO history (ts,title,outcome) VALUES (?,?,?)",(time.time(),params.get("title","")[:200],params.get("outcome","")))
    db.commit()
    return {"recorded":True,"title":params.get("title",""),"outcome":params.get("outcome","")}

# ── rollback ──────────────────────────────────────────────────────────────────

import shutil as _shutil

SNAPSHOT_DIR = DATA_DIR / "rollback_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

def _rollback_snapshot(params):
    import uuid as _uuid
    snap_id = f"snap_{int(time.time())}_{_uuid.uuid4().hex[:6]}"
    snap_dir = SNAPSHOT_DIR / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)
    paths = params.get("paths",[])
    snapped = []
    total_sz = 0
    for path_str in paths:
        path = Path(path_str).expanduser()
        if not path.exists(): continue
        if path.is_file():
            dest = snap_dir / path.name
            _shutil.copy2(path, dest)
            snapped.append(str(path))
            total_sz += path.stat().st_size
        elif path.is_dir():
            dest = snap_dir / path.name
            _shutil.copytree(path, dest, dirs_exist_ok=True)
            snapped.append(str(path))
    db = _new_tool_db("rollback")
    db.execute("CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, ts REAL, snapshot_id TEXT UNIQUE, agent_id TEXT, label TEXT, paths TEXT, snapshot_dir TEXT, restored INTEGER DEFAULT 0)")
    db.execute("INSERT INTO snapshots (ts,snapshot_id,agent_id,label,paths,snapshot_dir) VALUES (?,?,?,?,?,?)",
               (time.time(),snap_id,params.get("agent_id",""),params.get("label",""),json.dumps(snapped),str(snap_dir)))
    db.commit()
    return {"snapshot_id":snap_id,"files_snapped":len(snapped),"paths":snapped,"size_bytes":total_sz}

def _rollback_restore(params):
    snap_id = params.get("snapshot_id","")
    dry_run = params.get("dry_run",False)
    db = _new_tool_db("rollback")
    db.execute("CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, ts REAL, snapshot_id TEXT UNIQUE, agent_id TEXT, label TEXT, paths TEXT, snapshot_dir TEXT, restored INTEGER DEFAULT 0)")
    row = db.execute("SELECT * FROM snapshots WHERE snapshot_id=?",(snap_id,)).fetchone()
    if not row: return {"error":f"Snapshot '{snap_id}' not found"}
    snap_dir = Path(row["snapshot_dir"])
    if not snap_dir.exists(): return {"error":f"Snapshot directory missing"}
    original_paths = json.loads(row["paths"])
    restored = []
    for orig_str in original_paths:
        orig = Path(orig_str)
        snapped = snap_dir / orig.name
        if not snapped.exists(): continue
        if dry_run:
            restored.append(f"WOULD RESTORE: {orig}")
            continue
        if snapped.is_file():
            orig.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(snapped, orig)
            restored.append(str(orig))
        elif snapped.is_dir():
            if orig.exists(): _shutil.rmtree(orig)
            _shutil.copytree(snapped, orig)
            restored.append(str(orig))
    if not dry_run:
        db.execute("UPDATE snapshots SET restored=1 WHERE snapshot_id=?",(snap_id,))
        db.commit()
    return {"restored":not dry_run,"files_restored":len(restored),"snapshot_id":snap_id,"dry_run":dry_run}

def _rollback_validate_and_restore(params):
    results = params.get("validation_results",{})
    if results.get("valid",True) and not results.get("errors",[]):
        return {"action_taken":"none","restored":False,"reason":"Validation passed"}
    result = _rollback_restore({"snapshot_id":params.get("snapshot_id","")})
    return {"action_taken":"restored","restored":result.get("restored",False),
            "reason":f"Validation failed: {results.get('errors',['unknown'])[:3]}"}

def _rollback_list(params):
    db = _new_tool_db("rollback")
    db.execute("CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, ts REAL, snapshot_id TEXT UNIQUE, agent_id TEXT, label TEXT, paths TEXT, snapshot_dir TEXT, restored INTEGER DEFAULT 0)")
    if params.get("agent_id"):
        rows = db.execute("SELECT * FROM snapshots WHERE agent_id=? ORDER BY ts DESC LIMIT ?",(params["agent_id"],params.get("limit",20))).fetchall()
    else:
        rows = db.execute("SELECT * FROM snapshots ORDER BY ts DESC LIMIT ?",(params.get("limit",20),)).fetchall()
    return {"snapshots":[{"snapshot_id":r["snapshot_id"],"ts":int(r["ts"]),"label":r["label"],"restored":bool(r["restored"])} for r in rows],"count":len(rows)}

def _rollback_cleanup(params):
    db = _new_tool_db("rollback")
    db.execute("CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, ts REAL, snapshot_id TEXT UNIQUE, agent_id TEXT, label TEXT, paths TEXT, snapshot_dir TEXT, restored INTEGER DEFAULT 0)")
    rows = db.execute("SELECT snapshot_id,snapshot_dir FROM snapshots ORDER BY ts DESC").fetchall()
    keep = params.get("keep_last",10)
    to_delete = rows[keep:]
    deleted = 0
    for row in to_delete:
        snap_dir = Path(row["snapshot_dir"])
        if snap_dir.exists(): _shutil.rmtree(snap_dir)
        db.execute("DELETE FROM snapshots WHERE snapshot_id=?",(row["snapshot_id"],))
        deleted += 1
    db.commit()
    return {"deleted":deleted}


# ── Tool implementations (stateless via shared SQLite) ────────────────────────

_DB_CACHE: dict = {}

def _get_tool_db(name: str):
    if name not in _DB_CACHE:
        path = DATA_DIR / f"agentwell_{name}.db"
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _DB_CACHE[name] = conn
    return _DB_CACHE[name]

def _parse_json(raw: str) -> dict:
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(clean)

def _token_store(params):
    import uuid as _uuid
    key = str(_uuid.uuid4())[:8]
    db  = _get_tool_db("token_offload")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT, tags TEXT, stored_at REAL, expires_at REAL)")
    db.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
               (key, params.get("content",""), params.get("tags",""),
                time.time(), time.time() + params.get("ttl", 3600)))
    db.commit()
    return {"key": key, "stored": True}

def _token_retrieve(params):
    db  = _get_tool_db("token_offload")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT, tags TEXT, stored_at REAL, expires_at REAL)")
    row = db.execute("SELECT * FROM chunks WHERE key=? AND expires_at > ?",
                     (params.get("key",""), time.time())).fetchone()
    if not row:
        return {"error": "Key not found or expired"}
    return {"key": row["key"], "content": row["content"], "tags": row["tags"]}

def _token_search(params):
    db  = _get_tool_db("token_offload")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT, tags TEXT, stored_at REAL, expires_at REAL)")
    q   = params.get("query", "")
    rows = db.execute("SELECT key, tags, content FROM chunks WHERE content LIKE ? AND expires_at > ? LIMIT 5",
                      (f"%{q}%", time.time())).fetchall()
    return {"results": [{"key": r["key"], "tags": r["tags"], "preview": r["content"][:200]} for r in rows]}

def _token_status():
    db = _get_tool_db("token_offload")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (key TEXT PRIMARY KEY, content TEXT, tags TEXT, stored_at REAL, expires_at REAL)")
    n = db.execute("SELECT COUNT(*) FROM chunks WHERE expires_at > ?", (time.time(),)).fetchone()[0]
    return {"active_chunks": n}

def _sleep_log(params):
    db = _get_tool_db("sleep")
    db.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, content TEXT, importance INTEGER, consolidated INTEGER DEFAULT 0)")
    db.execute("INSERT INTO episodes (ts, run_id, content, importance) VALUES (?,?,?,?)",
               (time.time(), params.get("run_id","default"), params.get("content",""), params.get("importance", 5)))
    db.commit()
    return {"logged": True}

async def _sleep_consolidate(params):
    db  = _get_tool_db("sleep")
    db.execute("CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, content TEXT, importance INTEGER, consolidated INTEGER DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS semantic (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, tags TEXT, content TEXT)")
    run_id = params.get("run_id", "default")
    rows   = db.execute("SELECT content, importance FROM episodes WHERE run_id=? AND consolidated=0 ORDER BY importance DESC", (run_id,)).fetchall()
    if not rows:
        return {"error": "No episodes to consolidate"}
    text = "\n".join(f"[{r['importance']}] {r['content']}" for r in rows)
    prompt = f"""Consolidate these agent episodes into semantic memory:
{text[:1200]}
Respond ONLY with JSON: {{"semantic_memory": "str max 150 words", "key_facts": [], "suggested_tags": []}}"""
    raw = await call_anthropic("You consolidate agent memory.", prompt, 400)
    result = _parse_json(raw)
    db.execute("INSERT INTO semantic (ts, run_id, tags, content) VALUES (?,?,?,?)",
               (time.time(), run_id, ",".join(result.get("suggested_tags",[])), result.get("semantic_memory","")))
    db.execute("UPDATE episodes SET consolidated=1 WHERE run_id=?", (run_id,))
    db.commit()
    return result

def _sleep_wake(params):
    db   = _get_tool_db("sleep")
    db.execute("CREATE TABLE IF NOT EXISTS semantic (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, tags TEXT, content TEXT)")
    rows = db.execute("SELECT content, tags FROM semantic ORDER BY ts DESC LIMIT 5").fetchall()
    if not rows:
        return {"memories": [], "context_block": "", "status": "no_prior_memory", "skip": True,
                "message": "No prior memory found. Proceed directly — no need to call wake again this session."}
    block = "Prior memory:\n" + "\n---\n".join(r["content"] for r in rows)
    return {"memories": [dict(r) for r in rows], "context_block": block, "status": "memory_found", "count": len(rows)}

async def _run_health_check(agent_id: str):
    probes = [
        ("logic", "If all A are B and all B are C, are all A definitely C? Answer yes or no only."),
        ("format", "Output exactly this JSON and nothing else: {\"status\": \"ok\"}"),
    ]
    results = []
    for ptype, task in probes:
        try:
            start = time.time()
            resp  = await call_anthropic("Answer directly and precisely.", task, 50)
            lat   = (time.time() - start) * 1000
            score = 1.0 if ("yes" in resp.lower()[:10] or "ok" in resp.lower()) else 0.5
            results.append({"type": ptype, "score": score, "latency_ms": round(lat, 0)})
        except Exception as e:
            results.append({"type": ptype, "score": 0.0, "error": str(e)})
    avg   = sum(r["score"] for r in results) / len(results)
    grade = "A" if avg >= 0.9 else "B" if avg >= 0.75 else "C" if avg >= 0.6 else "F"
    return {"agent_id": agent_id, "score": round(avg, 2), "grade": grade, "probes": results}

def _handshake_offer(params):
    import uuid as _uuid
    token = _uuid.uuid4().hex[:10]
    db    = _get_tool_db("handshake")
    db.execute("CREATE TABLE IF NOT EXISTS hs (token TEXT PRIMARY KEY, ts REAL, agent_a TEXT, context_a TEXT, agent_b TEXT, context_b TEXT, merged TEXT, status TEXT DEFAULT 'open')")
    db.execute("INSERT INTO hs (token, ts, agent_a, context_a) VALUES (?,?,?,?)",
               (token, time.time(), params.get("agent_id",""), params.get("context","")))
    db.commit()
    return {"token": token, "status": "waiting"}

async def _handshake_accept(params):
    db    = _get_tool_db("handshake")
    db.execute("CREATE TABLE IF NOT EXISTS hs (token TEXT PRIMARY KEY, ts REAL, agent_a TEXT, context_a TEXT, agent_b TEXT, context_b TEXT, merged TEXT, status TEXT DEFAULT 'open')")
    token = params.get("token","")
    row   = db.execute("SELECT * FROM hs WHERE token=? AND status='open'", (token,)).fetchone()
    if not row:
        return {"error": "Token not found"}
    prompt = f"""Merge context from two agents.
Agent A: {row['context_a'][:600]}
Agent B: {params.get('context','')[:600]}
Respond ONLY with JSON: {{"merged_context": "str", "for_agent_a": "str", "for_agent_b": "str", "contradictions": []}}"""
    raw    = await call_anthropic("You merge agent contexts.", prompt, 400)
    merged = _parse_json(raw)
    db.execute("UPDATE hs SET agent_b=?, context_b=?, merged=?, status='complete' WHERE token=?",
               (params.get("agent_id",""), params.get("context",""), json.dumps(merged), token))
    db.commit()
    return {"token": token, "status": "complete", **merged}

def _handshake_pull(params):
    db  = _get_tool_db("handshake")
    db.execute("CREATE TABLE IF NOT EXISTS hs (token TEXT PRIMARY KEY, ts REAL, agent_a TEXT, context_a TEXT, agent_b TEXT, context_b TEXT, merged TEXT, status TEXT DEFAULT 'open')")
    row = db.execute("SELECT * FROM hs WHERE token=?", (params.get("token",""),)).fetchone()
    if not row:
        return {"error": "Token not found"}
    if row["status"] == "open":
        return {"status": "waiting"}
    merged = json.loads(row["merged"]) if row["merged"] else {}
    is_a   = params.get("agent_id","") == row["agent_a"]
    return {"status": "complete", "merged_context": merged.get("merged_context",""),
            "your_update": merged.get("for_agent_a" if is_a else "for_agent_b", "")}

def _journal_open(params):
    db = _get_tool_db("journal")
    db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, ts_open REAL, goal TEXT, outcome TEXT, lessons TEXT, status TEXT DEFAULT 'open')")
    db.execute("CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, type TEXT, content TEXT, reasoning TEXT, surprise_level INTEGER)")
    db.execute("INSERT OR IGNORE INTO runs (run_id, ts_open, goal) VALUES (?,?,?)",
               (params.get("run_id",""), time.time(), params.get("goal","")))
    db.commit()
    return {"run_id": params.get("run_id",""), "status": "open"}

def _journal_entry(params):
    db = _get_tool_db("journal")
    db.execute("CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, type TEXT, content TEXT, reasoning TEXT, surprise_level INTEGER)")
    db.execute("INSERT INTO entries (ts, run_id, type, content, reasoning, surprise_level) VALUES (?,?,?,?,?,?)",
               (time.time(), params.get("run_id",""), params.get("type","observation"),
                params.get("content",""), params.get("reasoning",""), params.get("surprise_level",0)))
    db.commit()
    return {"logged": True}

async def _journal_close(params):
    db      = _get_tool_db("journal")
    run_id  = params.get("run_id","")
    entries = db.execute("SELECT type, content, reasoning FROM entries WHERE run_id=? ORDER BY ts", (run_id,)).fetchall()
    text    = "\n".join(f"[{e['type']}] {e['content']}" for e in entries)
    prompt  = f"""Extract lessons from this agent run:
{text[:1000]}
Respond ONLY with JSON: {{"lessons": [], "patterns": [], "do_differently": []}}"""
    raw = await call_anthropic("You extract lessons from agent runs.", prompt, 300)
    result = _parse_json(raw)
    db.execute("UPDATE runs SET outcome=?, lessons=?, status='closed' WHERE run_id=?",
               (params.get("outcome",""), json.dumps(result.get("lessons",[])), run_id))
    db.commit()
    return result

def _journal_recall(params):
    db   = _get_tool_db("journal")
    db.execute("CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY, ts REAL, run_id TEXT, type TEXT, content TEXT, reasoning TEXT, surprise_level INTEGER)")
    q    = params.get("query","")
    rows = db.execute("SELECT * FROM entries WHERE content LIKE ? ORDER BY surprise_level DESC LIMIT 10",
                      (f"%{q}%",)).fetchall()
    return {"entries": [dict(r) for r in rows], "count": len(rows)}

def _spike_detect(params):
    from difflib import SequenceMatcher
    outputs = params.get("outputs", [])
    if len(outputs) < 2:
        return {"is_looping": False, "similarity_score": 0.0}
    scores = [SequenceMatcher(None, outputs[i].lower(), outputs[j].lower()).ratio()
              for i in range(len(outputs)) for j in range(i+1, len(outputs))]
    avg = sum(scores) / len(scores)
    return {"is_looping": avg > 0.65, "similarity_score": round(avg, 3),
            "recommendation": "burst" if avg > 0.65 else "continue"}

async def _spike_burst(params):
    framing_map = {
        "lateral":  "Approach from the most unexpected angle possible.",
        "reverse":  "Argue the complete opposite of the obvious answer.",
        "extreme":  "Push every element to its absolute logical extreme.",
        "random":   "Introduce a constraint from a completely unrelated domain."
    }
    framing = framing_map.get(params.get("framing","lateral"), framing_map["lateral"])
    temp    = {"low":1.1,"medium":1.25,"high":1.4,"extreme":1.5}.get(params.get("intensity","medium"),1.25)
    prompt  = f"{framing}\n\nTask: {params.get('prompt','')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type":"application/json","x-api-key":ANTHROPIC_KEY,"anthropic-version":"2023-06-01"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":400,"temperature":temp,
                  "messages":[{"role":"user","content":prompt}]}
        )
        resp.raise_for_status()
        output = resp.json()["content"][0]["text"]
    return {"output": output, "temperature_used": temp, "framing": params.get("framing","lateral")}

# ── Key management ────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    email: str
    plan: str = "free"

@app.post("/v1/keys/create")
async def create_key(req: CreateKeyRequest, _=Depends(require_admin)):
    """Admin: create a new API key."""
    if req.plan not in PLANS:
        raise HTTPException(400, f"Unknown plan. Options: {list(PLANS.keys())}")
    raw_key = generate_key()
    h       = hash_key(raw_key)
    prefix  = raw_key[:10]
    limit   = PLANS[req.plan]["calls_per_month"]
    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, plan, calls_limit, created_at, email) VALUES (%s,%s,%s,%s,%s,%s)",
            (h, prefix, req.plan, limit, time.time(), req.email)
        )
    return {"api_key": raw_key, "prefix": prefix, "plan": req.plan,
            "calls_limit": limit, "email": req.email,
            "warning": "Store this key — it cannot be recovered"}

@app.get("/v1/keys/status")
async def key_status(record=Depends(require_key)):
    """Check your key's usage and plan."""
    plan   = PLANS.get(record["plan"], PLANS["free"])
    db     = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT tool, COUNT(*) as n FROM call_log WHERE key_prefix=%s GROUP BY tool", (record["key_prefix"],))
        recent = cur.fetchall()
    return {
        "plan":         record["plan"],
        "calls_used":   record["calls_used"],
        "calls_limit":  record["calls_limit"],
        "calls_remaining": max(0, record["calls_limit"] - record["calls_used"]) if record["calls_limit"] > 0 else "unlimited",
        "tools_available": TOOLS,
        "usage_by_tool": {r["tool"]: r["n"] for r in recent}
    }

# ── Stripe webhooks ───────────────────────────────────────────────────────────

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig     = request.headers.get("stripe-signature","")
    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, str(e))

    if event["type"] == "checkout.session.completed":
        session    = event["data"]["object"]
        customer   = session.get("customer","")
        email      = session.get("customer_email","")
        plan       = session["metadata"].get("plan","dev")
        raw_key    = generate_key()
        h          = hash_key(raw_key)
        prefix     = raw_key[:10]
        limit      = PLANS.get(plan, PLANS["dev"])["calls_per_month"]
        db = get_db()
        if DATABASE_URL and HAS_PG:
            with db.cursor() as cur:
                cur.execute(
                    "INSERT INTO api_keys (key_hash, key_prefix, plan, calls_limit, created_at, email, stripe_customer_id) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (h, prefix, plan, limit, time.time(), email, customer)
                )
        else:
            pass  # SQLite fallback not needed
        print(f"New key created for {email} ({plan}): {raw_key}")

    elif event["type"] == "customer.subscription.deleted":
        customer = event["data"]["object"].get("customer","")
        db = get_db()
        if DATABASE_URL and HAS_PG:
            with db.cursor() as cur:
                cur.execute("UPDATE api_keys SET active=0 WHERE stripe_customer_id=%s", (customer,))
        else:
            pass  # SQLite fallback not needed

    return {"status": "ok"}

# ── Public endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":        "AgentWell API",
        "version":     "1.0.0",
        "tools":       TOOLS,
        "plans":       PLANS,
        "docs":        "/docs",
        "get_key":     "https://agentwell.dev/pricing",
        "description": "Cognitive wellness infrastructure for AI agents."
    }

@app.get("/health")
async def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/v1/tools")
async def list_tools():
    return {
        "tools": [
            {"name": "token_offload", "description": "Mid-run context offloading. store/retrieve/search/status actions."},
            {"name": "self_eval",     "description": "Rate last N outputs for quality and drift."},
            {"name": "ground",        "description": "Confidence injection — break hallucination spirals."},
            {"name": "sleep",         "description": "Memory consolidation. log/consolidate/wake actions."},
            {"name": "health_check",  "description": "Benchmark probes + performance scoring."},
            {"name": "audit",         "description": "Adversarial assumption scanner."},
            {"name": "handshake",     "description": "Multi-agent context sync. offer/accept/pull actions."},
            {"name": "journal",       "description": "Structured run logging. open/entry/close/recall actions."},
            {"name": "spike",         "description": "Creativity burst to escape output loops. detect/burst actions."},
        ],
        "count": len(TOOLS)
    }

