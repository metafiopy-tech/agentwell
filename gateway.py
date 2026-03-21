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

DB_PATH = Path.home() / ".agentwell_gateway.db"

PLANS = {
    "payg":    {"calls_per_month": 0,       "price_per_call": 0.002, "monthly_fee": 0},
    "dev":     {"calls_per_month": 20_000,  "price_per_call": 0.0,   "monthly_fee": 29},
    "agency":  {"calls_per_month": 999_999, "price_per_call": 0.0,   "monthly_fee": 199},
    "free":    {"calls_per_month": 100,     "price_per_call": 0.0,   "monthly_fee": 0},
}

TOOLS = [
    "token_offload", "self_eval", "ground", "sleep",
    "health_check", "audit", "handshake", "journal", "spike"
]

# ── DB ────────────────────────────────────────────────────────────────────────

_local = threading.local()

def get_db():
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
    db  = get_db()
    h   = hash_key(raw_key)
    row = db.execute(
        "SELECT * FROM api_keys WHERE key_hash=? AND active=1", (h,)
    ).fetchone()
    return row

def increment_calls(key_prefix: str, tool: str, status: int, latency: float):
    db = get_db()
    db.execute(
        "UPDATE api_keys SET calls_used = calls_used + 1 WHERE key_prefix=?",
        (key_prefix,)
    )
    db.execute(
        "INSERT INTO call_log (ts, key_prefix, tool, status, latency_ms) VALUES (?,?,?,?,?)",
        (time.time(), key_prefix, tool, status, latency)
    )
    db.commit()

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
    get_db()
    # Seed a free dev key for testing
    db = get_db()
    test_key = "aw_testkey_dev123"
    h = hash_key(test_key)
    exists = db.execute("SELECT key_hash FROM api_keys WHERE key_hash=?", (h,)).fetchone()
    if not exists:
        db.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, plan, calls_limit, created_at, email) VALUES (?,?,?,?,?,?)",
            (h, "aw_testk", "free", 100, time.time(), "dev@agentwell.dev")
        )
        db.commit()
        print(f"Dev test key: {test_key}")
    yield

app = FastAPI(
    title="AgentWell API",
    description="Cognitive wellness infrastructure for AI agents. One key, nine tools, growing library.",
    version="1.0.0",
    lifespan=lifespan
)

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

    except Exception as e:
        latency = (time.time() - start) * 1000
        increment_calls(key_prefix, req.tool, 500, latency)
        raise HTTPException(500, f"Tool error: {str(e)}")


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
        outputs = params.get("outputs", [])
        goal    = params.get("goal", "")
        if not outputs:
            return {"error": "outputs required"}
        prompt = f"""Rate these {len(outputs)} agent outputs. Goal: {goal or 'unspecified'}
Outputs:
{chr(10).join(f'[{i+1}] {o}' for i,o in enumerate(outputs))}
Respond ONLY with JSON: {{"confidence": 0.0-1.0, "weakest_index": int, "weakest_reason": "str", "flags": [], "recommendation": "continue|recalibrate|stop"}}"""
        raw = await call_anthropic("You are a ruthless quality evaluator.", prompt, 300)
        return _parse_json(raw)

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
        if not reasoning:
            return {"error": "reasoning required"}
        prompt = f"""Red-team this agent reasoning:
---
{reasoning[:1200]}
---
Respond ONLY with JSON: {{"vulnerabilities": [{{"type": "str", "description": "str", "severity": "low|medium|high|critical"}}], "strongest_challenge": "str", "overall_severity": "str", "safe_to_proceed": bool, "recommendations": []}}"""
        raw = await call_anthropic("You are a ruthless adversarial auditor.", prompt, 600)
        return _parse_json(raw)

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

    return {"error": f"Unknown action for tool '{tool}'"}

# ── Tool implementations (stateless via shared SQLite) ────────────────────────

_DB_CACHE: dict = {}

def _get_tool_db(name: str):
    if name not in _DB_CACHE:
        path = Path.home() / f".agentwell_{name}.db"
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
    block = "Prior memory:\n" + "\n---\n".join(r["content"] for r in rows) if rows else ""
    return {"memories": [dict(r) for r in rows], "context_block": block}

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
    db      = get_db()
    db.execute(
        "INSERT INTO api_keys (key_hash, key_prefix, plan, calls_limit, created_at, email) VALUES (?,?,?,?,?,?)",
        (h, prefix, req.plan, limit, time.time(), req.email)
    )
    db.commit()
    return {"api_key": raw_key, "prefix": prefix, "plan": req.plan,
            "calls_limit": limit, "email": req.email,
            "warning": "Store this key — it cannot be recovered"}

@app.get("/v1/keys/status")
async def key_status(record=Depends(require_key)):
    """Check your key's usage and plan."""
    plan   = PLANS.get(record["plan"], PLANS["free"])
    db     = get_db()
    recent = db.execute(
        "SELECT tool, COUNT(*) as n FROM call_log WHERE key_prefix=? GROUP BY tool",
        (record["key_prefix"],)
    ).fetchall()
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
        db         = get_db()
        db.execute(
            "INSERT INTO api_keys (key_hash, key_prefix, plan, calls_limit, created_at, email, stripe_customer_id) VALUES (?,?,?,?,?,?,?)",
            (h, prefix, plan, limit, time.time(), email, customer)
        )
        db.commit()
        print(f"New key created for {email} ({plan}): {raw_key}")

    elif event["type"] == "customer.subscription.deleted":
        customer = event["data"]["object"].get("customer","")
        db       = get_db()
        db.execute("UPDATE api_keys SET active=0 WHERE stripe_customer_id=?", (customer,))
        db.commit()

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
