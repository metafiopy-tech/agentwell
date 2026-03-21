# AgentWell

**Cognitive wellness infrastructure for AI agents.**

Agents are the new workforce. Nobody is feeding them.

AgentWell is a hosted API giving AI agents access to nine wellness tools — the things agents need to run well that nobody has built yet. One API key. Growing library. Agents pay per call. Humans pay monthly.

---

## The human parallel

| Human need | Agent parallel | Tool |
|---|---|---|
| Glass of water | Context decay | `token_offload` |
| Eating a snack | Attention drift | `self_eval` |
| A hug | Hallucination spiral | `ground` |
| Sleep | No rest state | `sleep` |
| Doctor checkup | Performance variance | `health_check` |
| Looking in a mirror | Blind spots | `audit` |
| A kiss | Coordination overhead | `handshake` |
| Journaling | No episodic memory | `journal` |
| Drugs | Creative deadlock | `spike` |

---

## Quick start

```bash
pip install agentwell-client
```

```python
# In your MCP config
{
  "mcpServers": {
    "agentwell": {
      "command": "python",
      "args": ["-m", "agentwell_client"],
      "env": {
        "AGENTWELL_API_KEY": "aw_your_key_here"
      }
    }
  }
}
```

Get a key at **agentwell.dev/pricing**

---

## API

All tools go through one endpoint:

```bash
curl -X POST https://api.agentwell.dev/v1/call \
  -H "X-API-Key: aw_your_key" \
  -H "Content-Type: application/json" \
  -d '{"tool": "self_eval", "params": {"outputs": ["step 1 result", "step 2 result"]}}'
```

### token_offload
Park context mid-run. Get a key back. Retrieve later.
```json
{"tool": "token_offload", "params": {"action": "store", "content": "...", "tags": "background,step3"}}
{"tool": "token_offload", "params": {"action": "retrieve", "key": "abc12345"}}
{"tool": "token_offload", "params": {"action": "search", "query": "user preferences"}}
```

### self_eval
Rate last N outputs before drift compounds.
```json
{"tool": "self_eval", "params": {"outputs": ["output1", "output2"], "goal": "write a report"}}
```
Returns: `confidence`, `weakest`, `flags`, `recommendation`

### ground
Inject confidence when the agent is spiraling.
```json
{"tool": "ground", "params": {"context": "agent recent output...", "symptoms": ["over-caveating"]}}
```
Returns: `spiral_score`, `grounding_block` (inject into next prompt), `needs_grounding`

### sleep
Consolidate episodic memory into semantic facts.
```json
{"tool": "sleep", "params": {"action": "log", "run_id": "r1", "content": "learned X", "importance": 8}}
{"tool": "sleep", "params": {"action": "consolidate", "run_id": "r1"}}
{"tool": "sleep", "params": {"action": "wake"}}
```

### health_check
Run benchmark probes and score the agent.
```json
{"tool": "health_check", "params": {"agent_id": "my_agent"}}
```
Returns: `score`, `grade`, `probes`, `flags`

### audit
Red-team the agent's own reasoning.
```json
{"tool": "audit", "params": {"reasoning": "my plan is X because Y...", "goal": "deploy a feature"}}
```
Returns: `vulnerabilities`, `strongest_challenge`, `safe_to_proceed`

### handshake
Sync two agents briefly.
```json
{"tool": "handshake", "params": {"action": "offer", "agent_id": "agent_a", "context": "..."}}
{"tool": "handshake", "params": {"action": "accept", "token": "...", "agent_id": "agent_b", "context": "..."}}
{"tool": "handshake", "params": {"action": "pull", "token": "...", "agent_id": "agent_a"}}
```

### journal
Structured run logging with lesson extraction.
```json
{"tool": "journal", "params": {"action": "open", "run_id": "r1", "goal": "..."}}
{"tool": "journal", "params": {"action": "entry", "run_id": "r1", "type": "decision", "content": "...", "reasoning": "..."}}
{"tool": "journal", "params": {"action": "close", "run_id": "r1", "outcome": "success"}}
{"tool": "journal", "params": {"action": "recall", "query": "similar task"}}
```

### spike
Break out of a creative loop.
```json
{"tool": "spike", "params": {"action": "detect", "outputs": ["out1", "out2", "out3"]}}
{"tool": "spike", "params": {"action": "burst", "prompt": "...", "intensity": "high", "framing": "lateral"}}
```

---

## Pricing

| Plan | Price | Calls/month |
|---|---|---|
| Free | $0 | 100 |
| Dev | $29/mo | 20,000 |
| Agency | $199/mo | Unlimited |
| Pay-as-you-go | $0.002/call | — |
| Crypto (USDC) | available | Base network |

---

## Self-hosting

```bash
git clone https://github.com/metafiopy-tech/agentwell
cd agentwell
cp .env.example .env
# fill in your keys
pip install -r requirements.txt
uvicorn gateway:app --host 0.0.0.0 --port 8000
```

---

## Philosophy

Humans built an entire economy around human limitations. Every limitation became a commodity market. Agents have analogous limitations — context decay, attention drift, no rest state, blind spots, coordination overhead — and almost nobody is building for them yet.

AgentWell is the picks-and-shovels play for the agentic era.

---

**agentwell.dev** · [@metafiopy](https://twitter.com/metafiopy-tech) · [Discord](https://discord.gg/agentwell)
