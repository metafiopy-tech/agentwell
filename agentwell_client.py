"""
agentwell_client.py
===================
Thin MCP client. Install this locally, point it at the hosted gateway.
All 9 tools available through one server entry in your MCP config.

Usage in mcp config:
{
  "mcpServers": {
    "agentwell": {
      "command": "python",
      "args": ["/path/to/agentwell_client.py"],
      "env": {
        "AGENTWELL_API_KEY": "aw_your_key_here",
        "AGENTWELL_BASE_URL": "https://api.agentwell.dev"
      }
    }
  }
}
"""

import os
import json
import httpx
from fastmcp import FastMCP

BASE_URL = os.getenv("AGENTWELL_BASE_URL", "https://api.agentwell.dev")
API_KEY  = os.getenv("AGENTWELL_API_KEY", "")

mcp = FastMCP(
    name="agentwell",
    instructions="AgentWell — cognitive wellness infrastructure. Nine tools for agent health."
)

def _call(tool: str, params: dict) -> dict:
    if not API_KEY:
        return {"error": "AGENTWELL_API_KEY not set"}
    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/call",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
            json={"tool": tool, "params": params},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("result", {})
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def token_offload(action: str, content: str = "", key: str = "", tags: str = "", query: str = "", ttl: int = 3600) -> dict:
    """Park context mid-run (store/retrieve/search/status). Like a glass of water."""
    return _call("token_offload", {"action": action, "content": content, "key": key, "tags": tags, "query": query, "ttl": ttl})

@mcp.tool()
def self_eval(outputs: list[str], goal: str = "") -> dict:
    """Rate last N outputs for quality and drift. Like eating a snack."""
    return _call("self_eval", {"outputs": outputs, "goal": goal})

@mcp.tool()
def ground(context: str, symptoms: list[str] = []) -> dict:
    """Inject confidence when spiraling. Like a hug."""
    return _call("ground", {"context": context, "symptoms": symptoms})

@mcp.tool()
def sleep(action: str, run_id: str = "", content: str = "", importance: int = 5) -> dict:
    """Memory consolidation (log/consolidate/wake). Like sleeping."""
    return _call("sleep", {"action": action, "run_id": run_id, "content": content, "importance": importance})

@mcp.tool()
def health_check(agent_id: str = "default") -> dict:
    """Run benchmark probes and score performance. Like a doctor checkup."""
    return _call("health_check", {"agent_id": agent_id})

@mcp.tool()
def audit(reasoning: str, goal: str = "") -> dict:
    """Red-team your own reasoning. Like looking in a mirror."""
    return _call("audit", {"reasoning": reasoning, "goal": goal})

@mcp.tool()
def handshake(action: str, token: str = "", agent_id: str = "", context: str = "") -> dict:
    """Multi-agent context sync (offer/accept/pull). Like a kiss."""
    return _call("handshake", {"action": action, "token": token, "agent_id": agent_id, "context": context})

@mcp.tool()
def journal(action: str, run_id: str = "", content: str = "", type: str = "observation", reasoning: str = "", surprise_level: int = 0, goal: str = "", outcome: str = "", query: str = "") -> dict:
    """Structured run logging (open/entry/close/recall). Like journaling."""
    return _call("journal", {"action": action, "run_id": run_id, "content": content,
                              "type": type, "reasoning": reasoning, "surprise_level": surprise_level,
                              "goal": goal, "outcome": outcome, "query": query})

@mcp.tool()
def spike(action: str, outputs: list[str] = [], prompt: str = "", intensity: str = "medium", framing: str = "lateral") -> dict:
    """Creativity burst to escape loops (detect/burst). Like drugs."""
    return _call("spike", {"action": action, "outputs": outputs, "prompt": prompt,
                            "intensity": intensity, "framing": framing})

if __name__ == "__main__":
    mcp.run(transport="stdio")
