"""
rollback_mcp.py
===============
Generic state snapshot and restore for self-modifying agents.
Before any self-modification, snapshot. If it breaks, restore.

Unlike bespoke backup/restore code baked into each system,
this is a standalone MCP tool any self-improving agent can use.

Works with:
- Python files
- JSON state files
- Any directory
- Arbitrary content blobs

Usage:
    python rollback_mcp.py
"""

import os
import json
import time
import shutil
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from fastmcp import FastMCP

DB_PATH      = Path.home() / ".rollback.db"
SNAPSHOT_DIR = Path.home() / ".rollback_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                snapshot_id TEXT UNIQUE NOT NULL,
                agent_id    TEXT DEFAULT '',
                label       TEXT DEFAULT '',
                paths       TEXT NOT NULL,
                snapshot_dir TEXT NOT NULL,
                restored    INTEGER DEFAULT 0,
                valid       INTEGER DEFAULT 1
            );
        """)
        _local.conn.commit()
    return _local.conn

def _make_id():
    return f"snap_{int(time.time())}_{os.getpid()}"

mcp = FastMCP(
    name="rollback",
    instructions=(
        "Generic snapshot and restore for self-modifying agents. "
        "Call snapshot() before any self-modification. "
        "Call restore() if validation fails. "
        "Works with Python files, JSON state, or any directory."
    )
)

@mcp.tool()
def snapshot(
    paths: list[str],
    agent_id: str = "",
    label: str = "",
    run_id: str = ""
) -> dict:
    """
    Snapshot files or directories before modification.
    Returns a snapshot_id you can use to restore later.

    Args:
        paths:    List of file or directory paths to snapshot
        agent_id: Agent taking the snapshot
        label:    Human-readable label e.g. "before error handling patch"
        run_id:   Optional run tag

    Returns:
        { snapshot_id, files_snapped, snapshot_dir, size_bytes }
    """
    snap_id  = _make_id()
    snap_dir = SNAPSHOT_DIR / snap_id
    snap_dir.mkdir(parents=True, exist_ok=True)

    snapped  = []
    total_sz = 0

    for path_str in paths:
        path = Path(path_str).expanduser()
        if not path.exists():
            continue

        if path.is_file():
            dest = snap_dir / path.name
            shutil.copy2(path, dest)
            snapped.append(str(path))
            total_sz += path.stat().st_size

        elif path.is_dir():
            dest = snap_dir / path.name
            shutil.copytree(path, dest, dirs_exist_ok=True)
            snapped.append(str(path))
            for f in path.rglob("*"):
                if f.is_file():
                    total_sz += f.stat().st_size

    db = get_db()
    db.execute(
        "INSERT INTO snapshots (ts, snapshot_id, agent_id, label, paths, snapshot_dir) VALUES (?,?,?,?,?,?)",
        (time.time(), snap_id, agent_id, label, json.dumps(snapped), str(snap_dir))
    )
    db.commit()

    return {
        "snapshot_id": snap_id,
        "files_snapped": len(snapped),
        "paths": snapped,
        "snapshot_dir": str(snap_dir),
        "size_bytes": total_sz,
        "label": label
    }


@mcp.tool()
def restore(
    snapshot_id: str,
    dry_run: bool = False
) -> dict:
    """
    Restore files from a snapshot.

    Args:
        snapshot_id: The ID from snapshot()
        dry_run:     If True, shows what would be restored without doing it

    Returns:
        { restored, files_restored, snapshot_id, dry_run }
    """
    db  = get_db()
    row = db.execute(
        "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
    ).fetchone()

    if not row:
        return {"error": f"Snapshot '{snapshot_id}' not found"}

    snap_dir = Path(row["snapshot_dir"])
    if not snap_dir.exists():
        return {"error": f"Snapshot directory missing: {snap_dir}"}

    original_paths = json.loads(row["paths"])
    restored = []

    for orig_str in original_paths:
        orig = Path(orig_str)
        snapped_name = orig.name
        snapped_path = snap_dir / snapped_name

        if not snapped_path.exists():
            continue

        if dry_run:
            restored.append(f"WOULD RESTORE: {snapped_path} → {orig}")
            continue

        if snapped_path.is_file():
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapped_path, orig)
            restored.append(str(orig))
        elif snapped_path.is_dir():
            if orig.exists():
                shutil.rmtree(orig)
            shutil.copytree(snapped_path, orig)
            restored.append(str(orig))

    if not dry_run:
        db.execute(
            "UPDATE snapshots SET restored=1 WHERE snapshot_id=?",
            (snapshot_id,)
        )
        db.commit()

    return {
        "restored": not dry_run,
        "files_restored": len(restored),
        "snapshot_id": snapshot_id,
        "paths": restored,
        "dry_run": dry_run
    }


@mcp.tool()
def validate_and_restore(
    snapshot_id: str,
    validation_results: dict
) -> dict:
    """
    Conditionally restore based on validation results.
    Pass in your validation output — if it indicates failure,
    automatically triggers a restore.

    Args:
        snapshot_id:         The snapshot to potentially restore
        validation_results:  Dict with at minimum {"valid": bool, "errors": [...]}

    Returns:
        { action_taken, restored, reason }
    """
    valid = validation_results.get("valid", True)
    errors = validation_results.get("errors", [])

    if valid and not errors:
        return {
            "action_taken": "none",
            "restored": False,
            "reason": "Validation passed — no restore needed"
        }

    result = restore(snapshot_id)
    return {
        "action_taken": "restored",
        "restored": result.get("restored", False),
        "files_restored": result.get("files_restored", 0),
        "reason": f"Validation failed: {errors[:3] if errors else 'unknown errors'}",
        "snapshot_id": snapshot_id
    }


@mcp.tool()
def list_snapshots(
    agent_id: str = "",
    limit: int = 20
) -> dict:
    """
    List available snapshots.

    Args:
        agent_id: Filter by agent. Empty = all.
        limit:    Max to return.

    Returns:
        { snapshots: [...] }
    """
    db = get_db()
    if agent_id:
        rows = db.execute(
            "SELECT * FROM snapshots WHERE agent_id=? ORDER BY ts DESC LIMIT ?",
            (agent_id, limit)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM snapshots ORDER BY ts DESC LIMIT ?",
            (limit,)
        ).fetchall()

    return {
        "snapshots": [
            {
                "snapshot_id": r["snapshot_id"],
                "ts": int(r["ts"]),
                "label": r["label"],
                "agent_id": r["agent_id"],
                "restored": bool(r["restored"]),
                "valid": bool(r["valid"])
            }
            for r in rows
        ],
        "count": len(rows)
    }


@mcp.tool()
def cleanup(
    keep_last: int = 10,
    agent_id: str = ""
) -> dict:
    """
    Delete old snapshots to free disk space.
    Keeps the most recent N snapshots.

    Args:
        keep_last: Number of recent snapshots to keep per agent. Default 10.
        agent_id:  Scope to specific agent. Empty = all agents.

    Returns:
        { deleted, freed_bytes }
    """
    db = get_db()

    if agent_id:
        rows = db.execute(
            "SELECT snapshot_id, snapshot_dir FROM snapshots WHERE agent_id=? ORDER BY ts DESC",
            (agent_id,)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT snapshot_id, snapshot_dir FROM snapshots ORDER BY ts DESC"
        ).fetchall()

    to_delete = rows[keep_last:]
    deleted = 0
    freed = 0

    for row in to_delete:
        snap_dir = Path(row["snapshot_dir"])
        if snap_dir.exists():
            sz = sum(f.stat().st_size for f in snap_dir.rglob("*") if f.is_file())
            freed += sz
            shutil.rmtree(snap_dir)
        db.execute("DELETE FROM snapshots WHERE snapshot_id=?", (row["snapshot_id"],))
        deleted += 1

    db.commit()
    return {"deleted": deleted, "freed_bytes": freed}


if __name__ == "__main__":
    print("rollback MCP running...")
    print(f"DB: {DB_PATH}")
    print(f"Snapshots: {SNAPSHOT_DIR}")
    mcp.run(transport="stdio")
