import json
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32
from fastmcp import FastMCP

# Config
EMBED_URL = "http://localhost:9090/v1/embeddings"
EMBED_MODEL = "your-model-name"
EMBED_DIMS = 1024
DB_PATH = Path("/path/to/engram.db")
MCP_PORT = 9005

mcp = FastMCP(
    name="engram",
    instructions="Semantic memory store. Save memories to named collections and search them by meaning, not just keywords. Multiple agents can share the same infrastructure with separate collections."
)


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vecs USING vec0(
            embedding FLOAT[{EMBED_DIMS}]
        )
    """)
    db.commit()
    return db


def get_embedding(text: str) -> list[float]:
    data = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        EMBED_URL, data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Embedding API unavailable at {EMBED_URL}: {e.reason}") from e
    return resp["data"][0]["embedding"]


@mcp.tool()
def save_memory(collection: str, text: str) -> str:
    """Save a memory to a named collection with semantic embedding.
    Creates the collection automatically if it doesn't exist.

    Args:
        collection: Name of the collection (e.g. 'liv-memories', 'research-notes')
        text: The memory text to store
    """
    db = get_db()
    try:
        embedding = get_embedding(text)
    except RuntimeError as e:
        return str(e)
    cursor = db.execute(
        "INSERT INTO memories(collection, text) VALUES (?, ?)",
        [collection, text]
    )
    memory_id = cursor.lastrowid
    db.execute(
        "INSERT INTO memory_vecs(rowid, embedding) VALUES (?, ?)",
        [memory_id, serialize_float32(embedding)]
    )
    db.commit()
    return f"Saved memory #{memory_id} to collection '{collection}'"


@mcp.tool()
def search_memory(collection: str, query: str, top_k: int = 5) -> str:
    """Search a memory collection by semantic similarity.
    Finds memories by meaning, not just keyword matching.

    Args:
        collection: Name of the collection to search
        query: What you're looking for (in any language)
        top_k: Number of results to return (default 5)
    """
    db = get_db()
    try:
        embedding = get_embedding(query)
    except RuntimeError as e:
        return str(e)

    # Over-fetch from vec search, then filter by collection
    # (vec0 doesn't support filtering by joined table columns directly)
    results = db.execute("""
        SELECT m.id, m.collection, m.text, m.created_at, v.distance
        FROM memory_vecs v
        JOIN memories m ON m.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
    """, [serialize_float32(embedding), top_k * 10]).fetchall()

    filtered = [
        (id, text, created_at, dist)
        for id, coll, text, created_at, dist in results
        if coll == collection
    ][:top_k]

    if not filtered:
        return f"No memories found in collection '{collection}'"

    lines = []
    for id, text, created_at, dist in filtered:
        similarity = (2 - dist) / 2  # cosine distance 0-2 → similarity 0-1
        lines.append(f"[#{id} · {created_at}] (similarity: {similarity:.3f})\n{text}")

    return "\n\n---\n\n".join(lines)


@mcp.tool()
def list_collections() -> str:
    """List all memory collections with their memory counts and last update time."""
    db = get_db()
    results = db.execute("""
        SELECT collection, COUNT(*) as count, MAX(created_at) as last_updated
        FROM memories
        GROUP BY collection
        ORDER BY last_updated DESC
    """).fetchall()

    if not results:
        return "No collections yet"

    lines = [
        f"{coll}: {count} {'memory' if count == 1 else 'memories'} (last updated: {updated})"
        for coll, count, updated in results
    ]
    return "\n".join(lines)


def main():
    mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)


if __name__ == "__main__":
    main()
