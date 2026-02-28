import json
import random
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32
from fastmcp import FastMCP

# Config
EMBED_URL = "http://localhost:9090/v1/embeddings"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0-q4_k_m.gguf"
EMBED_DIMS = 1024
DB_PATH = Path("./engram.db")

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
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            text,
            tokenize='porter ascii'
        )
    """)
    # Migrate: populate FTS for any memories not yet indexed
    db.execute("""
        INSERT INTO memory_fts(rowid, text)
        SELECT id, text FROM memories
        WHERE id NOT IN (SELECT rowid FROM memory_fts)
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


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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
    db.execute(
        "INSERT INTO memory_fts(rowid, text) VALUES (?, ?)",
        [memory_id, text]
    )
    db.commit()
    return f"Saved memory #{memory_id} to collection '{collection}'"


@mcp.tool()
def search_memory(collection: str, query: str, top_k: int = 5) -> str:
    """Search a memory collection by semantic similarity and keyword matching.
    Combines vector search and full-text search via reciprocal rank fusion.

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

    # Vector search — over-fetch then filter by collection
    vec_results = db.execute("""
        SELECT m.id, m.collection
        FROM memory_vecs v
        JOIN memories m ON m.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
        ORDER BY v.distance
    """, [serialize_float32(embedding), top_k * 10]).fetchall()

    vec_ids = [id for id, coll in vec_results if coll == collection]

    # FTS search — filter by collection in join
    fts_results = db.execute("""
        SELECT m.id
        FROM memory_fts f
        JOIN memories m ON m.id = f.rowid
        WHERE f.text MATCH ?
          AND m.collection = ?
        ORDER BY rank
        LIMIT ?
    """, [query, collection, top_k * 10]).fetchall()

    fts_ids = [id for (id,) in fts_results]

    # Reciprocal rank fusion
    fused = reciprocal_rank_fusion([vec_ids, fts_ids])
    top_ids = [id for id, _ in fused[:top_k]]

    if not top_ids:
        return f"No memories found in collection '{collection}'"

    # Fetch full records in fused order
    placeholders = ",".join("?" * len(top_ids))
    rows = db.execute(
        f"SELECT id, text, created_at FROM memories WHERE id IN ({placeholders})",
        top_ids
    ).fetchall()

    # Re-sort to match fused order
    row_map = {id: (text, created_at) for id, text, created_at in rows}
    lines = []
    for id, score in fused[:top_k]:
        if id not in row_map:
            continue
        text, created_at = row_map[id]
        lines.append(f"[#{id} · {created_at}] (score: {score:.3f})\n{text}")

    return "\n\n---\n\n".join(lines)


@mcp.tool()
def randomly_remember(collection: str, age: str = "any") -> str:
    """Retrieve a random memory from a collection, optionally weighted by age.

    Args:
        collection: Name of the collection
        age: 'recent' to draw from newest memories, 'old' to draw from oldest,
             'any' for fully random (default)
    """
    db = get_db()

    if age == "recent":
        rows = db.execute("""
            SELECT id, text, created_at FROM memories
            WHERE collection = ?
            ORDER BY created_at DESC
            LIMIT 20
        """, [collection]).fetchall()
    elif age == "old":
        rows = db.execute("""
            SELECT id, text, created_at FROM memories
            WHERE collection = ?
            ORDER BY created_at ASC
            LIMIT 20
        """, [collection]).fetchall()
    else:
        rows = db.execute("""
            SELECT id, text, created_at FROM memories
            WHERE collection = ?
            ORDER BY RANDOM()
            LIMIT 1
        """, [collection]).fetchall()

    if not rows:
        return f"No memories found in collection '{collection}'"

    id, text, created_at = random.choice(rows)
    return f"[#{id} · {created_at}]\n{text}"


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
    mcp.run(transport="http", host="0.0.0.0", port=9005)


if __name__ == "__main__":
    main()
