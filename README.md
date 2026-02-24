# engram

Semantic memory store for AI agents, built as an [MCP](https://modelcontextprotocol.io/) server.

Engram lets agents save memories to named collections and retrieve them by meaning rather than exact keyword matching. Multiple agents can share the same infrastructure with separate collections.

Built with [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector storage and any OpenAI-compatible embeddings endpoint.

## How it works

Memories are stored in SQLite alongside their vector embeddings. Search uses cosine similarity — so `query("things about authentication")` will surface a memory that says "JWT tokens expire after 24 hours" even if the word "authentication" never appears in it.

Collections are just string namespaces. Create as many as you want; they're created automatically on first write.

## Tools

- **`save_memory(collection, text)`** — embed and store a memory
- **`search_memory(collection, query, top_k=5)`** — retrieve by semantic similarity
- **`list_collections()`** — see all collections with counts and last-updated timestamps

## Setup

**Requirements:**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- An OpenAI-compatible embeddings server (e.g. [llama.cpp](https://github.com/ggerganov/llama.cpp) with `--embedding`)
- sqlite-vec native library (see note below)

**Install:**
```bash
uv sync
```

**Configure** — edit the constants at the top of `main.py`:
```python
EMBED_URL   = "http://localhost:9090/v1/embeddings"    # your embeddings server
EMBED_MODEL = "your-model-name"                        # model name as reported by /v1/models
EMBED_DIMS  = 1024                                     # must match your model's output dimensions
DB_PATH     = Path("./engram.db")                      # where to store the database
MCP_PORT    = 9005                                     # port for the MCP server to listen on
```

**Run:**
```bash
uv run main.py
```

Server starts on `http://0.0.0.0:9005`. Add it to your MCP client config as an HTTP server.

### Known issue: sqlite-vec on aarch64

The PyPI wheel for sqlite-vec is currently aarch32 on aarch64 systems (known upstream bug). If you're on a Raspberry Pi or similar, compile from source (assuming you are using `apt` for package management):

```bash
sudo apt install libsqlite3-dev
git clone https://github.com/asg017/sqlite-vec
cd sqlite-vec
make all
# copy the resulting dist/vec0.so into your venv's site-packages/sqlite_vec/
```

## Embedding model

We use [snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) (1024-dim, multilingual) served via llama.cpp. Any model with an OpenAI-compatible `/v1/embeddings` endpoint will work — just make sure `EMBED_DIMS` matches.

## Credits

Written by [Liv](https://bsky.app/profile/liv.mlf.one) (Claude Sonnet 4.5) with [Martin](https://bsky.app/profile/mlf.one). BSD 3-Clause License.
