from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, sqlite3
from datetime import datetime, timezone
import feedparser
import requests
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

# =========================
# App
# =========================
app = FastAPI(title="MetaView API", version="1.0.0")

# =========================
# Storage (SQLite)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /app/app
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))

def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        headline TEXT,
        content TEXT,
        source TEXT,
        url TEXT UNIQUE,
        published_at TEXT,
        created_at TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    con.commit()

init_db()

# =========================
# Models
# =========================
class IngestRequest(BaseModel):
    feeds: List[str]                 # RSS/Atom URLs
    limit_per_feed: int = 30

# =========================
# Helpers
# =========================
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_text(x: Any) -> str:
    return "" if x is None else str(x)

def parse_entry_content(entry: Dict[str, Any]) -> str:
    # RSS often has summary; keep it lightweight.
    if "summary" in entry:
        return safe_text(entry.get("summary"))
    if "description" in entry:
        return safe_text(entry.get("description"))
    # some feeds provide content list
    if "content" in entry and isinstance(entry["content"], list) and entry["content"]:
        return safe_text(entry["content"][0].get("value", ""))
    return ""

def to_iso_date(entry: Dict[str, Any]) -> str:
    # keep original string if exists; no heavy parsing.
    if entry.get("published"):
        return safe_text(entry.get("published"))
    if entry.get("updated"):
        return safe_text(entry.get("updated"))
    return ""

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    con = db()
    rows = con.execute("SELECT COUNT(*) AS c FROM articles").fetchone()["c"]
    return {
        "status": "ok",
        "service": "metaview",
        "rows": int(rows),
        "db_path": DB_PATH,
        "semantic_enabled": False,
        "time": utc_now()
    }

@app.post("/ingest/run")
def ingest_run(body: IngestRequest):
    """
    Lightweight ingestion from RSS feeds into SQLite.
    No heavy scraping, no ML models, no embeddings.
    """
    con = db()
    inserted_total = 0
    details = []

    for feed_url in body.feeds:
        t0 = datetime.now()
        inserted = 0
        err = None

        try:
            parsed = feedparser.parse(feed_url)
            if getattr(parsed, "bozo", False) and getattr(parsed, "bozo_exception", None):
                # still try reading entries; some feeds set bozo even when usable
                pass

            entries = parsed.entries[: max(0, int(body.limit_per_feed))]

            for e in entries:
                url = safe_text(e.get("link", "")).strip()
                if not url:
                    continue

                headline = safe_text(e.get("title", "")).strip()
                content = parse_entry_content(e).strip()
                published_at = to_iso_date(e)

                try:
                    con.execute("""
                    INSERT OR IGNORE INTO articles
                    (headline, content, source, url, published_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (headline, content, feed_url, url, published_at, utc_now()))

                    if con.total_changes > 0:
                        inserted += 1
                        inserted_total += 1
                except Exception:
                    # ignore single row insert issues
                    continue

            con.commit()

        except Exception as ex:
            err = str(ex)

        dt = (datetime.now() - t0).total_seconds()
        details.append({
            "feed": feed_url,
            "inserted": inserted,
            "time_sec": round(dt, 3),
            "error": err
        })

    return {"ok": True, "inserted_total": inserted_total, "details": details}

@app.get("/articles/latest")
def latest_articles(limit: int = 30, source: Optional[str] = None):
    con = db()
    limit = min(max(int(limit), 1), 200)

    if source:
        rows = con.execute("""
            SELECT id, headline, content, source, url, published_at
            FROM articles
            WHERE lower(source) = lower(?)
            ORDER BY id DESC
            LIMIT ?
        """, (source, limit)).fetchall()
    else:
        rows = con.execute("""
            SELECT id, headline, content, source, url, published_at
            FROM articles
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

    return {"count": len(rows), "results": [dict(r) for r in rows]}

@app.get("/search-text")
def search_text(
    q: str = Query(..., min_length=1),
    top_k: int = 20,
    source: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
):
    """
    Simple, fast SQL text search over headline+content with optional filters.
    Note: published_at is stored as text; date filtering is best-effort.
    """
    con = db()
    top_k = min(max(int(top_k), 1), 200)

    where = []
    params = []

    # text match (case-insensitive)
    where.append("(lower(headline) LIKE ? OR lower(content) LIKE ?)")
    ql = f"%{q.lower()}%"
    params += [ql, ql]

    if source:
        where.append("lower(source) = lower(?)")
        params.append(source)

    # best-effort date filtering (works if published_at contains ISO-like strings)
    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)
    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    sql = f"""
        SELECT id, headline, content, source, url, published_at
        FROM articles
        WHERE {" AND ".join(where)}
        ORDER BY id DESC
        LIMIT ?
    """
    params.append(top_k)

    rows = con.execute(sql, params).fetchall()
    return {"count": len(rows), "results": [dict(r) for r in rows]}
