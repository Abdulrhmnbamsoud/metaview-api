# app/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time
import threading
from datetime import datetime, timezone

from .config import APP_NAME, APP_VERSION, AUTO_INGEST, AUTO_INGEST_EVERY_MIN, DB_PATH
from .sources import DEFAULT_SOURCES
from .db import init_db, get_db, row_count
from .schemas import IngestRequest, SearchResponse
from .ingest import ingest_once

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

@app.get("/")
def home():
    return {"service": "metaview", "status": "ok", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "metaview",
        "rows": row_count(),
        "db_path": DB_PATH,
        "semantic_enabled": True,  # جاهزية (بس التحليل يكمله Studio AI)
        "time": datetime.now(timezone.utc).isoformat()
    }

@app.get("/sources")
def list_sources():
    return {"count": len(DEFAULT_SOURCES), "sources": DEFAULT_SOURCES}

@app.post("/ingest/run")
async def ingest_run(body: Optional[IngestRequest] = None):
    feeds = DEFAULT_SOURCES if not body or not body.feeds else body.feeds
    return await ingest_once(feeds)

@app.get("/search-text", response_model=SearchResponse)
def search_text(
    q: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=200),
    source: Optional[str] = None,
    country: Optional[str] = None,
    domain: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
):
    con = get_db()
    cur = con.cursor()

    where = []
    params = []

    where.append("(headline LIKE ? OR article_summary LIKE ?)")
    params.extend([f"%{q}%", f"%{q}%"])

    if source:
        where.append("source = ?")
        params.append(source)

    if country:
        where.append("country = ?")
        params.append(country)

    if domain:
        where.append("domain = ?")
        params.append(domain)

    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)

    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    where_sql = " AND ".join(where) if where else "1=1"

    sql = f"""
    SELECT headline, content, article_summary, published_at, source, domain, country, url
    FROM articles
    WHERE {where_sql}
    ORDER BY published_at DESC
    LIMIT ?
    """
    params.append(int(top_k))

    cur.execute(sql, params)
    rows = cur.fetchall()
    con.close()

    results = []
    for r in rows:
        results.append({
            "headline": r["headline"],
            "content": r["content"],
            "article_summary": r["article_summary"],
            "published_at": r["published_at"],
            "source": r["source"],
            "domain": r["domain"],
            "country": r["country"],
            "url": r["url"],
        })

    return {"count": len(results), "results": results}

# =========================
# Auto ingest loop (optional)
# =========================
def _auto_ingest_loop():
    time.sleep(5)
    while True:
        try:
            import asyncio
            asyncio.run(ingest_once(DEFAULT_SOURCES))
        except Exception:
            pass
        time.sleep(max(60, AUTO_INGEST_EVERY_MIN * 60))

if AUTO_INGEST:
    t = threading.Thread(target=_auto_ingest_loop, daemon=True)
    t.start()
