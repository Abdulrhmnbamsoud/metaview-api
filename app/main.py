# app/main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Literal
import time
import threading
from datetime import datetime, timezone

from .config import APP_NAME, APP_VERSION, AUTO_INGEST, AUTO_INGEST_EVERY_MIN, DB_PATH
from .sources import DEFAULT_SOURCES
from .db import init_db, get_db, row_count
from .schemas import IngestRequest
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

def _fetch(sql: str, params: list):
    con = get_db()
    cur = con.cursor()
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
            "category": r.get("category") if hasattr(r, "get") else r["category"],
            "entities": r.get("entities") if hasattr(r, "get") else r["entities"],
        })
    return results

def _count(where_sql: str, params: list) -> int:
    con = get_db()
    cur = con.cursor()
    cur.execute(f"SELECT COUNT(1) as total FROM articles WHERE {where_sql}", params)
    total = int(cur.fetchone()["total"])
    con.close()
    return total


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
        "time": datetime.now(timezone.utc).isoformat()
    }

@app.get("/sources")
def list_sources():
    return {"count": len(DEFAULT_SOURCES), "sources": DEFAULT_SOURCES}

@app.post("/ingest/run")
async def ingest_run(body: Optional[IngestRequest] = None):
    feeds = DEFAULT_SOURCES if not body or not body.feeds else body.feeds
    return await ingest_once(feeds)

# ✅ Latest/Articles endpoint (Studio-friendly)
@app.get("/articles")
def list_articles(
    # pagination style
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),

    # filters
    source: Optional[str] = None,
    country: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    entity: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,

    # sort
    sort: Literal["newest", "oldest"] = "newest",
):
    where = []
    params = []

    if source:
        where.append("source = ?")
        params.append(source)
    if country:
        where.append("country = ?")
        params.append(country)
    if domain:
        where.append("domain = ?")
        params.append(domain)
    if category:
        where.append("category = ?")
        params.append(category)
    if entity:
        where.append("(entities LIKE ? OR headline LIKE ? OR content LIKE ?)")
        params.extend([f"%{entity}%", f"%{entity}%", f"%{entity}%"])
    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)
    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    where_sql = " AND ".join(where) if where else "1=1"
    total = _count(where_sql, params)

    offset = (page - 1) * page_size
    if total > 0 and offset >= total:
        raise HTTPException(status_code=422, detail="page is out of range")

    order_sql = "published_at DESC" if sort == "newest" else "published_at ASC"

    sql = f"""
    SELECT headline, content, article_summary, published_at, source, domain, country, url, category, entities
    FROM articles
    WHERE {where_sql}
    ORDER BY {order_sql}
    LIMIT ? OFFSET ?
    """
    results = _fetch(sql, params + [int(page_size), int(offset)])

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "sort": sort,
        "returned": len(results),
        "results": results
    }

# ✅ Search endpoint (category/entity/sort/pagination)
@app.get("/search-text")
def search_text(
    q: Optional[str] = Query(default=None),

    # pagination
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),

    # filters
    source: Optional[str] = None,
    country: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    entity: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,

    # sort
    sort: Literal["newest", "oldest", "relevance"] = "newest",
):
    where = []
    params = []
    order_params = []

    q_clean = (q or "").strip()

    # إذا فيه q نبحث
    if q_clean:
        where.append("(headline LIKE ? OR article_summary LIKE ? OR content LIKE ?)")
        params.extend([f"%{q_clean}%", f"%{q_clean}%", f"%{q_clean}%"])

    if source:
        where.append("source = ?")
        params.append(source)
    if country:
        where.append("country = ?")
        params.append(country)
    if domain:
        where.append("domain = ?")
        params.append(domain)
    if category:
        where.append("category = ?")
        params.append(category)
    if entity:
        where.append("(entities LIKE ? OR headline LIKE ? OR content LIKE ?)")
        params.extend([f"%{entity}%", f"%{entity}%", f"%{entity}%"])
    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)
    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    where_sql = " AND ".join(where) if where else "1=1"
    total = _count(where_sql, params)

    offset = (page - 1) * page_size
    if total > 0 and offset >= total:
        raise HTTPException(status_code=422, detail="page is out of range")

    # sort
    if sort == "newest":
        order_sql = "published_at DESC"
    elif sort == "oldest":
        order_sql = "published_at ASC"
    else:
        # relevance بسيط (بدون FTS)
        order_sql = """
        (CASE
            WHEN headline LIKE ? THEN 3
            WHEN article_summary LIKE ? THEN 2
            WHEN content LIKE ? THEN 1
            ELSE 0
         END) DESC,
         published_at DESC
        """
        order_params = [f"%{q_clean}%", f"%{q_clean}%", f"%{q_clean}%"]

    sql = f"""
    SELECT headline, content, article_summary, published_at, source, domain, country, url, category, entities
    FROM articles
    WHERE {where_sql}
    ORDER BY {order_sql}
    LIMIT ? OFFSET ?
    """
    results = _fetch(sql, params + order_params + [int(page_size), int(offset)])

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "sort": sort,
        "returned": len(results),
        "results": results
    }

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
