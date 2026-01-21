# app/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple, List
import time
import threading
from datetime import datetime, timezone

from .config import (
    APP_NAME, APP_VERSION,
    AUTO_INGEST, AUTO_INGEST_EVERY_MIN,
    DB_PATH, MAX_ITEMS_PER_FEED
)
from .sources import DEFAULT_SOURCES
from .db import init_db, get_db, row_count
from .schemas import IngestRequest, SearchResponse
from .ingest import ingest_once

app = FastAPI(title=APP_NAME, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

ALLOWED_SORT_BY = {"published_at", "source", "country", "domain", "category", "entity"}
ALLOWED_SORT_DIR = {"asc", "desc"}


def _normalize_sort(sort_by: str, sort_dir: str) -> Tuple[str, str]:
    sb = (sort_by or "published_at").strip()
    sd = (sort_dir or "desc").strip().lower()
    if sb not in ALLOWED_SORT_BY:
        sb = "published_at"
    if sd not in ALLOWED_SORT_DIR:
        sd = "desc"
    return sb, sd


def _build_where(
    q: Optional[str],
    source: Optional[str],
    country: Optional[str],
    domain: Optional[str],
    category: Optional[str],
    entity: Optional[str],
    min_date: Optional[str],
    max_date: Optional[str],
) -> Tuple[str, list]:
    where = []
    params = []

    q_clean = (q or "").strip()
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
        where.append("entity LIKE ?")
        params.append(f"%{entity}%")

    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)

    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    return (" AND ".join(where) if where else "1=1"), params


def _count_where(where_sql: str, params: list) -> int:
    con = get_db()
    cur = con.cursor()
    cur.execute(f"SELECT COUNT(*) AS c FROM articles WHERE {where_sql}", params)
    c = int(cur.fetchone()["c"])
    con.close()
    return c


def _fetch(sql: str, params: list, include_content: bool):
    con = get_db()
    cur = con.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    con.close()

    results = []
    for r in rows:
        results.append({
            "headline": r["headline"],
            "content": (r["content"] if include_content else None),  # <-- ما يرجع content إلا لو طلبته
            "article_summary": r["article_summary"],
            "published_at": r["published_at"],
            "source": r["source"],
            "domain": r["domain"],
            "country": r["country"],
            "url": r["url"],
            "category": r["category"],
            "entity": r["entity"],
        })
    return results


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
    feeds = DEFAULT_SOURCES if (not body or not body.feeds) else body.feeds
    limit = MAX_ITEMS_PER_FEED if (not body) else int(body.limit_per_feed)
    return await ingest_once(feeds, limit_per_feed=limit)


@app.get("/articles", response_model=SearchResponse)
def list_articles(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    include_content: bool = Query(False),  # <-- مهم

    source: Optional[str] = None,
    country: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    entity: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,

    sort_by: str = Query("published_at"),
    sort_dir: str = Query("desc"),
):
    offset = (page - 1) * page_size
    sb, sd = _normalize_sort(sort_by, sort_dir)

    where_sql, params = _build_where(
        q=None,
        source=source, country=country, domain=domain,
        category=category, entity=entity,
        min_date=min_date, max_date=max_date
    )
    total = _count_where(where_sql, params)

    sql = f"""
    SELECT headline, content, article_summary, published_at, source, domain, country, url, category, entity
    FROM articles
    WHERE {where_sql}
    ORDER BY {sb} {sd}
    LIMIT ? OFFSET ?
    """
    results = _fetch(sql, params + [int(page_size), int(offset)], include_content)

    return {
        "count": len(results),
        "results": results,
        "total_rows": total,
        "returned": len(results),
        "page": page,
        "page_size": page_size,
        "offset": offset,
        "sort_by": sb,
        "sort_dir": sd,
    }


@app.get("/articles/")
def list_articles_slash(**kwargs):
    # نفس /articles بالضبط
    return list_articles(**kwargs)


@app.get("/search-text", response_model=SearchResponse)
def search_text(
    q: Optional[str] = Query(default=None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    include_content: bool = Query(False),  # <-- مهم

    source: Optional[str] = None,
    country: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    entity: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,

    sort_by: str = Query("published_at"),
    sort_dir: str = Query("desc"),
):
    offset = (page - 1) * page_size
    sb, sd = _normalize_sort(sort_by, sort_dir)

    where_sql, params = _build_where(
        q=q,
        source=source, country=country, domain=domain,
        category=category, entity=entity,
        min_date=min_date, max_date=max_date
    )
    total = _count_where(where_sql, params)

    sql = f"""
    SELECT headline, content, article_summary, published_at, source, domain, country, url, category, entity
    FROM articles
    WHERE {where_sql}
    ORDER BY {sb} {sd}
    LIMIT ? OFFSET ?
    """
    results = _fetch(sql, params + [int(page_size), int(offset)], include_content)

    return {
        "count": len(results),
        "results": results,
        "total_rows": total,
        "returned": len(results),
        "page": page,
        "page_size": page_size,
        "offset": offset,
        "sort_by": sb,
        "sort_dir": sd,
    }


def _auto_ingest_loop():
    time.sleep(5)
    while True:
        try:
            import asyncio
            asyncio.run(ingest_once(DEFAULT_SOURCES, limit_per_feed=MAX_ITEMS_PER_FEED))
        except Exception as e:
            print("AUTO_INGEST_ERROR:", str(e))
        time.sleep(max(60, AUTO_INGEST_EVERY_MIN * 60))


if AUTO_INGEST:
    t = threading.Thread(target=_auto_ingest_loop, daemon=True)
    t.start()
