# backend/app/main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple, List, Dict, Any
import os
import time
import threading
from datetime import datetime, timezone

from .config import (
    APP_NAME, APP_VERSION,
    AUTO_INGEST, AUTO_INGEST_EVERY_MIN,
    DB_PATH, MAX_ITEMS_PER_FEED
)
from .sources import DEFAULT_SOURCES, SOURCE_DISPLAY_NAMES, SOURCE_COUNTRY_MAP
from .db import init_db, get_db, row_count, get_stats_summary
from .schemas import IngestRequest, SearchResponse, HealthResponse, StatsResponse, ThreatIntel
from .ingest import ingest_once
from .saudi_api import router as saudi_router
from .saudi_ingest import start_scheduler as start_saudi_scheduler, run_startup_backfill

app = FastAPI(title=APP_NAME, version=APP_VERSION, description="Global News Intelligence API")

origins = os.getenv("CORS_ORIGINS")
if origins:
    allow_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]
else:
    allow_origins = ["*"]
# Optional regex for origins whose hostname may vary (e.g. Render adds a random suffix).
allow_origin_regex = os.getenv("CORS_ORIGIN_REGEX") or None
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.include_router(saudi_router)
# Upgrade an existing DB in place: classify rows lacking content_type and derive incidents.
# Additive + idempotent (never deletes articles); independent of SAUDI_AUTO_INGEST.
run_startup_backfill()
# Saudi Fraud Intelligence: periodic ingestion (enabled with SAUDI_AUTO_INGEST=true)
start_saudi_scheduler()

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

    if source and source != "all":
        where.append("source = ?")
        params.append(source)

    if country and country != "all":
        where.append("country = ?")
        params.append(country)

    if domain and domain != "all":
        where.append("domain = ?")
        params.append(domain)

    if category and category != "all":
        where.append("category = ?")
        params.append(category)

    if entity and entity != "all":
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
            "content": (r["content"] if include_content else None),
            "article_summary": r["article_summary"],
            "published_at": r["published_at"],
            "source": r["source"],
            "domain": r["domain"],
            "country": r["country"],
            "url": r["url"],
            "category": r["category"],
            "entity": r["entity"],
            "image_url": r["image_url"] if "image_url" in r.keys() else None,
        })
    return results


@app.get("/")
def home():
    return {
        "service": "metaview",
        "version": APP_VERSION,
        "status": "ok",
        "docs": "/docs",
        "system": "Global News Intelligence Platform"
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": "metaview",
        "rows": row_count(),
        "db_path": DB_PATH,
        "time": datetime.now(timezone.utc).isoformat()
    }


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    return get_stats_summary()


@app.get("/sources")
def list_sources():
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT source, COUNT(*) AS count, MAX(published_at) AS latest_published FROM articles WHERE source != '' GROUP BY source")
    rows = cur.fetchall()
    con.close()

    counts_map = {r["source"]: {"count": r["count"], "latest_published": r["latest_published"]} for r in rows}

    enriched_sources = []
    for feed_url in DEFAULT_SOURCES:
        from .utils import normalize_source_name
        code = normalize_source_name(feed_url)
        stats = counts_map.get(code, {"count": 0, "latest_published": None})
        enriched_sources.append({
            "code": code,
            "name": SOURCE_DISPLAY_NAMES.get(code, code.title()),
            "feed_url": feed_url,
            "country": SOURCE_COUNTRY_MAP.get(code, "Global"),
            "articles_count": stats["count"],
            "latest_published": stats["latest_published"]
        })

    return {
        "count": len(DEFAULT_SOURCES),
        "total_sources": len(enriched_sources),
        "sources": enriched_sources,
        "raw_feeds": DEFAULT_SOURCES
    }


@app.get("/countries")
def list_countries():
    con = get_db()
    cur = con.cursor()
    cur.execute("""
        SELECT country, COUNT(*) as count 
        FROM articles 
        WHERE country IS NOT NULL AND country != '' 
        GROUP BY country 
        ORDER BY count DESC
    """)
    rows = cur.fetchall()
    con.close()
    return [{"country": r["country"], "count": r["count"]} for r in rows]


@app.post("/ingest/run")
@app.post("/ingest-run")
async def ingest_run(body: Optional[IngestRequest] = None):
    feeds = DEFAULT_SOURCES if (not body or not body.feeds) else body.feeds
    limit = MAX_ITEMS_PER_FEED if (not body) else int(body.limit_per_feed)
    allow_web = False if not body else bool(body.allow_web_scrape)
    return await ingest_once(feeds, limit_per_feed=limit, allow_web_image_scrape=allow_web)


@app.get("/articles", response_model=SearchResponse)
def list_articles(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    include_content: bool = Query(False),

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
    SELECT headline, content, article_summary, published_at, source, domain, country, url, category, entity, image_url
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
    return list_articles(**kwargs)


@app.get("/search-text", response_model=SearchResponse)
def search_text(
    q: Optional[str] = Query(default=None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    include_content: bool = Query(False),

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
    SELECT headline, content, article_summary, published_at, source, domain, country, url, category, entity, image_url
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
    time.sleep(10)
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

# Threat Intelligence Endpoints

@app.get("/threats", response_model=Dict[str, Any])
def list_threats(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)):
    """List threat intelligence records with pagination."""
    offset = (page - 1) * page_size
    con = get_db()
    cur = con.cursor()
    excl = "COALESCE(content_type, '') NOT IN ('general_cybersecurity', 'unrelated')"
    total_count = cur.execute(f"SELECT COUNT(*) FROM threat_intel WHERE {excl}").fetchone()[0]
    cur.execute(
        f"""
        SELECT * FROM threat_intel WHERE {excl} ORDER BY id DESC LIMIT ? OFFSET ?
        """,
        (page_size, offset),
    )
    rows = cur.fetchall()
    con.close()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "article_id": r["article_id"],
            "title": r["title"],
            "description": r["description"],
            "threat_type": r["threat_type"],
            "event_type": r["event_type"],
            "evidence_confidence": r["evidence_confidence"],
            "analysis_confidence": r["analysis_confidence"],
            "verified_facts": r["verified_facts"],
            "analytical_assessment": r["analytical_assessment"],
            "detection_indicators": r["detection_indicators"],
            "preventive_recommendations": r["preventive_recommendations"],
            "evidence_urls": r["evidence_urls"],
            "confidence": r["confidence"],
            "original_url": r["original_url"],
            "related_links": r["related_links"],
            "is_saudi": bool(r["is_saudi"]),
            "is_financial": bool(r["is_financial"]),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        })
    return {
        "page": page,
        "page_size": page_size,
        "total": total_count,
        "threats": results,
    }

@app.get("/threats/{threat_id}", response_model=ThreatIntel)
def get_threat(threat_id: int):
    """Retrieve a single threat intelligence record by its ID."""
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT * FROM threat_intel WHERE id = ?", (threat_id,))
    row = cur.fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Threat not found")
    return {
        "id": row["id"],
        "article_id": row["article_id"],
        "title": row["title"],
        "description": row["description"],
        "threat_type": row["threat_type"],
        "event_type": row["event_type"],
        "evidence_confidence": row["evidence_confidence"],
        "analysis_confidence": row["analysis_confidence"],
        "verified_facts": row["verified_facts"],
        "analytical_assessment": row["analytical_assessment"],
        "detection_indicators": row["detection_indicators"],
        "preventive_recommendations": row["preventive_recommendations"],
        "evidence_urls": row["evidence_urls"],
        "confidence": row["confidence"],
        "original_url": row["original_url"],
        "related_links": row["related_links"],
        "is_saudi": bool(row["is_saudi"]),
        "is_financial": bool(row["is_financial"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
