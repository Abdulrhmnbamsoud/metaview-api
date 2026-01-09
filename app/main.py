from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import sqlite3
from datetime import datetime, timezone
import threading
import asyncio

import httpx
import feedparser

import trafilatura
from bs4 import BeautifulSoup

# =========================
# Config
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))              # .../app/app
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
USER_AGENT = os.getenv("USER_AGENT", "MetaViewBot/1.0 (+contact: you@example.com)")
AUTO_INGEST = os.getenv("AUTO_INGEST", "false").lower() == "true"
AUTO_INGEST_EVERY_MIN = int(os.getenv("AUTO_INGEST_EVERY_MIN", "15"))

# Concurrency limits (important for Railway stability)
RSS_CONCURRENCY = int(os.getenv("RSS_CONCURRENCY", "6"))
FULLTEXT_CONCURRENCY = int(os.getenv("FULLTEXT_CONCURRENCY", "5"))

# =========================
# Sources (RSS) - expand if you want more volume
# =========================
DEFAULT_SOURCES = [
    "http://rss.cnn.com/rss/cnn_world.rss",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.nbcnews.com/feeds/worldnews",
    "http://feeds.abcnews.com/abcnews/internationalheadlines",
    "https://www.cbsnews.com/latest/rss/world",
    "https://rss.dw.com/rdf/rss-en-world",
    "https://www.france24.com/en/rss",
    "https://www.lemonde.fr/rss/une.xml",
    "https://www.theguardian.com/world/rss",
    "http://feeds.washingtonpost.com/rss/world",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.economist.com/international/rss.xml",
    "https://www.foreignaffairs.com/rss.xml",
    "https://thehill.com/feed/",
    "https://www.axios.com/feeds/feed.rss",
    "https://www.newsweek.com/rss",
    "https://www.businessinsider.com/rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",  # WSJ: غالبًا paywall للنص الكامل
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",  # Bloomberg: غالبًا paywall
    "https://www.ft.com/world/rss",  # FT: sometimes 404
]

# =========================
# DB helpers
# =========================
def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db() -> None:
    con = get_db()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        domain TEXT,
        country TEXT,
        headline TEXT,
        content TEXT,
        article_summary TEXT,
        url TEXT UNIQUE,
        published_at TEXT,
        created_at TEXT,
        content_fetched INTEGER DEFAULT 0,
        content_error TEXT DEFAULT ''
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_content_fetched ON articles(content_fetched)")
    con.commit()
    con.close()

def row_count() -> int:
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM articles")
    c = int(cur.fetchone()["c"])
    con.close()
    return c

def upsert_article(item: Dict[str, Any]) -> bool:
    con = get_db()
    cur = con.cursor()
    try:
        cur.execute("""
        INSERT OR IGNORE INTO articles
        (source, domain, country, headline, content, article_summary, url, published_at, created_at, content_fetched, content_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item.get("source", ""),
            item.get("domain", ""),
            item.get("country", ""),
            item.get("headline", ""),
            item.get("content", ""),
            item.get("article_summary", ""),
            item.get("url", ""),
            item.get("published_at", ""),
            item.get("created_at", ""),
            int(item.get("content_fetched", 0)),
            item.get("content_error", ""),
        ))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()

def update_fulltext(url: str, content: str, err: str = "") -> None:
    con = get_db()
    cur = con.cursor()
    cur.execute("""
        UPDATE articles
        SET content = ?, content_fetched = ?, content_error = ?
        WHERE url = ?
    """, (content, 1 if content else 0, err[:500], url))
    con.commit()
    con.close()

def fetch_missing_urls(limit: int = 50) -> List[str]:
    con = get_db()
    cur = con.cursor()
    cur.execute("""
        SELECT url FROM articles
        WHERE (content IS NULL OR content = '') AND content_fetched = 0
        ORDER BY published_at DESC
        LIMIT ?
    """, (int(limit),))
    rows = cur.fetchall()
    con.close()
    return [r["url"] for r in rows if r["url"]]

# =========================
# Helpers
# =========================
def normalize_source_name(feed_url: str) -> str:
    u = feed_url.lower()
    if "reuters" in u: return "reuters"
    if "bloomberg" in u: return "bloomberg"
    if "dj.com" in u or "wsj" in u: return "wsj"
    if "nytimes" in u: return "nyt"
    if "economist" in u: return "economist"
    if "cnn" in u: return "cnn"
    if "ft.com" in u: return "financial_times"
    if "washingtonpost" in u: return "washington_post"
    if "theatlantic" in u: return "the_atlantic"
    if "foreignaffairs" in u: return "foreign_affairs"
    if "alarabiya" in u: return "alarabiya"
    if "apnews" in u: return "ap_news"
    if "politico" in u: return "politico"
    if "axios" in u: return "axios"
    if "bbci" in u or "bbc" in u: return "bbc"
    if "cbsnews" in u: return "cbs"
    if "newsweek" in u: return "newsweek"
    if "theguardian" in u: return "the_guardian"
    if "businessinsider" in u: return "business_insider"
    if "thehill" in u: return "the_hill"
    if "nbcnews" in u: return "nbc"
    if "abcnews" in u: return "abc"
    if "dw.com" in u: return "dw"
    if "france24" in u: return "france24"
    if "lemonde" in u: return "le_monde"
    return "other"

def safe_dt_to_iso(entry: Any) -> str:
    try:
        if getattr(entry, "published_parsed", None):
            ts = time.mktime(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if getattr(entry, "updated_parsed", None):
            ts = time.mktime(entry.updated_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        pass
    return ""

# =========================
# RSS ingestion (async, with concurrency)
# =========================
async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

async def parse_feed(client: httpx.AsyncClient, feed_url: str) -> Dict[str, Any]:
    src_name = normalize_source_name(feed_url)
    t0 = time.time()
    inserted = 0
    fetched_entries = 0
    err = ""

    try:
        xml = await fetch_text(client, feed_url)
        parsed = feedparser.parse(xml)
        entries = parsed.entries or []
        fetched_entries = len(entries)

        for e in entries:
            headline = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            published_at = safe_dt_to_iso(e)

            item = {
                "source": src_name,
                "domain": "",
                "country": "",
                "headline": headline.strip(),
                "content": "",
                "article_summary": summary.strip(),
                "url": link.strip(),
                "published_at": published_at,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "content_fetched": 0,
                "content_error": ""
            }

            if item["url"]:
                if upsert_article(item):
                    inserted += 1

    except Exception as e:
        err = str(e)

    took = round(time.time() - t0, 3)
    return {
        "source": src_name,
        "feed_url": feed_url,
        "fetched_entries": fetched_entries,
        "inserted": inserted,
        "time_sec": took,
        "error": err
    }

async def ingest_once(feed_urls: List[str]) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    started = datetime.now(timezone.utc).isoformat()

    sem = asyncio.Semaphore(RSS_CONCURRENCY)

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        async def _task(u: str):
            async with sem:
                return await parse_feed(client, u)

        results = await asyncio.gather(*[_task(u) for u in feed_urls], return_exceptions=False)

    inserted_total = sum(r["inserted"] for r in results)
    errors_total = sum(1 for r in results if r["error"])

    return {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "inserted_total": inserted_total,
        "errors_total": errors_total,
        "rows_after": row_count(),
        "sources": results
    }

# =========================
# Full-text scraping (safe + robust)
# =========================
def looks_paywalled(html: str) -> bool:
    h = html.lower()
    # simple heuristics
    keywords = ["subscribe", "sign in to continue", "paywall", "enable javascript"]
    return any(k in h for k in keywords)

def extract_fulltext(html: str, url: str) -> str:
    # First: trafilatura
    downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
    if downloaded and len(downloaded.strip()) > 400:
        return downloaded.strip()

    # Fallback: BeautifulSoup (very basic)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    # keep only reasonable length
    if len(text) > 800:
        return text[:20000]
    return ""

async def fetch_and_store_fulltext(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    t0 = time.time()
    try:
        r = await client.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        html = r.text

        if looks_paywalled(html):
            update_fulltext(url, "", "paywalled_or_blocked")
            return {"url": url, "ok": False, "reason": "paywalled_or_blocked", "time_sec": round(time.time() - t0, 3)}

        text = extract_fulltext(html, url)
        if text:
            update_fulltext(url, text, "")
            return {"url": url, "ok": True, "chars": len(text), "time_sec": round(time.time() - t0, 3)}
        else:
            update_fulltext(url, "", "empty_extraction")
            return {"url": url, "ok": False, "reason": "empty_extraction", "time_sec": round(time.time() - t0, 3)}

    except Exception as e:
        update_fulltext(url, "", str(e))
        return {"url": url, "ok": False, "reason": str(e), "time_sec": round(time.time() - t0, 3)}

async def fulltext_worker(limit: int = 30) -> Dict[str, Any]:
    urls = fetch_missing_urls(limit)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,*/*"}
    started = datetime.now(timezone.utc).isoformat()

    sem = asyncio.Semaphore(FULLTEXT_CONCURRENCY)

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        async def _task(u: str):
            async with sem:
                return await fetch_and_store_fulltext(client, u)

        out = await asyncio.gather(*[_task(u) for u in urls], return_exceptions=False)

    ok = sum(1 for x in out if x.get("ok"))
    fail = len(out) - ok
    return {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "requested": len(urls),
        "ok": ok,
        "fail": fail,
        "rows_after": row_count(),
        "details": out
    }

# =========================
# FastAPI app + CORS
# =========================
app = FastAPI(title="MetaView API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # للاختبار
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# =========================
# Models
# =========================
class IngestRequest(BaseModel):
    feeds: Optional[List[str]] = None

class FullTextRequest(BaseModel):
    limit: int = 30

class SearchResponse(BaseModel):
    count: int
    results: List[Dict[str, Any]]

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "metaview",
        "rows": row_count(),
        "db_path": DB_PATH,
        "semantic_enabled": True,
        "time": datetime.now(timezone.utc).isoformat()
    }

@app.post("/ingest/run")
async def ingest_run(body: Optional[IngestRequest] = None):
    feeds = DEFAULT_SOURCES if not body or not body.feeds else body.feeds
    return await ingest_once(feeds)

@app.post("/ingest/fulltext")
async def ingest_fulltext(body: Optional[FullTextRequest] = None):
    limit = 30 if not body else int(body.limit)
    limit = max(1, min(limit, 200))
    return await fulltext_worker(limit)

@app.get("/search-text", response_model=SearchResponse)
def search_text(
    q: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=200),
    source: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
):
    con = get_db()
    cur = con.cursor()

    where = ["(headline LIKE ? OR article_summary LIKE ? OR content LIKE ?)"]
    params = [f"%{q}%", f"%{q}%", f"%{q}%"]

    if source:
        where.append("source = ?")
        params.append(source)

    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)

    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    sql = f"""
    SELECT headline, article_summary, content, published_at, source, url, content_fetched, content_error
    FROM articles
    WHERE {" AND ".join(where)}
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
            "article_summary": r["article_summary"],
            "content": r["content"],
            "published_at": r["published_at"],
            "source": r["source"],
            "url": r["url"],
            "content_fetched": r["content_fetched"],
            "content_error": r["content_error"],
        })

    return {"count": len(results), "results": results}

# =========================
# Optional: auto-ingest loop
# =========================
def _auto_ingest_loop():
    time.sleep(5)
    while True:
        try:
            asyncio.run(ingest_once(DEFAULT_SOURCES))
            asyncio.run(fulltext_worker(limit=30))
        except Exception:
            pass
        time.sleep(max(60, AUTO_INGEST_EVERY_MIN * 60))

if AUTO_INGEST:
    t = threading.Thread(target=_auto_ingest_loop, daemon=True)
    t.start()
