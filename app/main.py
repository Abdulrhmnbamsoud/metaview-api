from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import time
import sqlite3
from datetime import datetime, timezone
import httpx
import feedparser
import threading

# =========================
# Config
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))              # .../app/app
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_DIR, "data"))   # .../app/app/data
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
USER_AGENT = os.getenv("USER_AGENT", "MetaViewBot/1.0 (+https://metaview)")
AUTO_INGEST = os.getenv("AUTO_INGEST", "false").lower() == "true"
AUTO_INGEST_EVERY_MIN = int(os.getenv("AUTO_INGEST_EVERY_MIN", "15"))

# =========================
# Sources (RSS)
# =========================
DEFAULT_SOURCES = [
    # Reuters
    "https://feeds.reuters.com/Reuters/worldNews",

    # Bloomberg (public feed example - Bloomberg is mostly paywalled)
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",

    # Wall Street Journal
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",

    # New York Times
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",

    # Economist
    "https://www.economist.com/international/rss.xml",

    # CNN
    "http://rss.cnn.com/rss/cnn_world.rss",

    # Financial Times (may return 404 sometimes; keep for attempts)
    "https://www.ft.com/world/rss",

    # Washington Post
    "http://feeds.washingtonpost.com/rss/world",

    # The Atlantic
    "https://www.theatlantic.com/feed/all/",

    # Foreign Affairs
    "https://www.foreignaffairs.com/rss.xml",

    # Al Arabiya English (can be 403 sometimes)
    "https://english.alarabiya.net/.mrss/en.xml",

    # Associated Press
    "https://apnews.com/apf-topnews?output=rss",

    # Politico (can be 403 sometimes)
    "https://www.politico.com/rss/politics-news.xml",

    # Axios
    "https://www.axios.com/feeds/feed.rss",

    # BBC
    "http://feeds.bbci.co.uk/news/world/rss.xml",

    # CBS News
    "https://www.cbsnews.com/latest/rss/world",

    # Newsweek
    "https://www.newsweek.com/rss",

    # The Guardian
    "https://www.theguardian.com/world/rss",

    # Business Insider
    "https://www.businessinsider.com/rss",

    # The Hill
    "https://thehill.com/feed/",

    # NBC News
    "http://feeds.nbcnews.com/feeds/worldnews",

    # ABC News
    "http://feeds.abcnews.com/abcnews/internationalheadlines",

    # DW (Germany)
    "https://rss.dw.com/rdf/rss-en-world",

    # France 24
    "https://www.france24.com/en/rss",

    # Le Monde (France)
    "https://www.lemonde.fr/rss/une.xml",
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
        created_at TEXT
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
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
    """
    Insert if new (unique by url). Returns True if inserted, False if skipped.
    """
    con = get_db()
    cur = con.cursor()
    try:
        cur.execute("""
        INSERT OR IGNORE INTO articles
        (source, domain, country, headline, content, article_summary, url, published_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        con.commit()
        inserted = cur.rowcount > 0
        return inserted
    finally:
        con.close()


# =========================
# RSS ingestion
# =========================
def normalize_source_name(feed_url: str) -> str:
    u = feed_url.lower()
    if "reuters" in u:
        return "reuters"
    if "bloomberg" in u:
        return "bloomberg"
    if "dj.com" in u or "wsj" in u:
        return "wsj"
    if "nytimes" in u:
        return "nyt"
    if "economist" in u:
        return "economist"
    if "cnn" in u:
        return "cnn"
    if "ft.com" in u:
        return "financial_times"
    if "washingtonpost" in u:
        return "washington_post"
    if "theatlantic" in u:
        return "the_atlantic"
    if "foreignaffairs" in u:
        return "foreign_affairs"
    if "alarabiya" in u:
        return "alarabiya"
    if "apnews" in u:
        return "ap_news"
    if "politico" in u:
        return "politico"
    if "axios" in u:
        return "axios"
    if "bbci" in u or "bbc" in u:
        return "bbc"
    if "cbsnews" in u:
        return "cbs"
    if "newsweek" in u:
        return "newsweek"
    if "theguardian" in u:
        return "the_guardian"
    if "businessinsider" in u:
        return "business_insider"
    if "thehill" in u:
        return "the_hill"
    if "nbcnews" in u:
        return "nbc"
    if "abcnews" in u:
        return "abc"
    if "dw.com" in u:
        return "dw"
    if "france24" in u:
        return "france24"
    if "lemonde" in u:
        return "le_monde"
    return "other"


def safe_dt_to_iso(entry: Any) -> str:
    # feedparser sometimes gives "published_parsed"
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


async def fetch_feed(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    """
    Return dict with status + parsed feed.
    """
    r = await client.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    parsed = feedparser.parse(r.text)
    return {"url": url, "parsed": parsed}


async def ingest_once(feed_urls: List[str]) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources": [],
        "inserted_total": 0,
        "errors_total": 0,
    }

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        for feed_url in feed_urls:
            src_name = normalize_source_name(feed_url)
            t0 = time.time()
            inserted = 0
            err = ""
            count = 0
            try:
                data = await fetch_feed(client, feed_url)
                parsed = data["parsed"]
                entries = parsed.entries or []
                count = len(entries)

                for e in entries:
                    headline = getattr(e, "title", "") or ""
                    link = getattr(e, "link", "") or ""
                    summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""

                    published_at = safe_dt_to_iso(e)
                    item = {
                        "source": src_name,
                        "domain": "",   # خطوة 2/3 لاحقاً
                        "country": "",  # خطوة 2/3 لاحقاً
                        "headline": headline.strip(),
                        "content": "",  # Full text scraping لاحقاً (عشان ما يطيح السيرفر)
                        "article_summary": summary.strip(),
                        "url": link.strip(),
                        "published_at": published_at,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                    if item["url"]:
                        if upsert_article(item):
                            inserted += 1

            except Exception as e:
                err = str(e)
                result["errors_total"] += 1

            took = round(time.time() - t0, 3)
            result["sources"].append({
                "source": src_name,
                "feed_url": feed_url,
                "fetched_entries": count,
                "inserted": inserted,
                "time_sec": took,
                "error": err
            })
            result["inserted_total"] += inserted

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["rows_after"] = row_count()
    return result


# =========================
# App
# =========================
app = FastAPI(title="MetaView API", version="1.0.0")
init_db()

# =========================
# Models
# =========================
class IngestRequest(BaseModel):
    feeds: Optional[List[str]] = None  # لو حبيت تمرر قائمة غير الافتراضية

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
        "semantic_enabled": False,
        "time": datetime.now(timezone.utc).isoformat()
    }


@app.post("/ingest/run")
async def ingest_run(body: Optional[IngestRequest] = None):
    """
    Run ingestion now (RSS). Stores items into SQLite (dedup by url).
    """
    feeds = DEFAULT_SOURCES if not body or not body.feeds else body.feeds
    out = await ingest_once(feeds)
    return out


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
    """
    Simple text search over headline + summary (stored from RSS).
    """
    con = get_db()
    cur = con.cursor()

    where = []
    params = []

    # Text query
    where.append("(headline LIKE ? OR article_summary LIKE ?)")
    params.extend([f"%{q}%", f"%{q}%"])

    # Filters
    if source:
        where.append("source = ?")
        params.append(source)

    if country:
        where.append("country = ?")
        params.append(country)

    if domain:
        where.append("domain = ?")
        params.append(domain)

    # Date filters (if published_at exists in ISO format)
    # published_at is TEXT, compare lexicographically if ISO
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
            "url": r["url"]
        })

    return {"count": len(results), "results": results}


# =========================
# Optional: auto-ingest loop (lightweight)
# =========================
def _auto_ingest_loop():
    # Wait a bit after boot
    time.sleep(5)
    while True:
        try:
            import asyncio
            asyncio.run(ingest_once(DEFAULT_SOURCES))
        except Exception:
            pass
        # sleep
        time.sleep(max(60, AUTO_INGEST_EVERY_MIN * 60))

if AUTO_INGEST:
    t = threading.Thread(target=_auto_ingest_loop, daemon=True)
    t.start()
