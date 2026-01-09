from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
import os
import time
import sqlite3
from datetime import datetime, timezone
import httpx
import feedparser
from dateutil import parser as dtparser
from urllib.parse import urlparse

# =========================
# Config
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
USER_AGENT = os.getenv("USER_AGENT", "MetaViewBot/1.0 (+https://metaview)")
# لتقليل الضغط: حد أعلى لكل مصدر
MAX_ITEMS_PER_FEED = int(os.getenv("MAX_ITEMS_PER_FEED", "50"))

# CORS (خله * مؤقتاً، بعدين قفلها على دوميناتك)
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS_LIST = ["*"] if ALLOW_ORIGINS.strip() == "*" else [x.strip() for x in ALLOW_ORIGINS.split(",")]

# =========================
# Sources (RSS) - حط كل المصادر اللي تبي
# =========================
DEFAULT_SOURCES = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "http://rss.cnn.com/rss/cnn_world.rss",
    "https://www.theguardian.com/world/rss",
    "https://rss.dw.com/rdf/rss-en-world",
    "https://www.france24.com/en/rss",
    "http://feeds.washingtonpost.com/rss/world",
    "https://www.axios.com/feeds/feed.rss",
    "https://thehill.com/feed/",
    "https://www.newsweek.com/rss",
    "https://www.businessinsider.com/rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.foreignaffairs.com/rss.xml",
    "https://www.economist.com/international/rss.xml",
    # Reuters sometimes picky — اتركه خيار
    "https://feeds.reuters.com/Reuters/worldNews",
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
        sentiment_label TEXT,
        sentiment_score REAL
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_country ON articles(country)")
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
        (source, domain, country, headline, content, article_summary, url, published_at, created_at, sentiment_label, sentiment_score)
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
            item.get("sentiment_label", None),
            item.get("sentiment_score", None),
        ))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()

def update_sentiment_by_id(article_id: int, label: str, score: float) -> None:
    con = get_db()
    cur = con.cursor()
    cur.execute(
        "UPDATE articles SET sentiment_label=?, sentiment_score=? WHERE id=?",
        (label, float(score), int(article_id))
    )
    con.commit()
    con.close()

# =========================
# Helpers
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_published(entry: Any) -> str:
    # Feedparser gives several possible fields
    for key in ["published", "updated", "created"]:
        val = getattr(entry, key, None)
        if val:
            try:
                dt = dtparser.parse(val)
                if not dt.tzinfo:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                pass

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

def domain_from_url(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d.replace("www.", "")
    except Exception:
        return ""

def normalize_source_name(feed_url: str) -> str:
    u = feed_url.lower()
    if "bbc" in u: return "bbc"
    if "nytimes" in u: return "nyt"
    if "cnn" in u: return "cnn"
    if "theguardian" in u: return "the_guardian"
    if "dw.com" in u: return "dw"
    if "france24" in u: return "france24"
    if "washingtonpost" in u: return "washington_post"
    if "axios" in u: return "axios"
    if "thehill" in u: return "the_hill"
    if "newsweek" in u: return "newsweek"
    if "businessinsider" in u: return "business_insider"
    if "dj.com" in u or "wsj" in u: return "wsj"
    if "foreignaffairs" in u: return "foreign_affairs"
    if "economist" in u: return "economist"
    if "reuters" in u: return "reuters"
    return "other"

# =========================
# Sentiment (خفيف وثابت)
# - هذا intentional: لا نستخدم Transformers هنا عشان ما يطيّح السيرفر / يكبر الدوكر
# - Studio AI بيكمل ترجمة/مقارنة/LLM
# =========================
AR_POS = {
    "ممتاز", "جيد", "جميل", "رائع", "قوي", "نجاح", "إيجابي", "تحسن", "ربح", "ارتفاع", "تفاؤل", "إنجاز"
}
AR_NEG = {
    "سيء", "ضعيف", "فشل", "سلبي", "هبوط", "خسارة", "تراجع", "حزين", "قلق", "أزمة", "مشكلة", "انهيار"
}
EN_POS = {"good","great","excellent","positive","success","improve","growth","profit","win","strong","optimistic"}
EN_NEG = {"bad","poor","negative","fail","loss","decline","crisis","problem","collapse","weak","fear"}

def sentiment_heuristic(text: str) -> Tuple[str, float]:
    t = (text or "").lower()
    if not t.strip():
        return ("neutral", 0.0)

    score = 0
    # Arabic keywords (case-sensitive partly, so check raw too)
    raw = text or ""
    for w in AR_POS:
        if w in raw:
            score += 1
    for w in AR_NEG:
        if w in raw:
            score -= 1

    # English keywords
    for w in EN_POS:
        if w in t:
            score += 1
    for w in EN_NEG:
        if w in t:
            score -= 1

    if score >= 2:
        return ("positive", min(1.0, score / 5.0))
    if score <= -2:
        return ("negative", max(-1.0, score / 5.0))
    return ("neutral", score / 5.0)

# =========================
# RSS ingestion (Light + Stable)
# =========================
async def ingest_once(feed_urls: List[str]) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    result = {
        "ok": True,
        "started_at": now_iso(),
        "inserted_total": 0,
        "took_sec": 0.0,
        "details": []
    }

    t_start = time.time()
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=REQUEST_TIMEOUT) as client:
        for feed_url in feed_urls:
            src_name = normalize_source_name(feed_url)
            inserted = 0
            err = None
            t0 = time.time()

            try:
                r = await client.get(feed_url)
                r.raise_for_status()

                parsed = feedparser.parse(r.text)
                entries = parsed.entries or []
                # Limit
                entries = entries[:MAX_ITEMS_PER_FEED]

                for e in entries:
                    headline = (getattr(e, "title", "") or "").strip()
                    link = (getattr(e, "link", "") or "").strip()
                    summary = (getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
                    published_at = parse_published(e)

                    if not link:
                        continue

                    item = {
                        "source": src_name,
                        "domain": domain_from_url(link),
                        "country": "",  # تقدر تعبيه لاحقاً حسب الدومين/المصدر
                        "headline": headline,
                        "content": "",  # Full scraping (اختياري لاحقاً) — نخليه فاضي عشان الاستقرار
                        "article_summary": summary,
                        "url": link,
                        "published_at": published_at,
                        "created_at": now_iso(),
                    }

                    # Sentiment (خفيف) من (headline+summary)
                    s_label, s_score = sentiment_heuristic(f"{headline} {summary}")
                    item["sentiment_label"] = s_label
                    item["sentiment_score"] = float(s_score)

                    if upsert_article(item):
                        inserted += 1

            except Exception as e:
                err = str(e)

            result["details"].append({
                "source": src_name,
                "feed_url": feed_url,
                "inserted": inserted,
                "error": err,
                "took_sec": round(time.time() - t0, 3)
            })
            result["inserted_total"] += inserted

    result["took_sec"] = round(time.time() - t_start, 3)
    result["finished_at"] = now_iso()
    result["rows_after"] = row_count()
    return result

# =========================
# App
# =========================
app = FastAPI(title="MetaView Live API", version="2.0.0")
init_db()

# CORS (حل مشكلة "فشل الاتصال" في الواجهات)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS_LIST,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================
class IngestRequest(BaseModel):
    sources: List[str]

class SearchResponse(BaseModel):
    count: int
    results: List[Dict[str, Any]]

class SentimentRequest(BaseModel):
    text: Optional[str] = None
    article_id: Optional[int] = None

# =========================
# Endpoints - Core
# =========================
@app.get("/")
def root():
    return {"service": "metaview", "status": "ok", "time": now_iso()}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "metaview",
        "rows": row_count(),
        "db_path": DB_PATH,
        "semantic_enabled": True,
        "time": now_iso()
    }

@app.get("/filters")
def filters():
    # للواجهة: تعبي dropdowns
    con = get_db()
    cur = con.cursor()

    cur.execute("SELECT DISTINCT source FROM articles WHERE source != '' ORDER BY source")
    sources = [r["source"] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT domain FROM articles WHERE domain != '' ORDER BY domain")
    domains = [r["domain"] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT country FROM articles WHERE country != '' ORDER BY country")
    countries = [r["country"] for r in cur.fetchall()]

    con.close()
    return {"sources": sources, "domains": domains, "countries": countries}

@app.get("/dashboard/metrics")
def dashboard_metrics():
    con = get_db()
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) c FROM articles")
    total = int(cur.fetchone()["c"])

    cur.execute("SELECT source, COUNT(*) c FROM articles GROUP BY source ORDER BY c DESC LIMIT 10")
    by_source = [{"source": r["source"], "count": int(r["c"])} for r in cur.fetchall()]

    cur.execute("""
        SELECT sentiment_label, COUNT(*) c
        FROM articles
        WHERE sentiment_label IS NOT NULL
        GROUP BY sentiment_label
        ORDER BY c DESC
    """)
    by_sentiment = [{"label": r["sentiment_label"], "count": int(r["c"])} for r in cur.fetchall()]

    con.close()
    return {
        "total_articles": total,
        "top_sources": by_source,
        "sentiment_distribution": by_sentiment,
        "time": now_iso()
    }

# =========================
# Ingest (Background Task) - عشان ما يعلق
# =========================
_last_ingest: Dict[str, Any] = {"status": "idle", "last_result": None}

def _run_ingest_task(sources: List[str]):
    global _last_ingest
    _last_ingest = {"status": "running", "last_result": None}
    try:
        import asyncio
        out = asyncio.run(ingest_once(sources))
        _last_ingest = {"status": "done", "last_result": out}
    except Exception as e:
        _last_ingest = {"status": "error", "last_result": {"error": str(e), "time": now_iso()}}

@app.post("/ingest/run")
def ingest_run(body: IngestRequest, background: BackgroundTasks):
    """
    يشغّل ingest في الخلفية (ثابت وما يطيح)
    لازم ترسل sources (قائمة RSS)
    """
    sources = body.sources or []
    background.add_task(_run_ingest_task, sources)
    return {"ok": True, "queued": True, "sources_count": len(sources), "time": now_iso()}

@app.get("/ingest/status")
def ingest_status():
    return _last_ingest

# =========================
# Search
# =========================
@app.get("/search-text", response_model=SearchResponse)
def search_text(
    q: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=200),
    source: Optional[str] = None,
    country: Optional[str] = None,
    domain: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    sentiment: Optional[str] = None,  # positive/neutral/negative
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

    if sentiment:
        where.append("sentiment_label = ?")
        params.append(sentiment)

    # ISO compare lexicographically
    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)

    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    where_sql = " AND ".join(where) if where else "1=1"
    sql = f"""
    SELECT id, headline, content, article_summary, published_at, source, domain, country, url, sentiment_label, sentiment_score
    FROM articles
    WHERE {where_sql}
    ORDER BY COALESCE(published_at, created_at) DESC
    LIMIT ?
    """
    params.append(int(top_k))

    cur.execute(sql, params)
    rows = cur.fetchall()
    con.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "headline": r["headline"],
            "content": r["content"],
            "article_summary": r["article_summary"],
            "published_at": r["published_at"],
            "source": r["source"],
            "domain": r["domain"],
            "country": r["country"],
            "url": r["url"],
            "sentiment_label": r["sentiment_label"],
            "sentiment_score": r["sentiment_score"],
        })

    return {"count": len(results), "results": results}

# =========================
# Sentiment Endpoint (إلى هنا ونوقف)
# =========================
@app.post("/analyze/sentiment")
def analyze_sentiment(body: SentimentRequest):
    """
    تحلل مشاعر نص أو Article من DB
    (الترجمة + المقارنة = في Studio AI لاحقاً)
    """
    if body.article_id:
        con = get_db()
        cur = con.cursor()
        cur.execute("SELECT id, headline, article_summary FROM articles WHERE id=?", (int(body.article_id),))
        row = cur.fetchone()
        con.close()
        if not row:
            return {"ok": False, "error": "article_id not found"}

        text = f"{row['headline']} {row['article_summary']}"
        label, score = sentiment_heuristic(text)
        update_sentiment_by_id(int(row["id"]), label, score)
        return {"ok": True, "mode": "heuristic", "label": label, "score": score, "article_id": int(row["id"])}

    text = body.text or ""
    label, score = sentiment_heuristic(text)
    return {"ok": True, "mode": "heuristic", "label": label, "score": score}
