from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os, time, sqlite3, hashlib
import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup

# =============================
# Config
# =============================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(APP_DIR, "data", "metaview.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DEFAULT_TIMEOUT = 20

# مصادر RSS (ابدأ بها اليوم عشان لايف سريع)
RSS_SOURCES = {
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "cnn_world": "http://rss.cnn.com/rss/cnn_world.rss",
    "nyt_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "the_guardian_world": "https://www.theguardian.com/world/rss",
    "dw_world": "https://rss.dw.com/rdf/rss-en-world",
    "france24_en": "https://www.france24.com/en/rss",
    "wsj_world": "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "foreign_affairs": "https://www.foreignaffairs.com/rss.xml",
    "le_monde": "https://www.lemonde.fr/rss/une.xml",
    "economist_international": "https://www.economist.com/international/rss.xml",
    "cbs_world": "https://www.cbsnews.com/latest/rss/world",
    "nbc_world": "http://feeds.nbcnews.com/feeds/worldnews",
    "abc_world": "http://feeds.abcnews.com/abcnews/internationalheadlines",
    # زيد اللي عندك اللي يشتغل RSS (بعضهم بيطلع 403 طبيعي)
}

# =============================
# DB helpers
# =============================
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        source TEXT,
        url TEXT,
        headline TEXT,
        content TEXT,
        summary TEXT,
        published_at TEXT,
        domain TEXT,
        country TEXT,
        lang TEXT,
        created_at TEXT
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_country ON articles(country);")
    conn.commit()
    conn.close()

def make_id(source: str, url: str) -> str:
    return hashlib.sha1(f"{source}|{url}".encode("utf-8")).hexdigest()

def upsert_article(row: Dict[str, Any]) -> bool:
    """
    returns True if inserted, False if already exists
    """
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM articles WHERE id = ?", (row["id"],))
    exists = cur.fetchone() is not None
    if exists:
        conn.close()
        return False

    cur.execute("""
        INSERT INTO articles
        (id, source, url, headline, content, summary, published_at, domain, country, lang, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        row["id"], row["source"], row["url"], row["headline"], row["content"],
        row["summary"], row["published_at"], row["domain"], row["country"], row["lang"]
    ))
    conn.commit()
    conn.close()
    return True

# =============================
# Scraping / parsing helpers
# =============================
def fetch_url(url: str) -> str:
    r = requests.get(url, timeout=DEFAULT_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # إزالة سكربت/ستايل
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = soup.get_text(separator=" ")
    # تنظيف بسيط
    text = " ".join(text.split())
    return text[:20000]  # cap

def rss_to_items(source: str, feed_url: str, limit: int = 50) -> List[Dict[str, Any]]:
    feed = feedparser.parse(feed_url)
    items = []
    for e in feed.entries[:limit]:
        url = getattr(e, "link", "") or ""
        title = getattr(e, "title", "") or ""
        published = ""
        if hasattr(e, "published"):
            published = e.published
        elif hasattr(e, "updated"):
            published = e.updated

        items.append({
            "source": source,
            "url": url,
            "headline": title,
            "published_at": published
        })
    return items

def enrich_item_with_content(item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("url", "")
    content = ""
    if url:
        try:
            html = fetch_url(url)
            content = extract_text_from_html(html)
        except Exception:
            content = ""  # بعض المواقع تمنع، عادي

    # Summary بسيط اليوم (لاحقًا نخليه LLM)
    summary = content[:600] if content else ""

    # domain/country/lang (اليوم نخليها فارغة أو بسيطة)
    item.update({
        "content": content,
        "summary": summary,
        "domain": "",   # لاحقًا تصنيف
        "country": "",  # لاحقًا NER
        "lang": "en"
    })
    return item

# =============================
# API
# =============================
app = FastAPI(title="MetaView Live API", version="1.0.0")
init_db()

class IngestRequest(BaseModel):
    sources: Optional[List[str]] = None
    limit_per_source: int = 30

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    domain: Optional[str] = None
    country: Optional[str] = None
    source: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None

@app.get("/health")
def health():
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles")
    n = cur.fetchone()[0]
    conn.close()
    return {"status": "ok", "service": "metaview", "rows": int(n), "db_path": DB_PATH}

@app.post("/ingest/run")
def run_ingestion(body: IngestRequest):
    """
    Run ingestion now (call it from n8n/cron to make it live)
    """
    start = time.time()
    chosen = body.sources if body.sources else list(RSS_SOURCES.keys())

    report = []
    total_inserted = 0
    for src in chosen:
        feed_url = RSS_SOURCES.get(src)
        if not feed_url:
            report.append({"source": src, "inserted": 0, "error": "unknown source"})
            continue

        try:
            items = rss_to_items(src, feed_url, limit=body.limit_per_source)
            inserted = 0
            for it in items:
                it = enrich_item_with_content(it)
                it["id"] = make_id(it["source"], it["url"])
                if upsert_article(it):
                    inserted += 1
            total_inserted += inserted
            report.append({"source": src, "inserted": inserted, "error": ""})
        except Exception as e:
            report.append({"source": src, "inserted": 0, "error": str(e)})

    return {
        "ok": True,
        "inserted_total": total_inserted,
        "took_sec": round(time.time() - start, 2),
        "details": report
    }

@app.get("/filters")
def filters():
    conn = db()
    cur = conn.cursor()

    def uniq(col: str):
        cur.execute(f"SELECT DISTINCT {col} FROM articles WHERE {col} IS NOT NULL AND TRIM({col}) != '' LIMIT 200")
        return sorted([r[0] for r in cur.fetchall() if r[0]])

    sources = uniq("source")
    domains = uniq("domain")
    countries = uniq("country")

    conn.close()
    return {"sources": sources, "domains": domains, "countries": countries}

@app.get("/search-text")
def search_text(
    q: str,
    domain: Optional[str] = None,
    country: Optional[str] = None,
    source: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    top_k: int = 20
):
    conn = db()
    cur = conn.cursor()

    where = []
    params: List[Any] = []

    # text match (SQLite LIKE)
    where.append("(LOWER(headline) LIKE ? OR LOWER(content) LIKE ?)")
    qn = f"%{q.strip().lower()}%"
    params.extend([qn, qn])

    if domain:
        where.append("LOWER(domain) = ?")
        params.append(domain.strip().lower())
    if country:
        where.append("LOWER(country) = ?")
        params.append(country.strip().lower())
    if source:
        where.append("LOWER(source) = ?")
        params.append(source.strip().lower())

    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)
    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    where_sql = " AND ".join(where)

    cur.execute(f"""
        SELECT source, url, headline, summary, published_at, domain, country, lang
        FROM articles
        WHERE {where_sql}
        ORDER BY published_at DESC
        LIMIT ?
    """, (*params, int(top_k)))

    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "source": r[0],
            "url": r[1],
            "headline": r[2],
            "article_summary": r[3],
            "published_at": r[4],
            "domain": r[5],
            "country": r[6],
            "lang": r[7],
        })

    return {"count": len(results), "results": results}

@app.post("/semantic-search")
def semantic_search(body: SearchRequest):
    # اليوم نخليه نفس text search، وبعدين نربطه FAISS
    return search_text(
        q=body.query,
        domain=body.domain,
        country=body.country,
        source=body.source,
        min_date=body.min_date,
        max_date=body.max_date,
        top_k=body.top_k
    )
