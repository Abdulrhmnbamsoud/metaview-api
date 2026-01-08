import os
import re
import time
import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import httpx
import feedparser
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------
# Optional heavy deps (won't crash if missing)
# -----------------------------
HAS_SEMANTIC = True
try:
    import numpy as np
    import faiss  # faiss-cpu
    from sentence_transformers import SentenceTransformer
except Exception:
    HAS_SEMANTIC = False

# -----------------------------
# Config
# -----------------------------
APP_NAME = "MetaView Live API"
APP_VERSION = "2.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()  # if you want Postgres later

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "20"))
MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "12000"))
DEFAULT_USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
)

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
SEMANTIC_MODEL_NAME = os.getenv("SEMANTIC_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------
# Helpers
# -----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

def detect_lang_simple(text: str) -> str:
    # lightweight heuristic
    t = (text or "").strip()
    if re.search(r"[\u0600-\u06FF]", t):
        return "ar"
    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    if re.search(r"[\u0400-\u04FF]", t):
        return "ru"
    return "en"

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    return norm_text(text)

def safe_parse_datetime(dt_str: str) -> Optional[str]:
    if not dt_str:
        return None
    try:
        # feedparser sometimes returns 'published_parsed'
        # We'll just store as is if not parseable
        return dt_str
    except Exception:
        return None

# -----------------------------
# DB Layer (SQLite now, Postgres later)
# -----------------------------
def get_conn() -> sqlite3.Connection:
    # SQLite is perfect for hackathon/demo; Postgres later via SQLAlchemy if needed
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        url_hash TEXT,
        source TEXT,
        domain TEXT,
        country TEXT,
        language TEXT,
        headline TEXT,
        content TEXT,
        summary TEXT,
        published_at TEXT,
        ingested_at TEXT,
        sentiment_score REAL DEFAULT 0,
        cluster_id INTEGER,
        cluster_summary TEXT
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_ingested ON articles(ingested_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_country ON articles(country);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_lang ON articles(language);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_urlhash ON articles(url_hash);")

    # semantic vectors table (optional)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vectors (
        url_hash TEXT PRIMARY KEY,
        dim INTEGER,
        vector BLOB,
        updated_at TEXT
    );
    """)

    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Rate limiting (very light)
# -----------------------------
_REQUEST_BUCKET: Dict[str, List[float]] = {}

def rate_limit(key: str, limit_per_min: int = RATE_LIMIT_PER_MIN):
    now = time.time()
    bucket = _REQUEST_BUCKET.get(key, [])
    bucket = [t for t in bucket if (now - t) < 60]
    if len(bucket) >= limit_per_min:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)
    _REQUEST_BUCKET[key] = bucket

# -----------------------------
# Semantic Engine (optional)
# -----------------------------
_sem_model = None
_faiss_index = None
_faiss_dim = None
_faiss_map: List[str] = []  # index -> url_hash

def semantic_ready() -> bool:
    return HAS_SEMANTIC

def semantic_init_if_needed():
    global _sem_model, _faiss_index, _faiss_dim, _faiss_map
    if not HAS_SEMANTIC:
        return

    if _sem_model is None:
        _sem_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    if _faiss_index is None:
        _faiss_dim = int(_sem_model.get_sentence_embedding_dimension())
        _faiss_index = faiss.IndexFlatIP(_faiss_dim)
        _faiss_map = []
        # lazy load from DB
        _load_vectors_into_faiss()

def _load_vectors_into_faiss():
    global _faiss_index, _faiss_map, _faiss_dim
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT url_hash, dim, vector FROM vectors")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return

    vecs = []
    keys = []
    for r in rows:
        if int(r["dim"]) != int(_faiss_dim):
            continue
        keys.append(r["url_hash"])
        vecs.append(np.frombuffer(r["vector"], dtype=np.float32))
    if vecs:
        mat = np.vstack(vecs)
        faiss.normalize_L2(mat)
        _faiss_index.add(mat)
        _faiss_map.extend(keys)

def _upsert_vector(url_hash: str, vec: "np.ndarray"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO vectors(url_hash, dim, vector, updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(url_hash) DO UPDATE SET
          dim=excluded.dim,
          vector=excluded.vector,
          updated_at=excluded.updated_at
    """, (url_hash, int(vec.shape[0]), vec.astype("float32").tobytes(), now_utc_iso()))
    conn.commit()
    conn.close()

def _embed_text(text: str) -> Optional["np.ndarray"]:
    if not HAS_SEMANTIC:
        return None
    semantic_init_if_needed()
    v = _sem_model.encode([text], normalize_embeddings=True)
    return v[0].astype("float32")

def _faiss_add(url_hash: str, vec: "np.ndarray"):
    global _faiss_index, _faiss_map
    semantic_init_if_needed()
    _faiss_index.add(vec.reshape(1, -1))
    _faiss_map.append(url_hash)

# -----------------------------
# Ingestion (RSS + optional scraping)
# -----------------------------
async def fetch_url(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, headers={"User-Agent": DEFAULT_USER_AGENT}, follow_redirects=True)
    r.raise_for_status()
    return r.text

def guess_domain_from_url(url: str) -> str:
    m = re.match(r"^https?://([^/]+)/?", url or "")
    return (m.group(1) if m else "").lower()

def tiny_summary(text: str, max_len: int = 240) -> str:
    t = norm_text(text)
    return (t[:max_len] + "…") if len(t) > max_len else t

def sentiment_light(text: str) -> float:
    # Very lightweight sentiment: not perfect, but stable & fast.
    t = (text or "").lower()
    pos = len(re.findall(r"\b(good|great|positive|gain|rise|success|improve)\b", t))
    neg = len(re.findall(r"\b(bad|negative|loss|fall|fail|crisis|attack|risk)\b", t))
    if pos + neg == 0:
        return 0.0
    return float((pos - neg) / (pos + neg))

def db_insert_article(article: Dict[str, Any]) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO articles(
                url, url_hash, source, domain, country, language,
                headline, content, summary, published_at, ingested_at,
                sentiment_score, cluster_id, cluster_summary
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            article["url"],
            article["url_hash"],
            article.get("source",""),
            article.get("domain",""),
            article.get("country",""),
            article.get("language",""),
            article.get("headline",""),
            article.get("content",""),
            article.get("summary",""),
            article.get("published_at",""),
            article.get("ingested_at",""),
            float(article.get("sentiment_score",0.0)),
            article.get("cluster_id"),
            article.get("cluster_summary",""),
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # duplicate url
        return False
    finally:
        conn.close()

def db_count() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM articles")
    c = int(cur.fetchone()["c"])
    conn.close()
    return c

def db_filters() -> Dict[str, List[str]]:
    conn = get_conn()
    cur = conn.cursor()
    out = {}
    for col in ["source","domain","country","language"]:
        cur.execute(f"SELECT DISTINCT {col} AS v FROM articles WHERE {col} IS NOT NULL AND {col} != '' ORDER BY v LIMIT 200")
        out[col] = [r["v"] for r in cur.fetchall()]
    conn.close()
    return out

def db_search_text(q: str, top_k: int, source: str, domain: str, country: str, language: str,
                   min_date: str, max_date: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()

    where = ["(headline LIKE ? OR content LIKE ? OR summary LIKE ?)"]
    params: List[Any] = [f"%{q}%", f"%{q}%", f"%{q}%"]

    if source:
        where.append("source = ?")
        params.append(source)
    if domain:
        where.append("domain = ?")
        params.append(domain)
    if country:
        where.append("country = ?")
        params.append(country)
    if language:
        where.append("language = ?")
        params.append(language)
    if min_date:
        where.append("published_at >= ?")
        params.append(min_date)
    if max_date:
        where.append("published_at <= ?")
        params.append(max_date)

    sql = f"""
        SELECT url, source, domain, country, language, headline, summary, published_at, ingested_at, sentiment_score, cluster_id, cluster_summary
        FROM articles
        WHERE {" AND ".join(where)}
        ORDER BY published_at DESC, ingested_at DESC
        LIMIT ?
    """
    params.append(int(top_k))

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    return [dict(r) for r in rows]

def db_get_by_hash(url_hashes: List[str]) -> List[Dict[str, Any]]:
    if not url_hashes:
        return []
    conn = get_conn()
    cur = conn.cursor()
    placeholders = ",".join(["?"] * len(url_hashes))
    cur.execute(f"""
        SELECT url, source, domain, country, language, headline, summary, published_at, ingested_at, sentiment_score, cluster_id, cluster_summary
        FROM articles
        WHERE url_hash IN ({placeholders})
    """, url_hashes)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# -----------------------------
# API Models
# -----------------------------
class IngestRequest(BaseModel):
    sources: List[str] = Field(..., description="List of RSS URLs OR source keys")
    limit_per_source: int = 30
    scrape_full_text: bool = True

class SearchTextQuery(BaseModel):
    q: str
    top_k: int = 20
    source: Optional[str] = None
    domain: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None

class SemanticRequest(BaseModel):
    query: str
    top_k: int = 20
    source: Optional[str] = None
    domain: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None

# -----------------------------
# Sources Registry (Bonus: keys بدل كتابة RSS كل مرة)
# -----------------------------
SOURCE_REGISTRY = {
    "bbc_world": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "reuters_world": "https://feeds.reuters.com/Reuters/worldNews",
    "nyt_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    # add your own...
}

def resolve_source(s: str) -> str:
    s = s.strip()
    return SOURCE_REGISTRY.get(s, s)  # if it's already a URL, keep it

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title=APP_NAME, version=APP_VERSION)

@app.get("/health")
def health():
    rate_limit("health")
    return {
        "status": "ok",
        "service": "metaview",
        "rows": db_count(),
        "db_path": DB_PATH,
        "semantic_enabled": bool(HAS_SEMANTIC),
        "time": now_utc_iso()
    }

@app.get("/filters")
def filters():
    rate_limit("filters")
    return db_filters()

@app.post("/ingest/run")
async def ingest_run(req: IngestRequest):
    rate_limit("ingest")
    started = time.time()

    sources = [resolve_source(s) for s in req.sources]
    details = []
    inserted_total = 0

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for src in sources:
            try:
                feed = feedparser.parse(src)
                entries = feed.entries[: int(req.limit_per_source)]
                ins = 0
                for e in entries:
                    url = getattr(e, "link", "") or ""
                    if not url:
                        continue

                    headline = norm_text(getattr(e, "title", "") or "")
                    published = norm_text(getattr(e, "published", "") or "") or norm_text(getattr(e, "updated", "") or "")

                    domain = guess_domain_from_url(url)
                    content = ""
                    summary = ""

                    if req.scrape_full_text:
                        try:
                            html = await fetch_url(client, url)
                            content = extract_text_from_html(html)[:MAX_ARTICLE_CHARS]
                        except Exception:
                            content = ""

                    if not content:
                        # fallback: RSS summary
                        rss_sum = getattr(e, "summary", "") or ""
                        content = norm_text(rss_sum)[:MAX_ARTICLE_CHARS]

                    lang = detect_lang_simple(headline + " " + content)
                    summary = tiny_summary(content)

                    article = {
                        "url": url,
                        "url_hash": sha1(url),
                        "source": src,
                        "domain": domain,
                        "country": "",  # optional: fill by your mapping later
                        "language": lang,
                        "headline": headline or "(no title)",
                        "content": content,
                        "summary": summary,
                        "published_at": published or now_utc_iso(),
                        "ingested_at": now_utc_iso(),
                        "sentiment_score": sentiment_light(content),
                        "cluster_id": None,
                        "cluster_summary": "",
                    }

                    ok = db_insert_article(article)
                    if ok:
                        ins += 1
                        inserted_total += 1

                        # OPTIONAL: Build semantic vectors if deps installed
                        if HAS_SEMANTIC:
                            try:
                                vec = _embed_text(article["headline"] + " " + article["summary"])
                                if vec is not None:
                                    _upsert_vector(article["url_hash"], vec)
                                    # keep faiss fresh (best effort)
                                    try:
                                        semantic_init_if_needed()
                                        _faiss_add(article["url_hash"], vec)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                details.append({"source": src, "inserted": ins, "error": None})
            except Exception as ex:
                details.append({"source": src, "inserted": 0, "error": str(ex)})

    return {
        "ok": True,
        "inserted_total": inserted_total,
        "took_sec": round(time.time() - started, 3),
        "details": details
    }

@app.get("/search-text")
def search_text(
    q: str,
    top_k: int = 20,
    source: Optional[str] = None,
    domain: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
):
    rate_limit("search-text")

    q = norm_text(q)
    if not q:
        raise HTTPException(status_code=400, detail="q is required")

    res = db_search_text(
        q=q,
        top_k=int(top_k),
        source=(source or "").strip(),
        domain=(domain or "").strip(),
        country=(country or "").strip(),
        language=(language or "").strip(),
        min_date=(min_date or "").strip(),
        max_date=(max_date or "").strip(),
    )

    return {"count": len(res), "results": res}

@app.post("/semantic-search")
def semantic_search(req: SemanticRequest):
    rate_limit("semantic-search")

    if not HAS_SEMANTIC:
        return {
            "ok": False,
            "message": "Semantic search not enabled. Install extras: sentence-transformers + faiss-cpu",
            "count": 0,
            "results": []
        }

    semantic_init_if_needed()
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return {
            "ok": True,
            "message": "No vectors yet. Run /ingest/run first.",
            "count": 0,
            "results": []
        }

    q = norm_text(req.query)
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    vec = _embed_text(q)
    if vec is None:
        return {"ok": False, "message": "Embedding failed", "count": 0, "results": []}

    # search
    x = vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(x)
    scores, idxs = _faiss_index.search(x, int(req.top_k))

    picked_hashes = []
    for i in idxs[0]:
        if i < 0:
            continue
        picked_hashes.append(_faiss_map[i])

    rows = db_get_by_hash(picked_hashes)

    # apply filters in python (fast enough for top_k)
    def ok_row(r):
        if req.source and r.get("source") != req.source:
            return False
        if req.domain and r.get("domain") != req.domain:
            return False
        if req.country and r.get("country") != req.country:
            return False
        if req.language and r.get("language") != req.language:
            return False
        return True

    rows = [r for r in rows if ok_row(r)]
    return {"ok": True, "count": len(rows), "results": rows}

@app.get("/")
def root():
    return {"service": "metaview", "docs": "/docs", "openapi": "/openapi.json"}
