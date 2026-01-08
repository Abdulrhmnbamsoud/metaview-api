from __future__ import annotations

import os
import re
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import httpx
import feedparser
import trafilatura

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Float, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError

from sentence_transformers import SentenceTransformer
import faiss


# =========================
# Config / ENV
# =========================
APP_NAME = "MetaView Live API"
APP_VERSION = "2.0.0"

DATABASE_URL = os.getenv("DATABASE_URL")  # Railway Postgres recommended
DB_PATH = os.getenv("DB_PATH", "/app/app/data/metaview.db")  # SQLite fallback

# If no DATABASE_URL -> use SQLite
if not DATABASE_URL:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    DATABASE_URL = f"sqlite:///{DB_PATH}"

FAISS_PATH = os.getenv("FAISS_PATH", "/app/app/data/faiss.index")
FAISS_META_PATH = os.getenv("FAISS_META_PATH", "/app/app/data/faiss_meta.json")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "MetaViewBot/2.0 (+contact: you@example.com) FastAPI"
)

# RSS-first, then fetch full text from article URL (if accessible).
DEFAULT_SOURCES = {
    # (ممكن تزيدها بعدين عبر API /sources)
    "bbc_world": {
        "type": "rss",
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml"
    },
    "cnn_world": {
        "type": "rss",
        "url": "http://rss.cnn.com/rss/edition_world.rss"
    },
    "nyt_world": {
        "type": "rss",
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
    },
    "dw_world": {
        "type": "rss",
        "url": "https://rss.dw.com/rdf/rss-en-world"
    },
    "france24_en": {
        "type": "rss",
        "url": "https://www.france24.com/en/rss"
    },
    "the_guardian_world": {
        "type": "rss",
        "url": "https://www.theguardian.com/world/rss"
    }
}


# =========================
# DB
# =========================
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Source(Base):
    __tablename__ = "sources"

    id = Column(String(100), primary_key=True)        # bbc_world
    type = Column(String(20), nullable=False)         # rss
    url = Column(Text, nullable=False)                # rss url
    enabled = Column(Integer, default=1)              # 1/0
    meta_json = Column(Text, default="{}")            # {"domain":"Security",...}


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uid = Column(String(64), nullable=False)          # hash(url or title+date)
    source = Column(String(100), nullable=False)
    domain = Column(String(100), default="")
    country = Column(String(100), default="")

    headline = Column(Text, default="")
    url = Column(Text, default="")
    published_at = Column(DateTime(timezone=True), nullable=True)

    content = Column(Text, default="")
    summary = Column(Text, default="")

    sentiment_score = Column(Float, nullable=True)    # optional (bonus later)
    cluster_id = Column(Integer, nullable=True)       # optional (bonus later)
    cluster_summary = Column(Text, default="")        # optional (bonus later)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("uid", name="uq_articles_uid"),
    )


Base.metadata.create_all(bind=engine)


# =========================
# FAISS (Semantic Search)
# =========================
_embedder: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.IndexFlatIP] = None
_faiss_meta: Dict[str, Any] = {"ids": []}  # maps faiss row -> article_id


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def normalize(v: np.ndarray) -> np.ndarray:
    # cosine similarity via inner product on normalized vectors
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norm


def load_faiss():
    global _faiss_index, _faiss_meta
    if os.path.exists(FAISS_PATH) and os.path.exists(FAISS_META_PATH):
        _faiss_index = faiss.read_index(FAISS_PATH)
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            _faiss_meta = json.load(f)
    else:
        _faiss_index = None
        _faiss_meta = {"ids": []}


def save_faiss():
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        return
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    faiss.write_index(_faiss_index, FAISS_PATH)
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(_faiss_meta, f)


def ensure_faiss(dim: int):
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(dim)
        _faiss_meta = {"ids": []}


def build_or_update_index(article_rows: List[Dict[str, Any]]):
    """
    Add new articles to FAISS.
    article_rows: [{"id":..., "text":...}, ...]
    """
    if not article_rows:
        return 0

    embedder = get_embedder()
    texts = [r["text"] for r in article_rows]
    embs = embedder.encode(texts, batch_size=32, show_progress_bar=False)
    embs = np.array(embs).astype("float32")
    embs = normalize(embs)

    ensure_faiss(embs.shape[1])

    _faiss_index.add(embs)
    _faiss_meta["ids"].extend([int(r["id"]) for r in article_rows])
    save_faiss()
    return len(article_rows)


# Load FAISS at startup
load_faiss()


# =========================
# Helpers
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def make_uid(url: str, title: str, published: Optional[datetime]) -> str:
    base = (url or "") + "||" + (title or "") + "||" + (published.isoformat() if published else "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


async def fetch_url(url: str) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text


def extract_full_text(html: str, url: str) -> str:
    """
    Safe extraction using trafilatura.
    If paywalled/blocked => returns "".
    """
    try:
        downloaded = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        return clean_text(downloaded or "")
    except Exception:
        return ""


def parse_published(entry: Any) -> Optional[datetime]:
    # feedparser may provide published_parsed
    try:
        if getattr(entry, "published_parsed", None):
            dt = datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
            return dt
    except Exception:
        pass
    return None


def to_article_text(row: Dict[str, Any]) -> str:
    # what we embed for semantic search
    return clean_text(
        (row.get("headline") or "") + "\n" +
        (row.get("summary") or "") + "\n" +
        (row.get("content") or "")
    )


# =========================
# API Schemas
# =========================
class SourceUpsert(BaseModel):
    id: str = Field(..., description="Unique source id, e.g. bbc_world")
    type: str = Field("rss", description="Only rss in this version")
    url: str
    enabled: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)  # domain/country defaults etc.


class IngestRequest(BaseModel):
    sources: Optional[List[str]] = Field(None, description="If null -> ingest all enabled sources")
    limit_per_source: int = Field(30, ge=1, le=500)
    since_minutes: int = Field(1440, ge=5, le=60*24*30, description="Only keep recent items window (minutes)")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(20, ge=1, le=200)
    domain: Optional[str] = None
    country: Optional[str] = None
    source: Optional[str] = None
    min_date: Optional[str] = None  # ISO date string
    max_date: Optional[str] = None  # ISO date string


# =========================
# FastAPI
# =========================
app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/health")
def health():
    with SessionLocal() as db:
        total = db.query(Article).count()
        sources = db.query(Source).count()
    return {
        "status": "ok",
        "service": "metaview",
        "version": APP_VERSION,
        "db": "postgres" if "postgres" in DATABASE_URL else "sqlite",
        "db_path": DB_PATH if "sqlite" in DATABASE_URL else None,
        "rows": int(total),
        "sources": int(sources),
        "faiss_loaded": _faiss_index is not None,
        "faiss_size": int(_faiss_index.ntotal) if _faiss_index is not None else 0,
    }


# -------------------------
# Sources (Dynamic)
# -------------------------
@app.get("/sources")
def list_sources():
    with SessionLocal() as db:
        rows = db.query(Source).all()
        return [{
            "id": r.id,
            "type": r.type,
            "url": r.url,
            "enabled": bool(r.enabled),
            "meta": json.loads(r.meta_json or "{}")
        } for r in rows]


@app.post("/sources")
def upsert_source(body: SourceUpsert):
    with SessionLocal() as db:
        existing = db.query(Source).filter(Source.id == body.id).first()
        if existing:
            existing.type = body.type
            existing.url = body.url
            existing.enabled = 1 if body.enabled else 0
            existing.meta_json = json.dumps(body.meta or {})
        else:
            db.add(Source(
                id=body.id,
                type=body.type,
                url=body.url,
                enabled=1 if body.enabled else 0,
                meta_json=json.dumps(body.meta or {})
            ))
        db.commit()
    return {"ok": True, "id": body.id}


@app.delete("/sources/{source_id}")
def delete_source(source_id: str):
    with SessionLocal() as db:
        row = db.query(Source).filter(Source.id == source_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Source not found")
        db.delete(row)
        db.commit()
    return {"ok": True, "deleted": source_id}


@app.on_event("startup")
def seed_sources():
    """
    Seed DEFAULT_SOURCES once (if not exist).
    """
    with SessionLocal() as db:
        for sid, cfg in DEFAULT_SOURCES.items():
            exist = db.query(Source).filter(Source.id == sid).first()
            if not exist:
                db.add(Source(
                    id=sid,
                    type=cfg["type"],
                    url=cfg["url"],
                    enabled=1,
                    meta_json=json.dumps({})
                ))
        db.commit()


# -------------------------
# Ingestion (Live)
# -------------------------
@app.post("/ingest/run")
async def ingest_run(body: IngestRequest):
    started = time.time()
    since_dt = now_utc() - timedelta(minutes=int(body.since_minutes))

    with SessionLocal() as db:
        q = db.query(Source).filter(Source.enabled == 1)
        if body.sources:
            q = q.filter(Source.id.in_(body.sources))
        sources = q.all()

    if not sources:
        return {"ok": True, "inserted_total": 0, "took_sec": round(time.time() - started, 3), "details": []}

    details = []
    inserted_total = 0
    to_index = []  # for FAISS

    for s in sources:
        sid = s.id
        meta = json.loads(s.meta_json or "{}")

        # RSS ingest
        try:
            feed_xml = await fetch_url(s.url)
            feed = feedparser.parse(feed_xml)
            entries = feed.entries[: int(body.limit_per_source)]
        except Exception as e:
            details.append({"source": sid, "inserted": 0, "error": f"rss_fetch_failed: {str(e)}"})
            continue

        inserted = 0
        for entry in entries:
            title = clean_text(getattr(entry, "title", "") or "")
            link = clean_text(getattr(entry, "link", "") or "")
            published = parse_published(entry)

            if published and published < since_dt:
                continue

            uid = make_uid(link, title, published)

            # try fetch full text from article URL
            content = ""
            if link:
                try:
                    html = await fetch_url(link)
                    content = extract_full_text(html, link)
                except Exception:
                    content = ""  # blocked / paywall / forbidden

            summary = clean_text(getattr(entry, "summary", "") or "")

            # fallback: if content empty use summary
            final_content = content if len(content) >= 200 else summary

            domain = str(meta.get("domain", "") or "")
            country = str(meta.get("country", "") or "")

            row = {
                "uid": uid,
                "source": sid,
                "domain": domain,
                "country": country,
                "headline": title,
                "url": link,
                "published_at": published,
                "content": final_content,
                "summary": ""  # keep empty for now (bonus later)
            }

            with SessionLocal() as db:
                try:
                    art = Article(
                        uid=row["uid"],
                        source=row["source"],
                        domain=row["domain"],
                        country=row["country"],
                        headline=row["headline"],
                        url=row["url"],
                        published_at=row["published_at"],
                        content=row["content"],
                        summary=row["summary"],
                    )
                    db.add(art)
                    db.commit()
                    db.refresh(art)

                    inserted += 1
                    inserted_total += 1

                    # prepare for semantic indexing
                    to_index.append({"id": art.id, "text": to_article_text({
                        "headline": art.headline,
                        "content": art.content,
                        "summary": art.summary
                    })})

                except IntegrityError:
                    db.rollback()  # duplicate uid
                except Exception as e:
                    db.rollback()
                    # skip this entry
                    continue

        details.append({"source": sid, "inserted": inserted, "error": ""})

    # update FAISS for newly inserted items
    indexed = 0
    try:
        indexed = build_or_update_index(to_index)
    except Exception:
        indexed = 0

    return {
        "ok": True,
        "inserted_total": inserted_total,
        "indexed_new": indexed,
        "took_sec": round(time.time() - started, 3),
        "details": details,
    }


# -------------------------
# Filters
# -------------------------
@app.get("/filters")
def filters():
    with SessionLocal() as db:
        sources = [r[0] for r in db.query(Article.source).distinct().all()]
        domains = [r[0] for r in db.query(Article.domain).distinct().all() if r[0]]
        countries = [r[0] for r in db.query(Article.country).distinct().all() if r[0]]
    return {
        "sources": sorted([s for s in sources if s]),
        "domains": sorted(domains),
        "countries": sorted(countries)
    }


# -------------------------
# Text Search (Advanced)
# -------------------------
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
    with SessionLocal() as db:
        query = db.query(Article)

        if domain:
            query = query.filter(Article.domain.ilike(domain))
        if country:
            query = query.filter(Article.country.ilike(country))
        if source:
            query = query.filter(Article.source.ilike(source))

        def parse_dt(s: str) -> Optional[datetime]:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return None

        if min_date:
            md = parse_dt(min_date)
            if md:
                query = query.filter(Article.published_at >= md)
        if max_date:
            xd = parse_dt(max_date)
            if xd:
                query = query.filter(Article.published_at <= xd)

        qlow = (q or "").strip().lower()
        if not qlow:
            raise HTTPException(status_code=400, detail="q is required")

        # Simple contains (headline/content)
        # NOTE: For full-text search later you can move to Postgres tsvector.
        rows = query.order_by(Article.published_at.desc().nullslast()).all()

        hits = []
        for r in rows:
            hay = f"{r.headline} {r.content}".lower()
            if qlow in hay:
                hits.append(r)
                if len(hits) >= int(top_k):
                    break

        return {
            "count": len(hits),
            "results": [
                {
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                    "source": r.source,
                    "domain": r.domain,
                    "country": r.country,
                    "headline": r.headline,
                    "article_summary": r.summary,
                    "sentiment_score": r.sentiment_score,
                    "cluster_id": r.cluster_id,
                    "cluster_summary": r.cluster_summary,
                    "url": r.url,
                } for r in hits
            ]
        }


# -------------------------
# Semantic Search (FAISS)
# -------------------------
@app.post("/semantic-search")
def semantic_search(body: SearchRequest):
    if _faiss_index is None or not _faiss_meta.get("ids"):
        return {"count": 0, "results": [], "message": "semantic index not built yet, run /ingest/run first"}

    # parse date filters
    def parse_dt(s: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    md = parse_dt(body.min_date) if body.min_date else None
    xd = parse_dt(body.max_date) if body.max_date else None

    embedder = get_embedder()
    q_emb = embedder.encode([body.query], show_progress_bar=False)
    q_emb = np.array(q_emb).astype("float32")
    q_emb = normalize(q_emb)

    k = min(int(body.top_k) * 5, 1000)  # pull more then filter down
    D, I = _faiss_index.search(q_emb, k)

    # map faiss rows -> article_id
    candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        try:
            article_id = int(_faiss_meta["ids"][idx])
        except Exception:
            continue
        candidates.append((article_id, float(score)))

    # fetch from DB and apply filters
    results = []
    with SessionLocal() as db:
        for aid, score in candidates:
            r = db.query(Article).filter(Article.id == aid).first()
            if not r:
                continue

            if body.source and r.source.lower() != body.source.lower():
                continue
            if body.domain and (r.domain or "").lower() != body.domain.lower():
                continue
            if body.country and (r.country or "").lower() != body.country.lower():
                continue
            if md and r.published_at and r.published_at < md:
                continue
            if xd and r.published_at and r.published_at > xd:
                continue

            results.append({
                "semantic_score": score,
                "published_at": r.published_at.isoformat() if r.published_at else None,
                "source": r.source,
                "domain": r.domain,
                "country": r.country,
                "headline": r.headline,
                "article_summary": r.summary,
                "sentiment_score": r.sentiment_score,
                "cluster_id": r.cluster_id,
                "cluster_summary": r.cluster_summary,
                "url": r.url,
            })
            if len(results) >= int(body.top_k):
                break

    return {"count": len(results), "results": results}


# -------------------------
# Utility: Force rebuild FAISS from DB
# -------------------------
@app.post("/index/rebuild")
def rebuild_index():
    started = time.time()
    global _faiss_index, _faiss_meta
    _faiss_index = None
    _faiss_meta = {"ids": []}

    rows_to_add = []
    with SessionLocal() as db:
        rows = db.query(Article).all()
        for r in rows:
            rows_to_add.append({"id": r.id, "text": to_article_text({
                "headline": r.headline,
                "content": r.content,
                "summary": r.summary
            })})

    added = build_or_update_index(rows_to_add)
    return {"ok": True, "added": added, "took_sec": round(time.time() - started, 3)}
