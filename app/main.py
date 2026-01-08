from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import pandas as pd

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="MetaView API", version="1.0.0")

# -----------------------------
# Resolve data path (PROD first)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../app
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))      # repo root

# Prefer production path inside app/data/
DEFAULT_PATHS = [
    os.path.join(BASE_DIR, "data", "news_latest.csv"),                 # app/data/news_latest.csv  ✅ (your current)
    os.path.join(REPO_ROOT, "data", "news_latest.csv"),                # data/news_latest.csv
    os.path.join(REPO_ROOT, "notebooks", "data", "news_latest.csv"),   # notebooks/data/news_latest.csv
    os.path.join(REPO_ROOT, "notebooks", "news_with_sentiment.csv"),   # notebooks/news_with_sentiment.csv
]

ENV_PATH = os.getenv("DATA_PATH")


def pick_data_path() -> str:
    if ENV_PATH and os.path.exists(ENV_PATH):
        return ENV_PATH
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    return DEFAULT_PATHS[0]  # will fail gracefully


DATA_PATH = pick_data_path()

# -----------------------------
# Load data
# -----------------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure expected columns exist
    for col, default in [
        ("headline", ""),
        ("content", ""),
        ("article_summary", ""),
        ("domain", ""),
        ("country", ""),
        ("source", ""),
        ("url", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    # Clean text columns
    df["headline"] = df["headline"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["article_summary"] = df["article_summary"].fillna("").astype(str)
    df["domain"] = df["domain"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)

    # Parse dates if present
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    return df


def safe_lower(x: str) -> str:
    return str(x).strip().lower()


def apply_filters(
    d: pd.DataFrame,
    domain: Optional[str] = None,
    country: Optional[str] = None,
    source: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
) -> pd.DataFrame:
    # Exact-match filters (case-insensitive)
    if domain:
        d = d[d["domain"].map(safe_lower) == safe_lower(domain)]
    if country:
        d = d[d["country"].map(safe_lower) == safe_lower(country)]
    if source:
        d = d[d["source"].map(safe_lower) == safe_lower(source)]

    # Date range filter
    if "published_at" in d.columns:
        if min_date:
            md = pd.to_datetime(min_date, errors="coerce", utc=True)
            if pd.notna(md):
                d = d[d["published_at"] >= md]
        if max_date:
            xd = pd.to_datetime(max_date, errors="coerce", utc=True)
            if pd.notna(xd):
                d = d[d["published_at"] <= xd]

    return d


def select_output_cols(d: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "published_at", "source", "domain", "country",
        "headline", "article_summary", "sentiment_score",
        "cluster_id", "cluster_summary", "url"
    ]
    cols = [c for c in wanted if c in d.columns]
    return d[cols]


try:
    df = load_df(DATA_PATH)
except Exception as e:
    df = pd.DataFrame()
    print(f"[WARN] Failed to load data from {DATA_PATH}: {e}")

# -----------------------------
# Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    domain: Optional[str] = None
    country: Optional[str] = None
    source: Optional[str] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "metaview",
        "rows": int(df.shape[0]) if not df.empty else 0,
        "data_path": DATA_PATH,
    }


@app.post("/reload")
def reload_dataset():
    global df, DATA_PATH
    DATA_PATH = pick_data_path()
    try:
        df = load_df(DATA_PATH)
        return {"status": "ok", "rows": int(df.shape[0]), "data_path": DATA_PATH}
    except Exception as e:
        df = pd.DataFrame()
        return {"status": "error", "message": str(e), "data_path": DATA_PATH}


@app.get("/sources")
def list_sources():
    if df.empty:
        return {"items": []}
    items = sorted([x for x in df["source"].dropna().astype(str).unique() if x.strip()])
    return {"items": items}


@app.get("/domains")
def list_domains():
    if df.empty:
        return {"items": []}
    items = sorted([x for x in df["domain"].dropna().astype(str).unique() if x.strip()])
    return {"items": items}


@app.get("/countries")
def list_countries():
    if df.empty:
        return {"items": []}
    items = sorted([x for x in df["country"].dropna().astype(str).unique() if x.strip()])
    return {"items": items}


@app.get("/search-text")
def search_text(
    q: str,
    domain: Optional[str] = None,
    country: Optional[str] = None,
    source: Optional[str] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    top_k: int = 20,
):
    """
    Text search over (headline + content) + advanced filters.
    """
    if df.empty:
        return {"count": 0, "results": [], "message": f"dataset not loaded from {DATA_PATH}"}

    d = df.copy()
    d = apply_filters(d, domain=domain, country=country, source=source, min_date=min_date, max_date=max_date)

    # Safe text search
    q = (q or "").strip()
    if not q:
        out = select_output_cols(d).head(int(top_k))
        return {"count": int(len(d)), "results": out.to_dict(orient="records")}

    hay = (d["headline"].fillna("") + " " + d["content"].fillna("")).str.lower()
    d = d[hay.str.contains(q.lower(), na=False, regex=False)].copy()

    out = select_output_cols(d).head(int(top_k))
    return {"count": int(len(d)), "results": out.to_dict(orient="records")}


@app.post("/semantic-search")
def semantic_search_api(body: SearchRequest):
    """
    For now: behaves like search-text (POST + filters).
    Next: plug FAISS here.
    """
    if df.empty:
        return {"count": 0, "results": [], "message": f"dataset not loaded from {DATA_PATH}"}

    d = df.copy()
    d = apply_filters(
        d,
        domain=body.domain,
        country=body.country,
        source=body.source,
        min_date=body.min_date,
        max_date=body.max_date,
    )

    q = (body.query or "").strip()
    if q:
        hay = (d["headline"].fillna("") + " " + d["content"].fillna("")).str.lower()
        d = d[hay.str.contains(q.lower(), na=False, regex=False)].copy()

    out = select_output_cols(d).head(int(body.top_k))
    return {"count": int(len(d)), "results": out.to_dict(orient="records")}
