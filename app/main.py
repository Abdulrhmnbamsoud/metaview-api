from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import pandas as pd

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="MetaView API", version="1.0.0")

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = os.getenv("DATA_PATH", "data/news_latest.csv")

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure expected text columns exist
    if "headline" not in df.columns:
        df["headline"] = ""
    if "content" not in df.columns:
        df["content"] = ""
    if "article_summary" not in df.columns:
        df["article_summary"] = ""
    if "domain" not in df.columns:
        df["domain"] = ""
    if "country" not in df.columns:
        df["country"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    if "url" not in df.columns:
        df["url"] = ""

    # Clean text
    df["headline"] = df["headline"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)
    df["article_summary"] = df["article_summary"].fillna("").astype(str)

    # Parse dates (optional)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

    return df

try:
    df = load_df(DATA_PATH)
except Exception as e:
    # If file not found or parse error, keep empty DF but don't crash the server
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
    return {"status": "ok", "service": "metaview", "rows": int(df.shape[0])}

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
    """
    Text search over (headline + content) + advanced filters.
    """
    if df.empty:
        return {"count": 0, "results": [], "message": "dataset not loaded"}

    d = df.copy()

    # Filters
    if domain:
        d = d[d["domain"].astype(str).str.lower() == domain.lower()]
    if country:
        d = d[d["country"].astype(str).str.lower() == country.lower()]
    if source:
        d = d[d["source"].astype(str).str.lower() == source.lower()]

    # Date filters
    if "published_at" in d.columns:
        if min_date:
            md = pd.to_datetime(min_date, errors="coerce", utc=True)
            if pd.notna(md):
                d = d[d["published_at"] >= md]
        if max_date:
            xd = pd.to_datetime(max_date, errors="coerce", utc=True)
            if pd.notna(xd):
                d = d[d["published_at"] <= xd]

    # Text search
    hay = (d["headline"].fillna("") + " " + d["content"].fillna("")).str.lower()
    d = d[hay.str.contains(q.lower(), na=False)].copy()

    # Output columns (only those that exist)
    wanted = [
        "published_at","source","domain","country",
        "headline","article_summary","sentiment_score",
        "cluster_id","cluster_summary","url"
    ]
    cols = [c for c in wanted if c in d.columns]

    out = d[cols].head(int(top_k))
    return {"count": int(len(d)), "results": out.to_dict(orient="records")}

@app.post("/semantic-search")
def semantic_search_api(body: SearchRequest):
    """
    Placeholder until we wire FAISS/embeddings endpoint.
    """
    return {
        "query": body.query,
        "filters": {
            "domain": body.domain,
            "country": body.country,
            "source": body.source,
            "min_date": body.min_date,
            "max_date": body.max_date,
        },
        "count": 0,
        "results": []
    }
