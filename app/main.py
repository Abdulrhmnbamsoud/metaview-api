import os
import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "data/news_latest.csv")

df = pd.read_csv(DATA_PATH)

# نظافة بسيطة
if "published_at" in df.columns:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

df["headline"] = df.get("headline", "").fillna("").astype(str)
df["content"] = df.get("content", "").fillna("").astype(str)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="MetaView API", version="1.0.0")

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
    return {"status": "ok", "service": "metaview"}

@app.post("/semantic-search")
def semantic_search_api(body: SearchRequest):
    # skeleton response (we will connect the real pipeline later)
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
