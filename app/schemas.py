# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


# =========================
# Ingest
# =========================
class IngestRequest(BaseModel):
    feeds: Optional[List[str]] = Field(default=None, description="Optional custom feed list")
    limit_per_feed: int = Field(default=30, ge=1, le=200, description="Max items per feed")


# =========================
# Core Article Model
# =========================
class Article(BaseModel):
    headline: str
    content: Optional[str] = None
    article_summary: Optional[str] = None
    published_at: Optional[str] = None

    source: Optional[str] = None
    domain: Optional[str] = None
    country: Optional[str] = None
    url: Optional[str] = None

    # مطلوبات اللجنة
    category: Optional[str] = None 
    entity: Optional[str] = None     


# =========================
# Responses
# =========================
SortField = Literal["published_at", "source", "country", "domain"]
SortDir = Literal["asc", "desc"]

class SearchResponse(BaseModel):
    count: int
    results: List[Article]

    # Pagination meta
    total_rows: Optional[int] = None
    returned: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    offset: Optional[int] = None

    # Sort meta
    sort_by: Optional[str] = None
    sort_dir: Optional[str] = None
