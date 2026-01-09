from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ArticleOut(BaseModel):
    headline: str
    content: Optional[str] = None
    article_summary: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    country: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None
    entities: Optional[str] = None

class SearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    sort: str
    count: int
    results: List[ArticleOut]
