# backend/app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal


# =========================
# Ingest
# =========================
class IngestRequest(BaseModel):
    feeds: Optional[List[str]] = Field(
        default=None,
        description="Optional custom feed list",
        examples=[[]]
    )
    limit_per_feed: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Max items per feed",
        examples=[30]
    )
    allow_web_scrape: bool = Field(
        default=False,
        description="Optionally scrape article webpage if RSS has no image"
    )

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

    category: Optional[str] = None 
    entity: Optional[str] = None     
    image_url: Optional[str] = None


# =========================
# Search Response
# =========================
class SearchResponse(BaseModel):
    count: int
    results: List[Article]

    total_rows: Optional[int] = None
    returned: Optional[int] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    offset: Optional[int] = None

    sort_by: Optional[str] = None
    sort_dir: Optional[str] = None


# =========================
# Health & Stats
# =========================
class HealthResponse(BaseModel):
    status: str
    service: str
    rows: int
    db_path: str
    time: str


class DistributionItem(BaseModel):
    source: Optional[str] = None
    country: Optional[str] = None
    count: int


class StatsResponse(BaseModel):
    total_articles: int
    total_sources: int
    articles_last_24h: int
    last_sync: Optional[str] = None
    source_distribution: List[Dict[str, Any]]

class ThreatIntel(BaseModel):
    id: int
    article_id: Optional[int] = None
    title: str
    description: str
    threat_type: str
    event_type: Optional[str] = None
    evidence_confidence: Optional[float] = None
    analysis_confidence: Optional[float] = None
    verified_facts: Optional[str] = None
    analytical_assessment: Optional[str] = None
    detection_indicators: Optional[str] = None
    preventive_recommendations: Optional[str] = None
    evidence_urls: Optional[str] = None
    confidence: Optional[float] = None
    original_url: Optional[str] = None
    related_links: Optional[str] = None
    is_saudi: Optional[bool] = None
    is_financial: Optional[bool] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class SourceTrust(BaseModel):
    source_id: int
    trust_score: Optional[float] = None
    last_checked: Optional[str] = None
    notes: Optional[str] = None

class CVEReference(BaseModel):
    cve_id: str
    description: Optional[str] = None
    severity: Optional[str] = None
    published_at: Optional[str] = None
    reference_url: Optional[str] = None

class IncidentLink(BaseModel):
    id: int
    threat_id: int
    related_threat_id: int
    relation_type: Optional[str] = None

    country_distribution: List[Dict[str, Any]]
