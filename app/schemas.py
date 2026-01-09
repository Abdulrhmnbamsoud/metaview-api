# app/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class IngestRequest(BaseModel):
    feeds: Optional[List[str]] = None

class SearchResponse(BaseModel):
    count: int
    results: List[Dict[str, Any]]
