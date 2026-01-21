# app/ingest.py
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import httpx
import feedparser

from .config import REQUEST_TIMEOUT, USER_AGENT, MAX_ITEMS_PER_FEED
from .utils import normalize_source_name, safe_dt_to_iso
from .db import upsert_article, row_count

async def fetch_feed(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    r = await client.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    parsed = feedparser.parse(r.text)
    return {"url": url, "parsed": parsed}

async def ingest_once(feed_urls: List[str], limit_per_feed: int = MAX_ITEMS_PER_FEED) -> Dict[str, Any]:
    max_items = min(int(limit_per_feed), int(MAX_ITEMS_PER_FEED))
                for e in entries[:max_items]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    result = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources": [],
        "inserted_total": 0,
        "errors_total": 0,
    }

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        for feed_url in feed_urls:
            src_name = normalize_source_name(feed_url)
            t0 = time.time()
            inserted = 0
            err = ""
            count = 0

            try:
                data = await fetch_feed(client, feed_url)
                entries = (data["parsed"].entries or [])
                count = len(entries)

                for e in entries[:MAX_ITEMS_PER_FEED]:
                    headline = getattr(e, "title", "") or ""
                    link = getattr(e, "link", "") or ""
                    summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                    published_at = safe_dt_to_iso(e)

                    item = {
                        "source": src_name,
                        "domain": "",
                        "country": "",
                        "headline": headline.strip(),
                        "content": "", 
                        "article_summary": summary.strip(),
                        "url": link.strip(),
                        "published_at": published_at,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }

                    if item["url"] and upsert_article(item):
                        inserted += 1

            except Exception as e:
                err = str(e)
                result["errors_total"] += 1

            took = round(time.time() - t0, 3)
            result["sources"].append({
                "source": src_name,
                "feed_url": feed_url,
                "fetched_entries": count,
                "inserted": inserted,
                "time_sec": took,
                "error": err
            })
            result["inserted_total"] += inserted

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["rows_after"] = row_count()
    return result
