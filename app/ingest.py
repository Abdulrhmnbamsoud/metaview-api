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
    r = await client.get(url)
    r.raise_for_status()
    parsed = feedparser.parse(r.text)
    return {"url": url, "parsed": parsed}


async def ingest_once(
    feed_urls: List[str],
    limit_per_feed: int = MAX_ITEMS_PER_FEED
) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}

    # clamp safety
    if not isinstance(limit_per_feed, int):
        limit_per_feed = MAX_ITEMS_PER_FEED
    limit_per_feed = max(1, min(limit_per_feed, 200))

    result: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources": [],
        "inserted_total": 0,
        "errors_total": 0,
        "skipped_no_url_total": 0,
        "skipped_dup_total": 0,
    }

    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        headers=headers,
        follow_redirects=True,
        timeout=timeout
    ) as client:
        for feed_url in feed_urls:
            src_name = normalize_source_name(feed_url)
            t0 = time.time()

            inserted = 0
            fetched_entries = 0
            skipped_no_url = 0
            skipped_dup = 0
            err = None

            try:
                data = await fetch_feed(client, feed_url)
                entries = (data["parsed"].entries or [])
                fetched_entries = len(entries)

                for e in entries[:limit_per_feed]:
                    headline = (getattr(e, "title", "") or "").strip()
                    link = (getattr(e, "link", "") or "").strip()
                    summary = (getattr(e, "summary", "") or getattr(e, "description", "") or "").strip()
                    published_at = safe_dt_to_iso(e)

                    if not link:
                        skipped_no_url += 1
                        continue

                    item = {
                        "source": src_name,
                        "domain": "",
                        "country": "",
                        "headline": headline,
                        "content": "",          # خفيف (Studio AI يجيب النص الكامل)
                        "article_summary": summary,
                        "url": link,
                        "published_at": published_at,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "category": None,
                        "entity": None,
                    }

                    ok = upsert_article(item)
                    if ok:
                        inserted += 1
                    else:
                        skipped_dup += 1

            except Exception as e:
                err = str(e)
                result["errors_total"] += 1

            took = round(time.time() - t0, 3)

            result["inserted_total"] += inserted
            result["skipped_no_url_total"] += skipped_no_url
            result["skipped_dup_total"] += skipped_dup

            result["sources"].append({
                "source": src_name,
                "feed_url": feed_url,
                "fetched_entries": fetched_entries,
                "inserted": inserted,
                "skipped_no_url": skipped_no_url,
                "skipped_dup": skipped_dup,
                "limit_per_feed": limit_per_feed,
                "time_sec": took,
                "error": err
            })

    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["rows_after"] = row_count()
    return result
