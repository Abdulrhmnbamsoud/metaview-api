# scratch/audit_sources_fast.py
import asyncio
import httpx
import feedparser
from datetime import datetime, timezone, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from backend.app.sources import GLOBAL_FRAUD_FEEDS, GENERAL_WORLD_FEEDS
from backend.app.utils import safe_dt_to_iso

async def check_one(client, sem, cat, url, day1, day7, day30):
    async with sem:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                return {
                    "cat": cat, "url": url, "status": r.status_code, "entries": 0,
                    "newest": "N/A", "last24h": 0, "last7d": 0, "last30d": 0,
                    "err": f"HTTP {r.status_code}"
                }
            parsed = feedparser.parse(r.text)
            entries = parsed.entries or []
            newest_dt = None
            c_24h, c_7d, c_30d = 0, 0, 0

            for e in entries:
                iso = safe_dt_to_iso(e)
                if iso:
                    try:
                        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                        if not newest_dt or dt > newest_dt:
                            newest_dt = dt
                        if dt >= day1: c_24h += 1
                        if dt >= day7: c_7d += 1
                        if dt >= day30: c_30d += 1
                    except:
                        pass

            return {
                "cat": cat, "url": url, "status": 200, "entries": len(entries),
                "newest": newest_dt.isoformat() if newest_dt else "No date",
                "last24h": c_24h, "last7d": c_7d, "last30d": c_30d,
                "err": "OK" if len(entries) > 0 else "Empty Feed"
            }
        except Exception as ex:
            return {
                "cat": cat, "url": url, "status": "ERR", "entries": 0,
                "newest": "N/A", "last24h": 0, "last7d": 0, "last30d": 0,
                "err": str(ex)[:40]
            }

async def audit():
    now = datetime.now(timezone.utc)
    day1 = now - timedelta(days=1)
    day7 = now - timedelta(days=7)
    day30 = now - timedelta(days=30)

    feeds = [("Global Fraud", u) for u in GLOBAL_FRAUD_FEEDS] + [("General World", u) for u in GENERAL_WORLD_FEEDS]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    sem = asyncio.Semaphore(10)
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=10.0) as client:
        tasks = [check_one(client, sem, cat, url, day1, day7, day30) for cat, url in feeds]
        results = await asyncio.gather(*tasks)

    print(f"| Category | Source URL | Status | Total Items | Newest Item Date | Last 24h | Last 7d | Last 30d | Notes |", flush=True)
    print(f"|---|---|---|---|---|---|---|---|---|", flush=True)
    for r in results:
        short_url = r["url"] if len(r["url"]) < 55 else r["url"][:52] + "..."
        print(f"| {r['cat']} | `{short_url}` | {r['status']} | {r['entries']} | {r['newest']} | {r['last24h']} | {r['last7d']} | {r['last30d']} | {r['err']} |", flush=True)

if __name__ == "__main__":
    asyncio.run(audit())
