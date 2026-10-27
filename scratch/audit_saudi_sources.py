# scratch/audit_saudi_sources.py
import asyncio
import httpx
import feedparser
from datetime import datetime, timezone, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from backend.app.saudi_sources import _DIRECT, _OFFICIAL, _PROBES, _GNEWS_QUERIES_AR, _gnews
from backend.app.utils import safe_dt_to_iso

async def audit():
    now = datetime.now(timezone.utc)
    day1 = now - timedelta(days=1)
    day7 = now - timedelta(days=7)
    day30 = now - timedelta(days=30)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    print(f"Auditing Saudi Sources at reference time: {now.isoformat()} ...\n")
    results = []

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=10.0) as client:
        # 1. Direct feeds
        for s in _DIRECT:
            try:
                r = await client.get(s.url)
                if r.status_code != 200:
                    results.append({"name": s.name, "kind": s.kind, "status": r.status_code, "entries": 0, "newest": "N/A", "c24": 0, "c7": 0, "err": f"HTTP {r.status_code}"})
                    continue
                parsed = feedparser.parse(r.text)
                entries = parsed.entries or []
                newest_dt = None
                c24, c7 = 0, 0
                for e in entries:
                    iso = safe_dt_to_iso(e)
                    if iso:
                        try:
                            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                            if not newest_dt or dt > newest_dt:
                                newest_dt = dt
                            if dt >= day1: c24 += 1
                            if dt >= day7: c7 += 1
                        except: pass
                results.append({"name": s.name, "kind": s.kind, "status": 200, "entries": len(entries), "newest": newest_dt.isoformat() if newest_dt else "None", "c24": c24, "c7": c7, "err": "OK" if len(entries)>0 else "Empty"})
            except Exception as ex:
                results.append({"name": s.name, "kind": s.kind, "status": "ERR", "entries": 0, "newest": "N/A", "c24": 0, "c7": 0, "err": str(ex)[:35]})

        # 2. GNews queries
        for code, q in _GNEWS_QUERIES_AR[:5]:
            url = _gnews(q)
            try:
                r = await client.get(url)
                parsed = feedparser.parse(r.text)
                entries = parsed.entries or []
                newest_dt = None
                c24, c7 = 0, 0
                for e in entries:
                    iso = safe_dt_to_iso(e)
                    if iso:
                        try:
                            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                            if not newest_dt or dt > newest_dt: newest_dt = dt
                            if dt >= day1: c24 += 1
                            if dt >= day7: c7 += 1
                        except: pass
                results.append({"name": f"GNews: {q}", "kind": "gnews", "status": 200, "entries": len(entries), "newest": newest_dt.isoformat() if newest_dt else "None", "c24": c24, "c7": c7, "err": "OK"})
            except Exception as ex:
                results.append({"name": f"GNews: {q}", "kind": "gnews", "status": "ERR", "entries": 0, "newest": "N/A", "c24": 0, "c7": 0, "err": str(ex)[:35]})

    print(f"| Source Name | Kind | Status | Entries | Newest Date | Last 24h | Last 7d | Notes |")
    print(f"|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['kind']} | {r['status']} | {r['entries']} | {r['newest']} | {r['c24']} | {r['c7']} | {r['err']} |")

if __name__ == "__main__":
    asyncio.run(audit())
