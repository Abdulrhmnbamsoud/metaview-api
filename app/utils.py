# app/utils.py
import time
from datetime import datetime, timezone
from typing import Any

def normalize_source_name(feed_url: str) -> str:
    u = feed_url.lower()
    if "reuters" in u: return "reuters"
    if "bloomberg" in u: return "bloomberg"
    if "dj.com" in u or "wsj" in u: return "wsj"
    if "nytimes" in u: return "nyt"
    if "economist" in u: return "economist"
    if "cnn" in u: return "cnn"
    if "ft.com" in u: return "financial_times"
    if "washingtonpost" in u: return "washington_post"
    if "theatlantic" in u: return "the_atlantic"
    if "foreignaffairs" in u: return "foreign_affairs"
    if "alarabiya" in u: return "alarabiya"
    if "apnews" in u: return "ap_news"
    if "politico" in u: return "politico"
    if "axios" in u: return "axios"
    if "bbci" in u or "bbc" in u: return "bbc"
    if "cbsnews" in u: return "cbs"
    if "newsweek" in u: return "newsweek"
    if "theguardian" in u: return "the_guardian"
    if "businessinsider" in u: return "business_insider"
    if "thehill" in u: return "the_hill"
    if "nbcnews" in u: return "nbc"
    if "abcnews" in u: return "abc"
    if "dw.com" in u: return "dw"
    if "france24" in u: return "france24"
    if "lemonde" in u: return "le_monde"
    return "other"

def safe_dt_to_iso(entry: Any) -> str:
    try:
        if getattr(entry, "published_parsed", None):
            ts = time.mktime(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if getattr(entry, "updated_parsed", None):
            ts = time.mktime(entry.updated_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        pass
    return ""
