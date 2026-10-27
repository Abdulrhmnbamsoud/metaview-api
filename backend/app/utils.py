# backend/app/utils.py
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Any, List
import re

def normalize_source_name(feed_url: str) -> str:
    u = feed_url.lower()
    # Specialized global fraud feeds
    if "news.google.com" in u:
        if "gl=us" in u:
            if "investment" in u: return "gnews_us_invest"
            return "gnews_us_fraud"
        if "gl=gb" in u:
            if "financial" in u: return "gnews_uk_fraud"
            return "gnews_uk_scam"
        if "gl=ca" in u: return "gnews_ca_fraud"
        if "gl=au" in u: return "gnews_au_fraud"
        if "gl=ae" in u: return "gnews_ae_fraud"
        if "gl=eg" in u: return "gnews_eg_fraud"
        return "gnews_global_fraud"
    if "bleepingcomputer" in u: return "bleeping_computer"
    if "krebsonsecurity" in u: return "krebs_security"
    if "feedburner" in u or "thehackersnews" in u: return "the_hackers_news"
    if "securelist" in u: return "securelist"
    # General news feeds
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
    if "aljazeera" in u: return "aljazeera"
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
    if "cnbc" in u: return "cnbc"
    if "dw.com" in u: return "dw"
    if "france24" in u: return "france24"
    if "lemonde" in u: return "le_monde"
    return "other"

def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")
    except Exception:
        return ""

def safe_dt_to_iso(entry: Any) -> Optional[str]:
    import email.utils
    try:
        if getattr(entry, "published_parsed", None):
            ts = time.mktime(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if getattr(entry, "updated_parsed", None):
            ts = time.mktime(entry.updated_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        # Fallback to string parsing if feedparser tuple was None
        for raw_field in ("published", "updated", "pubDate", "created"):
            val = getattr(entry, raw_field, None)
            if isinstance(val, str) and val.strip():
                try:
                    dt = email.utils.parsedate_to_datetime(val.strip())
                    return dt.astimezone(timezone.utc).isoformat()
                except Exception:
                    pass
                try:
                    dt = datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
                    return dt.astimezone(timezone.utc).isoformat()
                except Exception:
                    pass
    except Exception:
        pass
    return None
