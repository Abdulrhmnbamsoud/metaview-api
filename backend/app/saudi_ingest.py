# backend/app/saudi_ingest.py
"""Saudi Fraud Intelligence ingestion pipeline.

Flow per run
------------
1. For every enabled source in :mod:`saudi_sources` perform a REAL HTTP request
   (RSS / SPA JSON / Google-News search / reachability probe).
2. Each entry is checked against ``ingest_seen`` (never process a link twice), pre-gated
   on an explicit fraud signal, resolved to its original URL (Google News links are
   decoded), de-duplicated (canonical URL, content fingerprint, near-identical title)
   and - only for candidates - the article page is downloaded (robots.txt respected).
3. The fraud engine decides: fraud + Saudi => stored in ``articles`` (+ ``threat_intel``);
   anything else is only logged in ``ingest_seen`` with the exclusion reason.
4. Every source result (success or failure + reason) is written to ``fetch_log`` and
   ``source_registry``. Existing articles are never deleted or modified.

Run manually::

    python -m app.saudi_ingest --run [--only code1,code2]   # from the backend/ directory
"""
from __future__ import annotations

import calendar
import hashlib
import html as html_lib
import json
import logging
import random
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib import robotparser
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
from bs4 import BeautifulSoup

from . import fraud_intel
from .config import (
    REQUEST_TIMEOUT, SAUDI_AUTO_INGEST, SAUDI_INGEST_EVERY_HOURS, SAUDI_MAX_BODY_FETCH,
    SAUDI_MAX_DECODE_PER_RUN, USER_AGENT,
)
from .db import find_duplicate_article, get_db, insert_article_if_new
from .intel import analyze_and_store
from .saudi_sources import SAUDI_SOURCES, SaudiSource
from .utils import extract_domain

log = logging.getLogger("metaview.saudi")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

BOT_TOKEN = "MetaViewBot"
MAX_PAGE_BYTES = 1_500_000
MAX_BODY_CHARS = 8000

# --------------------------------------------------------------------------- #
# URL / text helpers
# --------------------------------------------------------------------------- #
_TRACKING_EXACT = {"fbclid", "gclid", "igshid", "yclid", "mc_cid", "mc_eid", "ref", "ref_src", "spm", "cmpid", "_ga"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_url(url: str) -> str:
    """Normalise a URL for storage/de-duplication (https, lowercase host, no tracking/fragment)."""
    try:
        p = urlsplit((url or "").strip())
    except ValueError:
        return url or ""
    scheme = "https" if p.scheme in ("http", "https") else p.scheme
    path = quote(unquote(p.path), safe="/%:@!$&'()*+,;=-._~")
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_EXACT]
    return urlunsplit((scheme, p.netloc.lower(), path, urlencode(query), ""))


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def title_key(title: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", fraud_intel.norm(title))).strip()


def title_tokens(title: str) -> frozenset:
    return frozenset(w for w in title_key(title).split() if len(w) > 2)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_fingerprint(title: str, text: str) -> str:
    base = title_key(title) + "|" + title_key(text)[:400]
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def strip_html(value: str) -> str:
    if not value:
        return ""
    if "<" in value and ">" in value:
        value = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", html_lib.unescape(value)).strip()


def _struct_to_iso(st) -> Optional[str]:
    try:
        return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc).isoformat()
    except Exception:
        return None


def is_saudi_outlet(domain: str) -> bool:
    d = (domain or "").lower()
    return d.endswith(".sa") or any(d.endswith(x) for x in fraud_intel.SAUDI_PUBLISHER_DOMAINS)


# --------------------------------------------------------------------------- #
# Network helpers
# --------------------------------------------------------------------------- #
MAX_RETRY_WAIT = 60.0          # never sleep longer than this for a single Retry-After / backoff
BACKOFF_BUDGET_PER_RUN = 240.0  # total seconds a run may spend sleeping on 429/503
MIN_COOLDOWN = 300.0           # after giving up on a host, leave it alone for at least this long
MAX_COOLDOWN = 3600.0
HOST_MIN_INTERVAL = {"news.google.com": 2.0}  # polite pacing per host (seconds between requests)
DEFAULT_MIN_INTERVAL = 0.25


def parse_retry_after(value: Optional[str], now: Optional[datetime] = None) -> Optional[float]:
    """Parse a Retry-After header (delta-seconds or HTTP-date) into seconds, or None."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - (now or datetime.now(timezone.utc))).total_seconds())
    except Exception:
        return None


class NetPolicy:
    """Per-host pacing, backoff budget and cooldowns shared by every request of the pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: Dict[str, float] = {}
        self.cooldown_until: Dict[str, float] = {}   # host -> monotonic deadline (persists across runs)
        self.backoff_budget = BACKOFF_BUDGET_PER_RUN
        self.events: List[Dict[str, Any]] = []

    def new_run(self) -> None:
        with self._lock:
            self.backoff_budget = BACKOFF_BUDGET_PER_RUN
            self.events = []

    def wait_turn(self, host: str) -> None:
        gap = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
        with self._lock:
            delay = self._last.get(host, 0.0) + gap - time.monotonic()
            self._last[host] = time.monotonic() + max(0.0, delay)
        if delay > 0:
            time.sleep(delay)

    def cooling(self, host: str) -> float:
        return max(0.0, self.cooldown_until.get(host, 0.0) - time.monotonic())

    def set_cooldown(self, host: str, seconds: float) -> None:
        seconds = min(MAX_COOLDOWN, max(MIN_COOLDOWN, seconds))
        self.cooldown_until[host] = time.monotonic() + seconds
        self.events.append({"host": host, "cooldown_sec": int(seconds), "at": now_iso()})

    def take_budget(self, seconds: float) -> bool:
        with self._lock:
            if seconds > self.backoff_budget:
                return False
            self.backoff_budget -= seconds
            return True


_NET = NetPolicy()
RATE_LIMITED_PREFIX = "RATE_LIMITED"


def _backoff_seconds(attempt: int, retry_after: Optional[float]) -> float:
    if retry_after is not None:
        return retry_after
    return min(MAX_RETRY_WAIT, 2.0 * (2 ** attempt)) + random.uniform(0.0, 1.0)


def _http_get(client: httpx.Client, url: str, tries: int = 3) -> Tuple[Optional[httpx.Response], Optional[str]]:
    """GET with pacing and 429/503 handling.

    * waits for the host's minimum interval before every request;
    * on 429/503 honours ``Retry-After`` (capped at MAX_RETRY_WAIT per wait) else exponential
      backoff with jitter, within a per-run backoff budget;
    * when it has to give up it puts the host in cooldown (>= Retry-After, 5-60 min) so no other
      request to that host is made until then, and returns the last response (429/503).
    """
    host = urlsplit(url).netloc.lower()
    remaining = _NET.cooling(host)
    if remaining > 0:
        return None, f"{RATE_LIMITED_PREFIX}: {host} is cooling down for {int(remaining)}s after HTTP 429/503"
    last_err: Optional[str] = None
    for i in range(max(1, tries)):
        _NET.wait_turn(host)
        try:
            r = client.get(url)
        except httpx.HTTPError as exc:
            last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
            if i + 1 < tries:
                time.sleep(0.6 * (2 ** i))
            continue
        if r.status_code not in (429, 503):
            return r, None
        ra = parse_retry_after(r.headers.get("retry-after"))
        wait = _backoff_seconds(i, ra)
        can_wait = wait <= MAX_RETRY_WAIT and i + 1 < tries and _NET.take_budget(wait)
        log.warning("HTTP %s from %s (Retry-After=%s) attempt %d/%d -> %s", r.status_code, host,
                    r.headers.get("retry-after"), i + 1, tries, f"waiting {wait:.1f}s" if can_wait else "giving up")
        if can_wait:
            time.sleep(wait)
            continue
        _NET.set_cooldown(host, ra if ra is not None else MIN_COOLDOWN)
        return r, None
    return None, last_err


def _bad_status(r: httpx.Response) -> Dict[str, Any]:
    """Source result for a non-200 response. 429 (or 503 + Retry-After) is *rate_limited*, not *failed*."""
    ra = r.headers.get("retry-after")
    if r.status_code == 429 or (r.status_code == 503 and ra):
        return {"state": "rate_limited", "http_status": r.status_code, "entries": [],
                "error": f"HTTP {r.status_code} rate limited (Retry-After: {ra or 'n/a'}); host in cooldown, retried next run"}
    return {"state": "failed", "http_status": r.status_code, "error": f"HTTP {r.status_code}", "entries": []}


def _net_failure(err: Optional[str]) -> Dict[str, Any]:
    state = "rate_limited" if (err or "").startswith(RATE_LIMITED_PREFIX) else "failed"
    return {"state": state, "http_status": None, "error": err, "entries": []}


class RobotsCache:
    """robots.txt check (fail-open when robots.txt is missing/unreachable)."""

    def __init__(self, client: httpx.Client):
        self.client = client
        self._cache: Dict[str, Optional[robotparser.RobotFileParser]] = {}

    def allowed(self, url: str) -> bool:
        p = urlsplit(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._cache:
            rp: Optional[robotparser.RobotFileParser] = None
            try:
                r = self.client.get(origin + "/robots.txt", timeout=8)
                if r.status_code == 200 and "html" not in r.headers.get("content-type", "").lower():
                    rp = robotparser.RobotFileParser()
                    rp.parse(r.text.splitlines())
            except Exception:
                rp = None
            self._cache[origin] = rp
        rp = self._cache[origin]
        return True if rp is None else rp.can_fetch(BOT_TOKEN, url)


def _find_long_text(obj: Any, depth: int = 0) -> str:
    """Look for an article body inside embedded JSON (JSON-LD / __NEXT_DATA__)."""
    if depth > 8:
        return ""
    if isinstance(obj, dict):
        for k in ("articleBody", "content", "body", "text", "description"):
            v = obj.get(k)
            if isinstance(v, str) and len(v) > 250:
                return strip_html(v)
        for v in obj.values():
            r = _find_long_text(v, depth + 1)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj[:20]:
            r = _find_long_text(v, depth + 1)
            if r:
                return r
    return ""


def extract_article(page_html: str) -> Dict[str, str]:
    """Best-effort article text + og:image extraction from an article page."""
    soup = BeautifulSoup(page_html, "html.parser")
    out = {"text": "", "image": "", "description": ""}
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        out["image"] = og["content"].strip()
    desc = soup.find("meta", attrs={"property": "og:description"}) or soup.find("meta", attrs={"name": "description"})
    if desc and desc.get("content"):
        out["description"] = strip_html(desc["content"])[:600]

    text = ""
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            text = _find_long_text(json.loads(s.string or ""))
        except Exception:
            text = ""
        if text:
            break
    if not text:
        nd = soup.find("script", attrs={"id": "__NEXT_DATA__"})
        if nd and nd.string:
            try:
                text = _find_long_text(json.loads(nd.string))
            except Exception:
                text = ""
    if not text:
        for t in soup(["script", "style", "nav", "footer", "aside", "form", "header", "noscript"]):
            t.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
        text = "\n".join(p for p in paras if len(p) >= 40)
    out["text"] = re.sub(r"[ \t]+", " ", text)[:MAX_BODY_CHARS]
    return out


# --------------------------------------------------------------------------- #
# Source fetchers -> normalised entries
# --------------------------------------------------------------------------- #
Entry = Dict[str, Any]


def _entry_image(e: Any) -> str:
    for key in ("media_content", "media_thumbnail"):
        for m in getattr(e, key, None) or []:
            if m.get("url"):
                return m["url"]
    for enc in getattr(e, "enclosures", None) or []:
        if "image" in (enc.get("type") or "") and enc.get("href"):
            return enc["href"]
    return ""


def fetch_rss(client: httpx.Client, src: SaudiSource) -> Dict[str, Any]:
    r, err = _http_get(client, src.url)
    if r is None:
        return _net_failure(err)
    if r.status_code != 200:
        return _bad_status(r)
    parsed = feedparser.parse(r.content)
    if not parsed.entries:
        if parsed.bozo:
            ctype = r.headers.get("content-type", "")
            return {"state": "failed", "http_status": 200, "entries": [],
                    "error": f"not a valid RSS/Atom feed (content-type: {ctype or 'unknown'})"}
        return {"state": "empty", "http_status": 200, "error": "feed returned 0 entries", "entries": []}
    entries: List[Entry] = []
    for e in parsed.entries:
        st = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        entry: Entry = {
            "title": strip_html(getattr(e, "title", "") or ""),
            "link": (getattr(e, "link", "") or "").strip(),
            "published_at": _struct_to_iso(st) if st else None,
            "summary": strip_html(getattr(e, "summary", "") or getattr(e, "description", "") or "")[:1500],
            "image": _entry_image(e),
            "publisher": src.name,
            "publisher_domain": src.domain,
        }
        if src.kind == "gnews":
            s = getattr(e, "source", None)
            pub_name = (s.get("title") if s else "") or ""
            pub_href = (s.get("href") if s else "") or ""
            title = entry["title"]
            if pub_name and title.lower().endswith(" - " + pub_name.lower()):
                title = title[: -(len(pub_name) + 3)].rstrip()
            entry.update({"title": title, "summary": "", "publisher": pub_name or "Google News",
                          "publisher_domain": extract_domain(pub_href)})
        entries.append(entry)
    return {"state": "ok", "http_status": 200, "error": None, "entries": entries}


def fetch_nextdata(client: httpx.Client, src: SaudiSource) -> Dict[str, Any]:
    r, err = _http_get(client, src.url)
    if r is None:
        return _net_failure(err)
    if r.status_code != 200:
        return _bad_status(r)
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
    if not m:
        return {"state": "failed", "http_status": 200, "error": "__NEXT_DATA__ not found (site layout changed?)", "entries": []}
    try:
        items = json.loads(m.group(1))["props"]["pageProps"]["mainNews"]
    except Exception as exc:
        return {"state": "failed", "http_status": 200, "error": f"unexpected JSON structure: {exc}", "entries": []}
    entries = []
    for it in items:
        uid = it.get("uuid")
        if not uid or not it.get("title"):
            continue
        ts = it.get("published_at")
        entries.append({
            "title": strip_html(it["title"]),
            "link": f"https://www.spa.gov.sa/ar/{uid}",
            "published_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if isinstance(ts, (int, float)) else None,
            "summary": strip_html(it.get("subtitle") or ""),
            "image": ((it.get("image") or {}).get("path") or ""),
            "publisher": src.name,
            "publisher_domain": src.domain,
        })
    if not entries:
        return {"state": "empty", "http_status": 200, "error": "no news items in page JSON", "entries": []}
    return {"state": "ok", "http_status": 200, "error": None, "entries": entries}


def probe_source(client: httpx.Client, src: SaudiSource) -> Dict[str, Any]:
    r, err = _http_get(client, src.url, tries=1)
    if r is None:
        reason = err or "connection failed"
        if "CERTIFICATE_VERIFY_FAILED" in reason:
            reason = "TLS certificate verification failed (server sends an incomplete certificate chain) - " + reason
        return _net_failure(reason)
    if r.status_code >= 400:
        return _bad_status(r)
    return {"state": "no_feed", "http_status": r.status_code, "entries": [],
            "error": "reachable but exposes no RSS/API (content is JavaScript-rendered or HTML only)"}


def fetch_source(client: httpx.Client, src: SaudiSource) -> Dict[str, Any]:
    if src.kind in ("rss", "gnews"):
        return fetch_rss(client, src)
    if src.kind == "nextdata":
        return fetch_nextdata(client, src)
    return probe_source(client, src)


# --------------------------------------------------------------------------- #
# Persistence helpers (seen cache, registry, log)
# --------------------------------------------------------------------------- #
def _seen(raw_link: str) -> bool:
    con = get_db()
    row = con.execute("SELECT 1 FROM ingest_seen WHERE raw_link = ?", (raw_link,)).fetchone()
    if row:
        con.execute("UPDATE ingest_seen SET last_seen_at = ? WHERE raw_link = ?", (now_iso(), raw_link))
        con.commit()
    con.close()
    return row is not None


def _mark_seen(raw_link: str, canonical: Optional[str], source_code: str, decision: str,
               reason: Optional[str], article_id: Optional[int] = None) -> None:
    ts = now_iso()
    con = get_db()
    con.execute(
        "INSERT INTO ingest_seen (raw_link, canonical_url, source_code, decision, reason, article_id, first_seen_at, last_seen_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(raw_link) DO UPDATE SET decision=excluded.decision, "
        "reason=excluded.reason, article_id=excluded.article_id, canonical_url=excluded.canonical_url, last_seen_at=excluded.last_seen_at",
        (raw_link, canonical, source_code, decision, reason, article_id, ts, ts),
    )
    con.commit()
    con.close()


def _record_source(src: SaudiSource, run_id: str, started: str, res: Dict[str, Any], c: Dict[str, int]) -> None:
    finished = now_iso()
    state = res["state"]
    ok = 1 if state == "ok" else 0
    con = get_db()
    con.execute(
        "INSERT INTO fetch_log (run_id, source_code, started_at, finished_at, ok, http_status, entries_found, "
        "new_articles, duplicates, excluded, fraud_matched, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, src.code, started, finished, ok, res.get("http_status"), len(res.get("entries", [])),
         c["new_articles"], c["duplicates"], c["excluded"], c["fraud_matched"], res.get("error")),
    )
    failed = state == "failed"
    con.execute(
        "INSERT INTO source_registry (code, name, url, kind, country, lang, official, enabled, last_status, "
        "last_success_at, last_failure_at, last_error, last_entries, consecutive_failures) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(code) DO UPDATE SET name=excluded.name, url=excluded.url, "
        "kind=excluded.kind, lang=excluded.lang, official=excluded.official, enabled=excluded.enabled, "
        "last_status=excluded.last_status, "
        "last_success_at=COALESCE(excluded.last_success_at, source_registry.last_success_at), "
        "last_failure_at=COALESCE(excluded.last_failure_at, source_registry.last_failure_at), "
        "last_error=excluded.last_error, last_entries=excluded.last_entries, "
        "consecutive_failures=CASE WHEN excluded.last_status='failed' THEN source_registry.consecutive_failures+1 "
        "WHEN excluded.last_status='rate_limited' THEN source_registry.consecutive_failures ELSE 0 END",
        (src.code, src.name, src.url, src.kind, "SA", src.lang, int(src.official), int(src.enabled), state,
         finished if state in ("ok", "empty", "no_feed") else None, finished if failed else None,
         res.get("error"), len(res.get("entries", [])), 1 if failed else 0),
    )
    con.commit()
    con.close()


def _recent_title_tokens(days: int = 30) -> List[Tuple[int, frozenset]]:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    con = get_db()
    rows = con.execute(
        "SELECT id, headline FROM articles WHERE source_code IS NOT NULL AND published_at >= ?", (since,)).fetchall()
    con.close()
    return [(r["id"], title_tokens(r["headline"] or "")) for r in rows]


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
class RunContext:
    def __init__(self, client: httpx.Client):
        self.client = client
        self.robots = RobotsCache(client)
        self.decode_budget = SAUDI_MAX_DECODE_PER_RUN
        self.body_budget = SAUDI_MAX_BODY_FETCH
        self.recent = _recent_title_tokens()
        self.excluded_by_reason: Dict[str, int] = {}
        self.decode_failures: List[str] = []
        self.decode_blocked = False  # circuit breaker: Google throttled the decoder in this run
        self.examples: List[Dict[str, Any]] = []

    def exclude(self, reason: str) -> None:
        self.excluded_by_reason[reason] = self.excluded_by_reason.get(reason, 0) + 1


DECODE_INTERVAL = 1.0
DECODE_THROTTLE_STREAK = 3
_THROTTLE_HINTS = ("429", "too many", "rate limit", "ratelimit", "quota", "unusual traffic", "sorry")


def _decode_google_links(ctx: RunContext, links: List[str]) -> List[Optional[str]]:
    """Decode Google-News redirect links to the original article URLs.

    Paced at DECODE_INTERVAL; after DECODE_THROTTLE_STREAK consecutive throttled responses the
    circuit breaker opens (``ctx.decode_blocked``) and the remaining links are left for the next run.
    """
    from googlenewsdecoder import gnewsdecoder
    out: List[Optional[str]] = []
    streak = 0
    for idx, link in enumerate(links):
        if ctx.decode_blocked:
            out.extend([None] * (len(links) - idx))
            break
        msg = ""
        try:
            res = gnewsdecoder(link, interval=DECODE_INTERVAL)
            if res.get("success") and res.get("decoded_url"):
                out.append(res["decoded_url"])
                streak = 0
                continue
            msg = str(res.get("message", "decode failed"))
            out.append(None)
        except Exception as exc:  # network / library error - keep the run going
            msg = f"{type(exc).__name__}: {str(exc)[:100]}"
            out.append(None)
        ctx.decode_failures.append(msg[:120])
        if any(h in msg.lower() for h in _THROTTLE_HINTS):
            streak += 1
            if streak >= DECODE_THROTTLE_STREAK:
                ctx.decode_blocked = True
                _NET.set_cooldown("news.google.com", MIN_COOLDOWN)
                log.warning("Google News decoder throttled (%d in a row): pausing decoding for this run", streak)
        else:
            streak = 0
    return out


def _fetch_body(ctx: RunContext, url: str) -> Tuple[str, str, str]:
    """Download the article page politely. Returns (text, image, og_description)."""
    if ctx.body_budget <= 0 or not ctx.robots.allowed(url):
        return "", "", ""
    ctx.body_budget -= 1
    time.sleep(0.25)
    r, _err = _http_get(ctx.client, url, tries=1)
    if r is None or r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower():
        return "", "", ""
    try:
        data = extract_article(r.text[:MAX_PAGE_BYTES])
    except Exception:
        return "", "", ""
    return data["text"], data["image"], data["description"]


def _process_entry(ctx: RunContext, src: SaudiSource, entry: Entry, resolved_url: Optional[str],
                   c: Dict[str, int]) -> None:
    raw = entry["link"]
    url = resolved_url or raw
    cleaned = canonical_url(url)
    domain = extract_domain(cleaned)
    title = entry["title"]

    # --- duplicate checks (cheap, before downloading anything) ---
    if find_duplicate_article(cleaned, None) is not None:
        c["duplicates"] += 1
        _mark_seen(raw, cleaned, src.code, "duplicate", "url")
        return
    toks = title_tokens(title)
    for aid, other in ctx.recent:
        if jaccard(toks, other) >= 0.85:
            c["duplicates"] += 1
            _mark_seen(raw, cleaned, src.code, "duplicate", f"near_title(article {aid})")
            return

    body, og_image, og_desc = _fetch_body(ctx, cleaned)
    summary = entry.get("summary") or og_desc or ""
    meta = {
        "domain": domain,
        "saudi_publisher": is_saudi_outlet(domain) or is_saudi_outlet(src.domain),
        "saudi_scope": src.saudi_scope and src.kind != "gnews",
        "official": src.official or domain.endswith(".gov.sa"),
    }
    analysis, reason = fraud_intel.analyze(title, summary, body, source=meta, require_saudi=True)
    if analysis is None:
        c["excluded"] += 1
        ctx.exclude(reason or "excluded")
        _mark_seen(raw, cleaned, src.code, "excluded", reason)
        return

    fingerprint = content_fingerprint(title, body or summary)
    fetched = now_iso()
    agencies = analysis["verified_facts"].get("agencies_mentioned") or []
    item = {
        "source": entry.get("publisher") or src.name,
        "domain": domain,
        "country": "SA",
        "headline": title,
        "content": body,
        "article_summary": (summary or body[:300]),
        "url": cleaned,
        "published_at": entry.get("published_at") or fetched,
        "created_at": fetched,
        "category": analysis["fraud_category_ar"],
        "entity": agencies[0] if agencies else None,
        "image_url": entry.get("image") or og_image or "",
        "content_hash": fingerprint,
        "fetched_at": fetched,
        "source_code": src.code,
        "language": analysis["language"],
    }
    article_id, created = insert_article_if_new(item)
    if article_id is None:
        c["errors"] += 1
        return
    if not created:
        c["duplicates"] += 1
        _mark_seen(raw, cleaned, src.code, "duplicate", "content_hash", article_id)
        return

    c["new_articles"] += 1
    intel_article = {**item, "country": "", "article_id": article_id, "saudi_scope": meta["saudi_scope"],
                     "saudi_publisher": meta["saudi_publisher"], "official": meta["official"]}
    threat_id, why = analyze_and_store(intel_article, require_saudi=True)
    if threat_id is None:
        c["errors"] += 1
        log.warning("intel store failed for %s (%s)", cleaned, why)
    else:
        c["fraud_matched"] += 1
        ctx.recent.append((article_id, toks))
        ctx.examples.append({"threat_id": threat_id, "article_id": article_id, "title": title, "url": cleaned,
                             "source": item["source"], "published_at": item["published_at"],
                             "category": analysis["fraud_category_ar"], "saudi_basis": analysis["saudi_basis"]})
    _mark_seen(raw, cleaned, src.code, "stored", None, article_id)


def _process_source(ctx: RunContext, src: SaudiSource, res: Dict[str, Any]) -> Dict[str, int]:
    c = {"new_articles": 0, "duplicates": 0, "excluded": 0, "fraud_matched": 0, "errors": 0, "unseen": 0,
         "deferred": 0, "no_url": 0}
    entries: List[Entry] = sorted(res.get("entries", []), key=lambda e: e.get("published_at") or "", reverse=True)

    todo: List[Entry] = []
    for e in entries:
        if not e.get("link"):
            c["no_url"] += 1
            continue
        if _seen(e["link"]):
            c["duplicates"] += 1
            continue
        c["unseen"] += 1
        # explicit fraud signal in the headline/summary is required before any further work
        if not fraud_intel.has_fraud_signal(f"{e['title']} {e.get('summary', '')}"):
            c["excluded"] += 1
            ctx.exclude("no_fraud_signal")
            _mark_seen(e["link"], None, src.code, "excluded", "no_fraud_signal")
            continue
        todo.append(e)

    resolved: List[Optional[str]]
    if src.kind == "gnews" and todo and ctx.decode_blocked:
        c["deferred"] = len(todo)  # Google is throttling us: leave unseen, retry next run
        todo, resolved = [], []
    elif src.kind == "gnews" and todo:
        allowed = todo[: max(0, ctx.decode_budget)]
        c["deferred"] = len(todo) - len(allowed)  # left unseen -> handled by a later run
        ctx.decode_budget -= len(allowed)
        resolved = _decode_google_links(ctx, [e["link"] for e in allowed])
        todo = allowed
    else:
        resolved = [None] * len(todo)

    for e, url in zip(todo, resolved):
        if src.kind == "gnews" and not url:
            c["errors"] += 1  # not marked seen -> retried next run
            continue
        try:
            _process_entry(ctx, src, e, url, c)
        except Exception as exc:
            c["errors"] += 1
            log.exception("entry failed (%s): %s", e.get("link"), exc)
    return c


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
_RUN_LOCK = threading.Lock()
RUN_STATE: Dict[str, Any] = {"running": False, "run_id": None, "started_at": None, "last_report": None, "trigger": None}

LEASE_NAME = "saudi_ingest"
LEASE_TTL_MIN = 20  # a crashed run frees the lease after this long; every source extends it


# --- cross-process lease ---------------------------------------------------- #
# A threading.Lock only protects one Python process. The CLI, the API server (and, briefly, an
# old+new instance during a deploy) are different processes sharing one SQLite file, so mutual
# exclusion is enforced through an expiring row in ``ingest_lease``.
def _acquire_lease(holder: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Try to take the lease. Returns (acquired, current_holder, expires_at)."""
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(minutes=LEASE_TTL_MIN)).isoformat()
    con = get_db()
    try:
        con.execute("BEGIN IMMEDIATE")  # serialises competing processes
        row = con.execute("SELECT holder, expires_at FROM ingest_lease WHERE name = ?", (LEASE_NAME,)).fetchone()
        if row and row["holder"] != holder and row["expires_at"] > now.isoformat():
            con.execute("ROLLBACK")
            return False, row["holder"], row["expires_at"]
        con.execute(
            "INSERT INTO ingest_lease (name, holder, acquired_at, expires_at) VALUES (?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET holder=excluded.holder, acquired_at=excluded.acquired_at, "
            "expires_at=excluded.expires_at", (LEASE_NAME, holder, now.isoformat(), expires))
        con.execute("COMMIT")
        return True, holder, expires
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


def _heartbeat_lease(holder: str) -> None:
    expires = (datetime.now(timezone.utc) + timedelta(minutes=LEASE_TTL_MIN)).isoformat()
    con = get_db()
    con.execute("UPDATE ingest_lease SET expires_at = ? WHERE name = ? AND holder = ?", (expires, LEASE_NAME, holder))
    con.commit()
    con.close()


def _release_lease(holder: str) -> None:
    con = get_db()
    con.execute("DELETE FROM ingest_lease WHERE name = ? AND holder = ?", (LEASE_NAME, holder))
    con.commit()
    con.close()


def lease_status() -> Dict[str, Any]:
    """Is an ingestion currently running anywhere (any process/instance)?"""
    try:
        con = get_db()
        row = con.execute("SELECT holder, acquired_at, expires_at FROM ingest_lease WHERE name = ?", (LEASE_NAME,)).fetchone()
        con.close()
    except Exception:
        return {"active": False}
    if row and row["expires_at"] > datetime.now(timezone.utc).isoformat():
        return {"active": True, "holder": row["holder"], "acquired_at": row["acquired_at"], "expires_at": row["expires_at"]}
    return {"active": False}


def run_saudi_ingest(trigger: str = "manual", only: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute one full ingestion run synchronously and return a detailed report."""
    if not _RUN_LOCK.acquire(blocking=False):
        return {"status": "already_running", "scope": "this_process", "run_id": RUN_STATE.get("run_id")}
    run_id = uuid.uuid4().hex[:12]
    try:
        acquired, owner, until = _acquire_lease(run_id)
    except Exception:
        _RUN_LOCK.release()
        raise
    if not acquired:
        _RUN_LOCK.release()
        log.warning("ingest %s skipped: lease held by %s until %s", run_id, owner, until)
        return {"status": "already_running", "scope": "other_process", "holder": owner, "lease_expires_at": until}
    started = now_iso()
    t0 = time.time()
    _NET.new_run()
    RUN_STATE.update(running=True, run_id=run_id, started_at=started, trigger=trigger)
    try:
        sources = [s for s in SAUDI_SOURCES if s.enabled and (not only or s.code in only)]
        per_source: List[Dict[str, Any]] = []
        totals = {"entries_found": 0, "unseen": 0, "duplicates": 0, "excluded": 0, "new_articles": 0,
                  "fraud_matched": 0, "errors": 0, "deferred": 0}
        with httpx.Client(headers={"User-Agent": USER_AGENT, "Accept": "*/*", "Accept-Language": "ar,en;q=0.8"},
                          follow_redirects=True, timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=REQUEST_TIMEOUT)) as client:
            ctx = RunContext(client)
            for src in sources:
                s_started = now_iso()
                _heartbeat_lease(run_id)
                try:
                    res = fetch_source(client, src)
                    c = _process_source(ctx, src, res) if res["state"] == "ok" else {
                        "new_articles": 0, "duplicates": 0, "excluded": 0, "fraud_matched": 0, "errors": 0,
                        "unseen": 0, "deferred": 0, "no_url": 0}
                except Exception as exc:  # a source must never abort the whole run
                    log.exception("source %s crashed", src.code)
                    res = {"state": "failed", "http_status": None, "error": f"{type(exc).__name__}: {exc}", "entries": []}
                    c = {"new_articles": 0, "duplicates": 0, "excluded": 0, "fraud_matched": 0, "errors": 1,
                         "unseen": 0, "deferred": 0, "no_url": 0}
                _record_source(src, run_id, s_started, res, c)
                totals["entries_found"] += len(res.get("entries", []))
                for k in ("unseen", "duplicates", "excluded", "new_articles", "fraud_matched", "errors", "deferred"):
                    totals[k] += c[k]
                per_source.append({
                    "code": src.code, "name": src.name, "kind": src.kind, "url": src.url, "state": res["state"],
                    "http_status": res.get("http_status"), "error": res.get("error"),
                    "entries_found": len(res.get("entries", [])), **{k: c[k] for k in (
                        "unseen", "duplicates", "excluded", "new_articles", "fraud_matched", "errors", "deferred")},
                })
                log.info("source %-26s %-12s entries=%-3d new=%-3d dup=%-3d excl=%-3d err=%s", src.code, res["state"],
                         len(res.get("entries", [])), c["new_articles"], c["duplicates"], c["excluded"], res.get("error"))

        con = get_db()
        saudi_total = con.execute("SELECT COUNT(*) AS n FROM threat_intel WHERE is_saudi = 1").fetchone()["n"]
        con.close()
        states = [p["state"] for p in per_source]
        report = {
            "status": "completed", "run_id": run_id, "trigger": trigger, "started_at": started,
            "finished_at": now_iso(), "duration_sec": round(time.time() - t0, 1),
            "sources_tested": len(per_source),
            "sources_ok": states.count("ok"), "sources_failed": states.count("failed"),
            "sources_rate_limited": states.count("rate_limited"),
            "sources_empty": states.count("empty"), "sources_no_feed": states.count("no_feed"),
            "entries_found": totals["entries_found"],
            "entries_already_seen_or_duplicate": totals["duplicates"],
            "entries_new_evaluated": totals["unseen"],
            "excluded": totals["excluded"], "excluded_by_reason": ctx.excluded_by_reason,
            "new_fraud_articles_saved": totals["fraud_matched"],
            "deferred_for_next_run": totals["deferred"],
            "errors": totals["errors"], "decode_failures": len(ctx.decode_failures),
            "decode_failure_samples": ctx.decode_failures[:5],
            "decode_rate_limited": ctx.decode_blocked,
            "rate_limit_events": list(_NET.events),
            "saudi_fraud_total_in_db": saudi_total,
            "new_examples": ctx.examples[:25],
            "sources": per_source,
        }
        RUN_STATE["last_report"] = report
        return report
    finally:
        RUN_STATE.update(running=False)
        try:
            _release_lease(run_id)
        except Exception:
            log.exception("could not release ingest lease %s", run_id)
        _RUN_LOCK.release()


def start_background_run(trigger: str = "manual", only: Optional[List[str]] = None) -> Dict[str, Any]:
    """Start a run in a background thread (used by the API); returns immediately."""
    if RUN_STATE.get("running"):
        return {"status": "already_running", "scope": "this_process", "run_id": RUN_STATE.get("run_id")}
    lease = lease_status()
    if lease.get("active"):
        return {"status": "already_running", "scope": "other_process", "holder": lease.get("holder"),
                "lease_expires_at": lease.get("expires_at")}
    t = threading.Thread(target=run_saudi_ingest, args=(trigger, only), daemon=True, name="saudi-ingest")
    t.start()
    time.sleep(0.2)
    return {"status": "started", "run_id": RUN_STATE.get("run_id"), "trigger": trigger}


# --------------------------------------------------------------------------- #
# Scheduler (every SAUDI_INGEST_EVERY_HOURS, survives restarts via fetch_log)
# --------------------------------------------------------------------------- #
_SCHED_STARTED = False
_SCHED_THREAD: Optional[threading.Thread] = None
SCHED_STATE: Dict[str, Any] = {"enabled": SAUDI_AUTO_INGEST, "interval_hours": SAUDI_INGEST_EVERY_HOURS,
                               "started_at": None, "last_tick_at": None, "last_decision": None,
                               "last_run_started_at": None, "last_error": None}


def scheduler_info() -> Dict[str, Any]:
    return {**SCHED_STATE, "thread_alive": bool(_SCHED_THREAD and _SCHED_THREAD.is_alive())}


def _last_finished() -> Optional[datetime]:
    try:
        con = get_db()
        row = con.execute("SELECT MAX(finished_at) AS m FROM fetch_log").fetchone()
        con.close()
        if row and row["m"]:
            return datetime.fromisoformat(row["m"])
    except Exception:
        pass
    return None


def _scheduler_loop() -> None:
    time.sleep(20)
    interval = timedelta(hours=max(0.25, SAUDI_INGEST_EVERY_HOURS))
    while True:
        try:
            last = _last_finished()
            due = last is None or datetime.now(timezone.utc) - last >= interval
            SCHED_STATE.update(last_tick_at=now_iso(), last_decision="due" if due else "not_due", last_error=None)
            if due:
                SCHED_STATE["last_run_started_at"] = now_iso()
                run_saudi_ingest("scheduled")
        except Exception as exc:
            SCHED_STATE["last_error"] = f"{type(exc).__name__}: {exc}"
            log.exception("scheduled Saudi ingest failed")
        time.sleep(300)


def start_scheduler() -> bool:
    """Start the periodic ingestion thread once (no-op if disabled or already running)."""
    global _SCHED_STARTED, _SCHED_THREAD
    if not SAUDI_AUTO_INGEST or _SCHED_STARTED:
        return False
    _SCHED_STARTED = True
    _SCHED_THREAD = threading.Thread(target=_scheduler_loop, daemon=True, name="saudi-scheduler")
    _SCHED_THREAD.start()
    SCHED_STATE["started_at"] = now_iso()
    log.info("Saudi fraud ingest scheduler started (every %sh)", SAUDI_INGEST_EVERY_HOURS)
    return True


def reclassify_stored() -> Dict[str, Any]:
    """Re-evaluate every article stored by this pipeline with the current engine.

    Articles are never deleted. If an article no longer qualifies, only its derived
    ``threat_intel`` row and fraud ``category`` are removed and the reason is logged.
    """
    by_code = {s.code: s for s in SAUDI_SOURCES}
    con = get_db()
    rows = con.execute("SELECT * FROM articles WHERE source_code IS NOT NULL AND source_code != ''").fetchall()
    con.close()
    kept: List[int] = []
    demoted: List[Dict[str, Any]] = []
    for r in rows:
        src = by_code.get(r["source_code"])
        domain = r["domain"] or extract_domain(r["url"] or "")
        meta = {
            "domain": domain,
            "saudi_publisher": is_saudi_outlet(domain) or bool(src and is_saudi_outlet(src.domain)),
            "saudi_scope": bool(src and src.saudi_scope and src.kind != "gnews"),
            "official": bool(src and src.official) or domain.endswith(".gov.sa"),
        }
        analysis, reason = fraud_intel.analyze(r["headline"] or "", r["article_summary"] or "", r["content"] or "",
                                               source=meta, require_saudi=True)
        con = get_db()
        try:
            if analysis is None:
                con.execute("DELETE FROM threat_intel WHERE article_id = ?", (r["id"],))
                con.execute("UPDATE articles SET category = NULL WHERE id = ?", (r["id"],))
                con.execute("UPDATE ingest_seen SET decision='excluded', reason=? WHERE article_id = ?",
                            (f"reclassified:{reason}", r["id"]))
                con.commit()
                demoted.append({"article_id": r["id"], "title": r["headline"], "reason": reason})
                continue
            con.commit()
        finally:
            con.close()
        art = {"article_id": r["id"], "headline": r["headline"], "title": r["headline"],
               "content": r["content"], "article_summary": r["article_summary"], "url": r["url"],
               "source": r["source"], "domain": domain, "country": "", "published_at": r["published_at"],
               "saudi_scope": meta["saudi_scope"], "saudi_publisher": meta["saudi_publisher"],
               "official": meta["official"]}
        tid, _why = analyze_and_store(art, require_saudi=True)
        if tid:
            kept.append(tid)
    return {"checked": len(rows), "kept": len(kept), "demoted": len(demoted), "demoted_items": demoted}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the Saudi Fraud Intelligence ingestion")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reclassify", action="store_true", help="re-evaluate stored articles with the current engine")
    ap.add_argument("--only", default="", help="comma separated source codes")
    args = ap.parse_args()
    if args.reclassify:
        print(json.dumps(reclassify_stored(), ensure_ascii=False, indent=2))
        return
    if not args.run:
        ap.print_help()
        return
    only = [x.strip() for x in args.only.split(",") if x.strip()] or None
    print(json.dumps(run_saudi_ingest("cli", only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
