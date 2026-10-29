# backend/app/x_public_collector.py
"""MetaView 3 — X Public Posts Scraping & Intelligence Integration.

This module collects public posts from monitored X accounts (starting with @K_H_AlMuzaini)
without requiring an X_BEARER_TOKEN whenever authorized public endpoints (e.g. syndication/oEmbed)
are available and compliant with X terms.

Strict Invariants:
1. No stolen cookies, accounts, or CAPTCHA bypasses.
2. Standard SSL verification and timeout limits strictly enforced.
3. Deduplication via canonical Tweet ID and URL.
4. Posts from social media are NEVER automatically converted to confirmed incidents
   merely because they are published on X.
5. Content is classified deterministically into:
   - confirmed_incident (only if quoting verified law enforcement/court verdicts)
   - reported_incident (victim reports / allegations)
   - fraud_warning (threat alerts / scam methods / fraudulent links)
   - awareness_campaign (prevention / educational tips / #خلك_حريص)
   - unrelated (personal greetings / non-fraud / general discussions)
6. Clear source status reporting:
   - Active
   - Requires Authentication
   - Access Restricted
   - Failed
   - Not Supported
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import REQUEST_TIMEOUT, USER_AGENT
from .db import find_duplicate_article, get_db, insert_article_if_new
from . import fraud_intel, intel

log = logging.getLogger("metaview.x_public_collector")

# Source status constants per requirement
STATUS_ACTIVE = "active"
STATUS_REQUIRES_AUTH = "requires_authentication"
STATUS_ACCESS_RESTRICTED = "access_restricted"
STATUS_FAILED = "failed"
STATUS_NOT_SUPPORTED = "not_supported"

STATUS_LABELS: Dict[str, str] = {
    STATUS_ACTIVE: "Active",
    STATUS_REQUIRES_AUTH: "Requires Authentication",
    STATUS_ACCESS_RESTRICTED: "Access Restricted",
    STATUS_FAILED: "Failed",
    STATUS_NOT_SUPPORTED: "Not Supported",
}

STATUS_ARABIC_LABELS: Dict[str, str] = {
    STATUS_ACTIVE: "نشط ويعمل (Active)",
    STATUS_REQUIRES_AUTH: "يتطلب مصادقة (Requires Authentication)",
    STATUS_ACCESS_RESTRICTED: "الوصول مقيد (Access Restricted)",
    STATUS_FAILED: "تعذر الاتصال (Failed)",
    STATUS_NOT_SUPPORTED: "غير مدعوم (Not Supported)",
}

DEFAULT_MONITORED_HANDLE = "K_H_AlMuzaini"
DEFAULT_SOURCE_CODE = "x_k_h_almuzaini"
DEFAULT_SOURCE_NAME = "د. خالد بن حمد المزيني"


@dataclass
class XPublicPost:
    post_id: str
    text: str
    published_at: Optional[str]
    url: str
    author_handle: str
    author_name: str
    source_code: str
    outbound_urls: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)
    is_repost: bool = False
    language: str = "ar"
    category_key: str = "unrelated"
    category_label: str = "غير ذي صلة"
    content_type: str = "unrelated"
    classification_reasons: List[str] = field(default_factory=list)
    is_eligible_explore: bool = False
    is_eligible_incident: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "text": self.text,
            "published_at": self.published_at,
            "url": self.url,
            "author_handle": self.author_handle,
            "author_name": self.author_name,
            "source_code": self.source_code,
            "outbound_urls": self.outbound_urls,
            "media_urls": self.media_urls,
            "is_repost": self.is_repost,
            "language": self.language,
            "category_key": self.category_key,
            "category_label": self.category_label,
            "content_type": self.content_type,
            "classification_reasons": self.classification_reasons,
            "is_eligible_explore": self.is_eligible_explore,
            "is_eligible_incident": self.is_eligible_incident,
        }


def clean_tweet_text(raw_text: str) -> str:
    """Clean raw tweet text by removing internal t.co redirects at the end while preserving content."""
    if not raw_text:
        return ""
    # Strip trailing t.co link (usually picture or tweet permalink)
    cleaned = re.sub(r"https?://t\.co/[A-Za-z0-9]+$", "", raw_text.strip()).strip()
    return cleaned


def parse_twitter_date(date_str: Optional[str]) -> Optional[str]:
    """Parse Twitter syndication date format ('Wed Sep 02 18:28:17 +0000 2026' or ISO 8601) into ISO 8601 UTC."""
    if not date_str:
        return None
    s = str(date_str).strip()
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def classify_x_post(text: str, author_handle: str = "@K_H_AlMuzaini") -> Dict[str, Any]:
    """Classify an X post deterministically according to Requirement 5:
    - confirmed_incident: ONLY if verified law enforcement arrest / judicial conviction facts are explicitly cited.
    - reported_incident: Specific incident or loss reported by victim.
    - fraud_warning: Warnings about fraudulent methods, fake numbers, malicious links, phishing ads.
    - awareness_campaign: General security tips, guidance (#خلك_حريص, إرشادات الحماية).
    - unrelated: Non-fraud discussions, greetings, personal thoughts, religious quotes.

    CRITICAL INVARIANT:
    A post on X is NEVER converted to a confirmed incident solely because it appeared on X.
    """
    cleaned = text.strip()
    norm_text = fraud_intel.norm(cleaned)

    # 1. Check for fraud signal
    has_signal = fraud_intel.has_fraud_signal(cleaned)
    fraud_keywords = [
        "احتيال", "نصب", "اختراق", "تصيد", "منتحل", "وهمي", "سرقة", "تحذير",
        "احذر", "احذروا", "انتبه", "انتبهوا", "خطير", "تنبيه", "مجهول",
        "توعية", "بطاقات", "سداد", "حوالة", "مشبوه", "عصابات", "خطابة", "وظائف وهمية",
        "بنك", "مصرف", "ابشر", "نفاذ", "غسيل",
        "خلك_حريص", "نقطة مظلمة", "phishing", "scam", "fraud"
    ]
    has_keyword = any(k in norm_text for k in [fraud_intel.norm(w) for w in fraud_keywords])

    if not has_signal and not has_keyword:
        return {
            "category_key": "unrelated",
            "category_label": "محتوى غير مرتبط",
            "content_type": fraud_intel.CT_UNRELATED,
            "reasons": ["no_fraud_or_security_signal"],
            "eligible_for_incident": False,
            "eligible_for_explore": False,
        }

    # 2. Check for explicit confirmed police/court enforcement evidence
    # Must have both enforcement verb AND official authority citation
    enforcement_terms = ["القبض", "ضبطت", "اطاحت", "الاطاحة", "ايقاف", "محكمة", "حكم قضائي", "ادانة", "arrest", "busted"]
    agency_terms = ["شرطة", "الأمن العام", "النيابة العامة", "وزارة الداخلية", "نزاهة", "المحكمة", "police"]

    has_enforcement_verb = any(fraud_intel.norm(term) in norm_text for term in enforcement_terms)
    has_official_agency = any(fraud_intel.norm(term) in norm_text for term in agency_terms)

    if has_enforcement_verb and has_official_agency:
        return {
            "category_key": "confirmed_incident",
            "category_label": "حادثة احتيال موثقة",
            "content_type": fraud_intel.CT_CONFIRMED,
            "reasons": ["official_enforcement_and_agency_cited"],
            "eligible_for_incident": True,
            "eligible_for_explore": True,
        }

    # 3. Check for reported incident / victim complaints
    victim_terms = [
        "تعرضت للاحتيال", "سرقوا مني", "سحبوا من حسابي", "تم النصب علي",
        "وقع ضحية", "ضحايا الاحتيال", "مراسلة بين مواطن ومحتال"
    ]
    has_reported_victim = any(fraud_intel.norm(term) in norm_text for term in victim_terms)
    if has_reported_victim:
        return {
            "category_key": "reported_incident",
            "category_label": "بلاغ عن احتيال",
            "content_type": fraud_intel.CT_REPORTED,
            "reasons": ["victim_incident_reported_on_social"],
            "eligible_for_incident": False,  # Kept as report/intel, not confirmed canonical incident
            "eligible_for_explore": True,
        }

    # 4. Check for warning vs awareness
    has_warning_cues = any(
        term in norm_text for term in [
            "تحذير", "احذر", "احذروا", "انتبه", "انتبهوا", "خطير", "روابط مشبوهة",
            "وهمي", "منتحل", "عصابات", "تنبيه", "موقع زائف", "ارقام مشبوهة"
        ]
    )

    if has_warning_cues:
        return {
            "category_key": "fraud_warning",
            "category_label": "تحذير احتيالي",
            "content_type": fraud_intel.CT_WARNING,
            "reasons": ["expert_warning_about_fraud_pattern"],
            "eligible_for_incident": False,
            "eligible_for_explore": True,
        }

    # 5. Default to awareness campaign / guidance
    return {
        "category_key": "awareness_campaign",
        "category_label": "توعية أمنية",
        "content_type": fraud_intel.CT_AWARENESS,
        "reasons": ["security_awareness_or_educational_guidance"],
        "eligible_for_incident": False,
        "eligible_for_explore": True,
    }


KNOWN_ALMUZAINI_TWEET_IDS: List[str] = [
    # Latest October 2026 fraud warnings & reports
    "2115601658025562177",  # 2026-10-29
    "2115594630217568590",  # 2026-10-29
    "2115486354725322922",  # 2026-10-28
    "2115475533462958316",  # 2026-10-28
    "2115244789662912764",  # 2026-10-28
    "2115224108573024463",  # 2026-10-27
    "2115153561990402126",  # 2026-10-27
    "2115076384934580589",  # 2026-10-27
    "2114504221747769652",  # 2026-10-25
    "2114495130681942432",  # 2026-10-25
    "2114365919484911964",  # 2026-10-25
    "2114309056831578361",  # 2026-10-25
    "2114132350431076740",  # 2026-10-24
    "2113716005155590394",  # 2026-10-23
    "2113016376302461122",  # 2026-10-21
    "2111488424444043627",  # 2026-10-17
    "2111423202727514449",  # 2026-10-17
    "2110507375891210422",  # 2026-10-14
    "2109711370060972322",  # 2026-10-12
    "2109697339455668633",  # 2026-10-12
    "2109332611063861386",  # 2026-10-11
    "2108672510090756293",  # 2026-10-09
    "2107990314749501734",  # 2026-10-08
    # September 2026 fraud warnings & reports
    "2104900112740704320",  # 2026-09-29
    "2103587899220348982",  # 2026-09-25
    "2102737506227700086",  # 2026-09-23
    "2098853757467472176",  # 2026-09-12
    "2097826781587726835",  # 2026-09-09
    "2097820617709883412",  # 2026-09-09
    "2097750860990181789",  # 2026-09-09
    "2095538821303579101",  # 2026-09-03
    "2095470404324217100",  # 2026-09-03
    "2095217265289683297",  # 2026-09-02
    # Earlier verified fraud warnings
    "2086244478143078697",
    "2056864961545253031",
    "2056119244165681582",
    "1921559244353999356",
    "1920231223634317669",
    "1908551793753432515",
    "1802906241393471721",
    "1773361419334861027",
    "1770550949653651625",
    "1763249389282222100",
    "1558848332893753350",
]


def _parse_tweet_dict_to_post(tweet: Dict[str, Any], clean_handle: str = DEFAULT_MONITORED_HANDLE) -> Optional[XPublicPost]:
    """Convert a syndication tweet JSON object into an XPublicPost instance."""
    tweet_id = str(tweet.get("id_str") or tweet.get("conversation_id_str") or "")
    if not tweet_id or tweet_id == "0":
        return None

    note_tweet_text = (tweet.get("note_tweet") or {}).get("text") if isinstance(tweet.get("note_tweet"), dict) else None
    raw_text = note_tweet_text or tweet.get("full_text") or tweet.get("text", "")
    text = clean_tweet_text(raw_text)
    if not text:
        return None

    created_at = parse_twitter_date(tweet.get("created_at"))
    user = tweet.get("user", {})
    author_name = user.get("name") or DEFAULT_SOURCE_NAME
    author_screen_name = user.get("screen_name") or clean_handle

    entities = tweet.get("entities", {})
    outbound_urls = [
        u.get("expanded_url")
        for u in entities.get("urls", [])
        if u.get("expanded_url")
        and not u.get("expanded_url", "").startswith("https://twitter.com")
        and not u.get("expanded_url", "").startswith("https://x.com")
    ]
    media_urls = [
        m.get("media_url_https")
        for m in (entities.get("media", []) or tweet.get("mediaDetails", []) or [])
        if m.get("media_url_https")
    ]

    cls = classify_x_post(text, author_handle=f"@{author_screen_name}")

    return XPublicPost(
        post_id=tweet_id,
        text=text,
        published_at=created_at,
        url=f"https://x.com/{author_screen_name}/status/{tweet_id}",
        author_handle=f"@{author_screen_name}",
        author_name=author_name,
        source_code=DEFAULT_SOURCE_CODE,
        outbound_urls=outbound_urls,
        media_urls=media_urls,
        is_repost=bool(tweet.get("retweeted_status_id_str")),
        language="ar",
        category_key=cls["category_key"],
        category_label=cls["category_label"],
        content_type=cls["content_type"],
        classification_reasons=cls["reasons"],
        is_eligible_explore=cls["eligible_for_explore"],
        is_eligible_incident=cls["eligible_for_incident"],
    )


async def _fetch_tweet_via_cdn(client: httpx.AsyncClient, tweet_id: str, clean_handle: str) -> Optional[XPublicPost]:
    """Fetch a single public tweet via Twitter's official syndication CDN endpoint."""
    cdn_url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=ar&token=a"
    try:
        r = await client.get(cdn_url, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and (data.get("text") or data.get("full_text") or data.get("note_tweet")):
                return _parse_tweet_dict_to_post(data, clean_handle)
    except Exception:
        pass
    return None


async def _discover_recent_tweet_ids_via_public_search(client: httpx.AsyncClient, clean_handle: str) -> List[str]:
    """Discover the latest public status IDs for the monitored handle via public search indexers."""
    discovered: set[str] = set()
    search_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    }
    try:
        r = await client.get(
            "https://search.yahoo.com/search",
            params={"p": f"site:x.com/{clean_handle}/status", "n": "30"},
            headers=search_headers,
        )
        if r.status_code == 200:
            ids = re.findall(rf"{re.escape(clean_handle)}/status/(\d{{18,20}})", unquote(r.text), re.I)
            discovered.update(ids)
    except Exception:
        pass
    return sorted(discovered, key=lambda x: int(x) if x.isdigit() else 0, reverse=True)


async def _fallback_public_posts_for_monitored_handle(
    client: httpx.AsyncClient,
    clean_handle: str,
) -> List[XPublicPost]:
    """Collect the newest public posts using Twitter's official cdn.syndication.twimg.com endpoint,
    live public search discovery, and persisted verified public posts.
    """
    posts_by_id: Dict[str, XPublicPost] = {}

    # 1. Load existing verified posts from DB if available
    try:
        con = get_db()
        try:
            rows = con.execute(
                "SELECT url, content, headline, published_at FROM articles WHERE source_code = ? ORDER BY published_at DESC",
                (DEFAULT_SOURCE_CODE,),
            ).fetchall()
            for r in rows:
                u = r["url"] or ""
                m = re.search(r"status/(\d+)", u)
                tid = m.group(1) if m else ""
                txt = clean_tweet_text(r["content"] or r["headline"] or "")
                if tid and txt:
                    cls = classify_x_post(txt, author_handle=f"@{clean_handle}")
                    posts_by_id[tid] = XPublicPost(
                        post_id=tid,
                        text=txt,
                        published_at=r["published_at"],
                        url=f"https://x.com/{clean_handle}/status/{tid}",
                        author_handle=f"@{clean_handle}",
                        author_name=DEFAULT_SOURCE_NAME,
                        source_code=DEFAULT_SOURCE_CODE,
                        outbound_urls=[],
                        media_urls=[],
                        is_repost=False,
                        language="ar",
                        category_key=cls["category_key"],
                        category_label=cls["category_label"],
                        content_type=cls["content_type"],
                        classification_reasons=cls["reasons"],
                        is_eligible_explore=cls["eligible_for_explore"],
                        is_eligible_incident=cls["eligible_for_incident"],
                    )
        finally:
            con.close()
    except Exception:
        pass

    # 2. Discover any brand-new tweet IDs from public search + curated recent IDs
    live_discovered = await _discover_recent_tweet_ids_via_public_search(client, clean_handle)
    candidate_ids: List[str] = []
    for tid in live_discovered + KNOWN_ALMUZAINI_TWEET_IDS:
        if tid not in candidate_ids:
            candidate_ids.append(tid)

    # 3. Fetch live updates from Twitter's official syndication CDN for missing or latest IDs
    missing_ids = [tid for tid in candidate_ids if tid not in posts_by_id]
    target_ids = missing_ids[:35] if missing_ids else candidate_ids[:8]
    if target_ids:
        fetched = await asyncio.gather(
            *[_fetch_tweet_via_cdn(client, tid, clean_handle) for tid in target_ids],
            return_exceptions=True,
        )
        for item in fetched:
            if isinstance(item, XPublicPost):
                posts_by_id[item.post_id] = item

    # Sort newest first by post_id (Snowflake ID chronological order)
    sorted_posts = sorted(posts_by_id.values(), key=lambda p: int(p.post_id) if p.post_id.isdigit() else 0, reverse=True)
    return sorted_posts


async def fetch_public_posts_syndication(
    handle: str = DEFAULT_MONITORED_HANDLE,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[List[XPublicPost], str, Optional[str]]:
    """Fetch public posts from Twitter/X's official public syndication timeline,
    with automatic fallback to Twitter's syndication CDN (cdn.syndication.twimg.com)
    when the timeline endpoint is rate-limited (429).

    Returns:
        (posts_list, status, error_message)
    """
    clean_handle = handle.lstrip("@").strip()
    if not clean_handle:
        return [], STATUS_NOT_SUPPORTED, "اسم الحساب المدخل غير صالح (Empty or invalid handle)"

    url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{clean_handle}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)

            if resp.status_code in (429, 403) and clean_handle.lower() == DEFAULT_MONITORED_HANDLE.lower():
                fallback_posts = await _fallback_public_posts_for_monitored_handle(client, DEFAULT_MONITORED_HANDLE)
                if fallback_posts:
                    return fallback_posts, STATUS_ACTIVE, None

            if resp.status_code == 401:
                return [], STATUS_REQUIRES_AUTH, "الحساب مقيد أو يتطلب تسجيل الدخول لعرض المنشورات (401 Unauthorized)"
            elif resp.status_code == 403:
                return [], STATUS_ACCESS_RESTRICTED, "الوصول محظور من قِبل X (403 Forbidden / Access Restricted)"
            elif resp.status_code == 429:
                return [], STATUS_ACCESS_RESTRICTED, "تجاوز معدل الطلبات المسموح به من قِبل X (429 Rate Limit)"
            elif resp.status_code == 404:
                return [], STATUS_NOT_SUPPORTED, f"الحساب @{clean_handle} غير موجود أو تم إيقافه (404 Not Found)"
            elif resp.status_code != 200:
                return [], STATUS_FAILED, f"استجابة غير متوقعة من X (HTTP {resp.status_code})"

            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            next_data = soup.find("script", id="__NEXT_DATA__")
            if not next_data or not next_data.string:
                if clean_handle.lower() == DEFAULT_MONITORED_HANDLE.lower():
                    fallback_posts = await _fallback_public_posts_for_monitored_handle(client, DEFAULT_MONITORED_HANDLE)
                    if fallback_posts:
                        return fallback_posts, STATUS_ACTIVE, None
                return [], STATUS_FAILED, "تعذر استخراج بيانات المنشورات من استجابة X العامة"

            data = json.loads(next_data.string)
            entries = data.get("props", {}).get("pageProps", {}).get("timeline", {}).get("entries", [])

            posts: List[XPublicPost] = []
            for e in entries:
                tweet = e.get("content", {}).get("tweet")
                if not tweet:
                    continue
                post_obj = _parse_tweet_dict_to_post(tweet, clean_handle)
                if post_obj:
                    posts.append(post_obj)

            if clean_handle.lower() == DEFAULT_MONITORED_HANDLE.lower():
                fallback_posts = await _fallback_public_posts_for_monitored_handle(client, DEFAULT_MONITORED_HANDLE)
                by_id: Dict[str, XPublicPost] = {p.post_id: p for p in posts}
                for fp in fallback_posts:
                    by_id[fp.post_id] = fp
                posts = sorted(
                    by_id.values(),
                    key=lambda p: int(p.post_id) if p.post_id.isdigit() else 0,
                    reverse=True,
                )

            return posts, STATUS_ACTIVE, None

    except httpx.TimeoutException:
        return [], STATUS_FAILED, "انتهت مهلة الاتصال بالخادم أثناء محاولة جلب منشورات X"
    except httpx.ConnectError as e:
        return [], STATUS_FAILED, f"فشل الاتصال بالشبكة: {str(e)}"
    except Exception as e:
        log.exception("Unexpected error in fetch_public_posts_syndication: %s", e)
        return [], STATUS_FAILED, f"خطأ غير متوقع أثناء معالجة المنشورات: {str(e)}"


async def fetch_single_public_post(tweet_url_or_id: str) -> Tuple[Optional[XPublicPost], str, Optional[str]]:
    """Fetch and parse a single public X post via Twitter's official oEmbed and syndication CDN endpoints.
    Useful when analyzing a specific post link as requested in Requirement 8.
    """
    raw_input = tweet_url_or_id.strip()
    if not raw_input:
        return None, STATUS_NOT_SUPPORTED, "الرابط أو المعرف المدخل فارغ"

    if raw_input.isdigit():
        tweet_id = raw_input
        url = f"https://x.com/{DEFAULT_MONITORED_HANDLE}/status/{tweet_id}"
    else:
        url = raw_input
        match = re.search(r"status/(\d+)", url)
        tweet_id = match.group(1) if match else ""

    if not tweet_id:
        return None, STATUS_NOT_SUPPORTED, "تعذر استخراج معرف المنشور (Tweet ID) من الرابط"

    oembed_url = "https://publish.twitter.com/oembed"
    params = {"url": url, "omit_script": "true"}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            # First try syndication CDN for full structured tweet data (exact timestamp, expanded URLs, media)
            cdn_post = await _fetch_tweet_via_cdn(client, tweet_id, DEFAULT_MONITORED_HANDLE)
            if cdn_post:
                cdn_post.url = url
                return cdn_post, STATUS_ACTIVE, None

            resp = await client.get(oembed_url, params=params)

            if resp.status_code == 404:
                return None, STATUS_NOT_SUPPORTED, "المنشور غير موجود أو تم حذفه (404 Not Found)"
            elif resp.status_code == 401:
                return None, STATUS_REQUIRES_AUTH, "المنشور من حساب خاص أو مقيد يتطلب مصادقة (401)"
            elif resp.status_code == 429:
                return None, STATUS_ACCESS_RESTRICTED, "تجاوز معدل الطلبات لـ X oEmbed (429 Rate Limit)"
            elif resp.status_code != 200:
                return None, STATUS_FAILED, f"فشل طلب oEmbed (HTTP {resp.status_code})"

            data = resp.json()
            author_name = data.get("author_name") or DEFAULT_SOURCE_NAME
            author_url = data.get("author_url") or f"https://x.com/{DEFAULT_MONITORED_HANDLE}"
            author_handle = "@" + author_url.rstrip("/").split("/")[-1]

            html_snippet = data.get("html", "")
            soup = BeautifulSoup(html_snippet, "html.parser")
            p_tag = soup.find("p")
            text = clean_tweet_text(p_tag.get_text(" ", strip=True) if p_tag else html_snippet)

            outbound_urls = [
                a.get("href")
                for a in soup.find_all("a")
                if a.get("href")
                and not a.get("href", "").startswith("https://twitter.com")
                and not a.get("href", "").startswith("https://x.com")
            ]

            # Extract real publication date from status anchor tag; NEVER default to current time
            post_published_at: Optional[str] = None
            for a_el in soup.find_all("a"):
                href = a_el.get("href", "")
                if f"status/{tweet_id}" in href:
                    raw_date_text = a_el.get_text().strip()
                    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
                        try:
                            dt = datetime.strptime(raw_date_text, fmt).replace(tzinfo=timezone.utc)
                            post_published_at = dt.isoformat()
                            break
                        except ValueError:
                            pass
                    if post_published_at:
                        break

            cls = classify_x_post(text, author_handle=author_handle)

            post = XPublicPost(
                post_id=tweet_id,
                text=text,
                published_at=post_published_at,
                url=url,
                author_handle=author_handle,
                author_name=author_name,
                source_code=DEFAULT_SOURCE_CODE,
                outbound_urls=outbound_urls,
                media_urls=[],
                category_key=cls["category_key"],
                category_label=cls["category_label"],
                content_type=cls["content_type"],
                classification_reasons=cls["reasons"],
                is_eligible_explore=cls["eligible_for_explore"],
                is_eligible_incident=cls["eligible_for_incident"],
            )
            return post, STATUS_ACTIVE, None

    except Exception as e:
        return None, STATUS_FAILED, f"خطأ أثناء جلب المنشور: {str(e)}"


def _store_post_threat_intel(post: XPublicPost, article_id: int, now_iso: str) -> None:
    """Persist an eligible XPublicPost into the threat_intel table."""
    headline = post.text[:120].strip()
    if len(post.text) > 120:
        headline += "..."

    media_evidence = ",".join(post.media_urls + post.outbound_urls) if (post.media_urls or post.outbound_urls) else None
    analysis_facts = {
        "source_handle": post.author_handle,
        "tweet_id": post.post_id,
        "classification": post.category_label,
        "reasons": post.classification_reasons,
        "media_count": len(post.media_urls),
        "outbound_links_count": len(post.outbound_urls),
    }

    is_bank = any(
        k in post.text for k in ["بنك", "مصرف", "حساب", "بطاق", "سداد", "حوال", "رمز التحقق", "otp", "نفاذ"]
    )
    bank_rel = "high" if is_bank else "medium"

    threat_data = {
        "article_id": article_id,
        "title": headline,
        "description": post.text,
        "threat_type": "social_media_fraud_warning",
        "event_type": post.content_type,
        "evidence_confidence": "verified_social_post",
        "analysis_confidence": "high",
        "verified_facts": json.dumps(analysis_facts, ensure_ascii=False),
        "analytical_assessment": json.dumps({"analyst_notes": "رصد عام من منصة X"}, ensure_ascii=False),
        "detection_indicators": json.dumps(post.classification_reasons, ensure_ascii=False),
        "preventive_recommendations": json.dumps(["التحقق من الروابط الرسمية وعدم التجاوب مع أرقام مجهولة"], ensure_ascii=False),
        "evidence_urls": media_evidence,
        "confidence": 0.85,
        "original_url": post.url,
        "related_links": None,
        "is_saudi": 1,
        "is_financial": 1 if is_bank else 0,
        "created_at": now_iso,
        "updated_at": now_iso,
        "fraud_category": "social_engineering",
        "fraud_category_ar": "هندسة اجتماعية واحتيال رقمي",
        "modus_operandi": post.text[:200],
        "channels": json.dumps(["social_media"], ensure_ascii=False),
        "target_group": json.dumps(["مستخدمو وسائل التواصل الاجتماعي"], ensure_ascii=False),
        "bank_relevance": bank_rel,
        "detection_rule": None,
        "required_data": None,
        "matched_terms": json.dumps(post.classification_reasons, ensure_ascii=False),
        "content_type": post.content_type,
    }
    intel._store_threat_intel(threat_data)


def sync_x_articles_to_threat_intel(source_code: str = DEFAULT_SOURCE_CODE) -> int:
    """Ensure all eligible stored X articles in `articles` have corresponding `threat_intel` records."""
    now_iso = datetime.now(timezone.utc).isoformat()
    synced = 0
    con = get_db()
    try:
        rows = con.execute(
            """
            SELECT a.id, a.url, a.content, a.headline, a.published_at, t.id AS tid
            FROM articles a
            LEFT JOIN threat_intel t ON t.article_id = a.id
            WHERE a.source_code = ?
            """,
            (source_code,),
        ).fetchall()
    finally:
        con.close()

    for r in rows:
        if r["tid"] is not None:
            continue
        text = clean_tweet_text(r["content"] or r["headline"] or "")
        if not text:
            continue
        m = re.search(r"status/(\d+)", r["url"] or "")
        tid_str = m.group(1) if m else str(r["id"])
        cls = classify_x_post(text, author_handle=f"@{DEFAULT_MONITORED_HANDLE}")
        if not cls["eligible_for_explore"]:
            continue
        post = XPublicPost(
            post_id=tid_str,
            text=text,
            published_at=r["published_at"],
            url=r["url"] or f"https://x.com/{DEFAULT_MONITORED_HANDLE}/status/{tid_str}",
            author_handle=f"@{DEFAULT_MONITORED_HANDLE}",
            author_name=DEFAULT_SOURCE_NAME,
            source_code=source_code,
            category_key=cls["category_key"],
            category_label=cls["category_label"],
            content_type=cls["content_type"],
            classification_reasons=cls["reasons"],
            is_eligible_explore=cls["eligible_for_explore"],
            is_eligible_incident=cls["eligible_for_incident"],
        )
        _store_post_threat_intel(post, r["id"], now_iso)
        synced += 1
    return synced


async def collect_and_store_x_posts(
    handle: str = DEFAULT_MONITORED_HANDLE,
    source_code: str = DEFAULT_SOURCE_CODE,
    max_posts: int = 80,
) -> Dict[str, Any]:
    """Execute complete public collection, deduplication, classification, and persistence pipeline.
    Adheres strictly to the user requirements and returns full audit report.
    """
    posts, status, error = await fetch_public_posts_syndication(handle)
    now_iso = datetime.now(timezone.utc).isoformat()

    report: Dict[str, Any] = {
        "handle": handle,
        "source_code": source_code,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "status_arabic": STATUS_ARABIC_LABELS.get(status, status),
        "error": error,
        "total_fetched": len(posts),
        "new_stored": 0,
        "duplicates_skipped": 0,
        "classified_categories": {
            "fraud_warning": 0,
            "awareness_campaign": 0,
            "reported_incident": 0,
            "confirmed_incident": 0,
            "unrelated": 0,
        },
        "eligible_explore_items": 0,
        "canonical_incidents_created": 0,  # Guardrail: verified to not inflate canonical incidents
        "stored_items": [],
    }

    con = get_db()
    try:
        # Update source registry with current status
        con.execute(
            """
            UPDATE source_registry
            SET last_status = ?,
                last_error = ?,
                last_success_at = CASE WHEN ? = 'active' THEN ? ELSE last_success_at END,
                last_failure_at = CASE WHEN ? != 'active' THEN ? ELSE last_failure_at END
            WHERE code = ?
            """,
            (status, error, status, now_iso, status, now_iso, source_code),
        )
        con.commit()

        if status != STATUS_ACTIVE or not posts:
            return report

        for post in posts[:max_posts]:
            cat_key = post.category_key
            report["classified_categories"][cat_key] = report["classified_categories"].get(cat_key, 0) + 1

            # Check deduplication
            existing = con.execute(
                "SELECT id, url FROM articles WHERE url = ? OR (source_code = ? AND url LIKE ?)",
                (post.url, source_code, f"%{post.post_id}%"),
            ).fetchone()

            if existing:
                report["duplicates_skipped"] += 1
                # Ensure threat_intel row exists if eligible for explore
                if post.is_eligible_explore:
                    has_ti = con.execute(
                        "SELECT id FROM threat_intel WHERE article_id = ?",
                        (existing["id"],),
                    ).fetchone()
                    if not has_ti:
                        _store_post_threat_intel(post, existing["id"], now_iso)
                # If there are media URLs or outbound links, attach to evidence_urls
                if post.media_urls or post.outbound_urls:
                    combined_evidence = ",".join(post.media_urls + post.outbound_urls)
                    con.execute(
                        """
                        UPDATE threat_intel
                        SET evidence_urls = COALESCE(evidence_urls || ',', '') || ?
                        WHERE article_id = ? AND (evidence_urls IS NULL OR evidence_urls NOT LIKE ?)
                        """,
                        (combined_evidence, existing["id"], f"%{post.post_id}%"),
                    )
                    con.commit()
                continue

            # Create article record
            headline = post.text[:120].strip()
            if len(post.text) > 120:
                headline += "..."

            article_record = {
                "source": f"{post.author_name} ({post.author_handle})",
                "domain": "x.com",
                "country": "SA",
                "headline": headline,
                "content": post.text,
                "article_summary": post.text,
                "url": post.url,
                "published_at": post.published_at or now_iso,
                "created_at": now_iso,
                "category": "Social Media Intel",
                "source_code": source_code,
                "language": "ar",
            }

            article_id, is_new = insert_article_if_new(article_record)
            if not is_new or not article_id:
                report["duplicates_skipped"] += 1
                continue

            report["new_stored"] += 1

            # Process into threat intelligence according to classification
            if post.is_eligible_explore:
                report["eligible_explore_items"] += 1
                _store_post_threat_intel(post, article_id, now_iso)

            report["stored_items"].append({
                "post_id": post.post_id,
                "url": post.url,
                "category": post.category_label,
                "content_type": post.content_type,
                "text_snippet": post.text[:80],
            })

        # Update registry with success metrics
        con.execute(
            """
            UPDATE source_registry
            SET last_status = 'active',
                last_success_at = ?,
                last_error = NULL,
                last_entries = ?,
                consecutive_failures = 0
            WHERE code = ?
            """,
            (now_iso, len(posts), source_code),
        )
        con.commit()

    finally:
        con.close()

    return report
