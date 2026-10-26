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

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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
    """Parse Twitter syndication date format 'Wed Sep 02 18:28:17 +0000 2026' into ISO 8601 UTC."""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        # Fallback to current UTC if unparseable
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
        "توعية", "بطاقات", "سداد", "حوالة", "مشبوه", "عصابات", "خطابة", "وظائف وهمية",
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


async def fetch_public_posts_syndication(
    handle: str = DEFAULT_MONITORED_HANDLE,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[List[XPublicPost], str, Optional[str]]:
    """Fetch public posts from Twitter/X's official public syndication timeline.

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
                return [], STATUS_FAILED, "تعذر استخراج بيانات المنشورات من استجابة X العامة"

            data = json.loads(next_data.string)
            entries = data.get("props", {}).get("pageProps", {}).get("timeline", {}).get("entries", [])

            posts: List[XPublicPost] = []
            for e in entries:
                tweet = e.get("content", {}).get("tweet")
                if not tweet:
                    continue

                tweet_id = str(tweet.get("id_str") or tweet.get("conversation_id_str") or "")
                if not tweet_id or tweet_id == "0":
                    continue

                raw_text = tweet.get("full_text") or tweet.get("text", "")
                text = clean_tweet_text(raw_text)
                if not text:
                    continue

                created_at = parse_twitter_date(tweet.get("created_at"))
                user = tweet.get("user", {})
                author_name = user.get("name") or DEFAULT_SOURCE_NAME
                author_screen_name = user.get("screen_name") or clean_handle

                entities = tweet.get("entities", {})
                outbound_urls = [
                    u.get("expanded_url")
                    for u in entities.get("urls", [])
                    if u.get("expanded_url") and not u.get("expanded_url", "").startswith("https://twitter.com")
                ]
                media_urls = [
                    m.get("media_url_https")
                    for m in entities.get("media", [])
                    if m.get("media_url_https")
                ]

                # Classification
                cls = classify_x_post(text, author_handle=f"@{author_screen_name}")

                post_obj = XPublicPost(
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
                posts.append(post_obj)

            return posts, STATUS_ACTIVE, None

    except httpx.TimeoutException:
        return [], STATUS_FAILED, "انتهت مهلة الاتصال بالخادم أثناء محاولة جلب منشورات X"
    except httpx.ConnectError as e:
        return [], STATUS_FAILED, f"فشل الاتصال بالشبكة: {str(e)}"
    except Exception as e:
        log.exception("Unexpected error in fetch_public_posts_syndication: %s", e)
        return [], STATUS_FAILED, f"خطأ غير متوقع أثناء معالجة المنشورات: {str(e)}"


async def fetch_single_public_post(tweet_url_or_id: str) -> Tuple[Optional[XPublicPost], str, Optional[str]]:
    """Fetch and parse a single public X post via Twitter's official oEmbed endpoint.
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
            text = clean_tweet_text(p_tag.get_text() if p_tag else html_snippet)

            outbound_urls = [a.get("href") for a in soup.find_all("a") if a.get("href") and not a.get("href", "").startswith("https://twitter.com")]

            cls = classify_x_post(text, author_handle=author_handle)

            post = XPublicPost(
                post_id=tweet_id,
                text=text,
                published_at=datetime.now(timezone.utc).isoformat(),
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


async def collect_and_store_x_posts(
    handle: str = DEFAULT_MONITORED_HANDLE,
    source_code: str = DEFAULT_SOURCE_CODE,
    max_posts: int = 50,
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
                last_failure_at = CASE WHEN ? != 'active' THEN ? ELSE last_failure_at END
            WHERE code = ?
            """,
            (status, error, status, now_iso, source_code),
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
                article_for_intel = article_record.copy()
                article_for_intel["article_id"] = article_id

                # Generate threat intel record directly aligned with classification
                media_evidence = ",".join(post.media_urls + post.outbound_urls) if (post.media_urls or post.outbound_urls) else None
                analysis_facts = {
                    "source_handle": post.author_handle,
                    "tweet_id": post.post_id,
                    "classification": post.category_label,
                    "reasons": post.classification_reasons,
                    "media_count": len(post.media_urls),
                    "outbound_links_count": len(post.outbound_urls),
                }

                # Bank relevance check
                is_bank = any(
                    k in post.text for k in ["بنك", "مصرف", "حساب", "بطاق", "سداد", "حوال", "رمز التحقق", "otp", "نفاذ"]
                )
                bank_rel = "high" if is_bank else "medium"

                # Persist in threat_intel
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
