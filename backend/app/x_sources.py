# backend/app/x_sources.py
"""X (formerly Twitter) intelligence sources management for MetaView 3.

Handles:
- Registration and metadata for tracked X accounts.
- API authentication check (X API v2 Bearer Token).
- Legitimate feed / article fetching for linked publications.
- Classification into the 6 intelligence categories.
- Deduplication against existing articles and incidents.
- Safe fallback to 'requires_integration' status when API access is unconfigured,
  avoiding unauthorized scraping or fabricated data.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import feedparser

from .config import USER_AGENT, REQUEST_TIMEOUT
from .db import get_db, insert_article_if_new, find_duplicate_article
from . import fraud_intel, intel, incidents


@dataclass(frozen=True)
class XSourceDefinition:
    code: str
    name: str
    handle: str
    url: str
    kind: str = "x_social"              # X / Social Media
    country: str = "Global"             # SA | Global
    lang: str = "ar"
    official: bool = False              # Dr. Al-Muzaini is specialist, not official
    publisher_feed: Optional[str] = None  # Legitimate publisher RSS/news feed if available
    notes: str = ""


# The two tracked X accounts
X_SOURCES: List[XSourceDefinition] = [
    XSourceDefinition(
        code="x_k_h_almuzaini",
        name="د. خالد بن حمد المزيني",
        handle="@K_H_AlMuzaini",
        url="https://x.com/K_H_AlMuzaini",
        kind="x_social",
        country="SA",
        lang="ar",
        official=False,
        publisher_feed=None,
        notes="خبير واستشاري في أمن المعلومات ومكافحة الاحتيال المالي. رصد تحليلي شخصي غير رسمي، لا يُفترض أن كافة منشوراته احتيال.",
    ),
    XSourceDefinition(
        code="x_thehackersnews",
        name="The Hacker News",
        handle="@thehackersnews",
        url="https://x.com/thehackersnews",
        kind="x_social",
        country="Global",
        lang="en",
        official=False,
        publisher_feed="https://feeds.feedburner.com/TheHackersNews",
        notes="منصة إخبارية عالمية رائدة في الأمن السيبراني والتهديدات الرقمية تنشر روابط مقالاتها الأصلية على X.",
    ),
]

X_SOURCES_BY_CODE: Dict[str, XSourceDefinition] = {s.code: s for s in X_SOURCES}


def get_x_bearer_token() -> Optional[str]:
    """Retrieve X API v2 Bearer Token from environment if configured."""
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    return token if token else None


def register_x_sources_in_db() -> None:
    """Ensure X sources are properly registered in source_registry table."""
    con = get_db()
    cur = con.cursor()
    has_token = bool(get_x_bearer_token())

    for src in X_SOURCES:
        # Check existing row
        cur.execute("SELECT code, last_status, last_success_at FROM source_registry WHERE code = ?", (src.code,))
        row = cur.fetchone()

        initial_status = "requires_integration" if (not has_token and not src.publisher_feed) else "ok"
        initial_error = (
            "يتطلب تكامل X API (Requires Integration - X_BEARER_TOKEN) للوصول المصرح إلى المنشورات."
            if (not has_token and not src.publisher_feed)
            else None
        )

        if not row:
            cur.execute(
                """
                INSERT INTO source_registry (
                    code, name, url, kind, country, lang, official, enabled,
                    last_status, last_error, handle
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    src.code,
                    src.name,
                    src.url,
                    src.kind,
                    src.country,
                    src.lang,
                    1 if src.official else 0,
                    initial_status,
                    initial_error,
                    src.handle,
                ),
            )
        else:
            # Update metadata while preserving last_success_at and history
            cur.execute(
                """
                UPDATE source_registry
                SET name = ?, url = ?, kind = ?, country = ?, lang = ?, official = ?, handle = ?
                WHERE code = ?
                """,
                (
                    src.name,
                    src.url,
                    src.kind,
                    src.country,
                    src.lang,
                    1 if src.official else 0,
                    src.handle,
                    src.code,
                ),
            )

    con.commit()
    con.close()


async def check_source_connectivity(src: XSourceDefinition) -> Dict[str, Any]:
    """Check connectivity to the source legitimately.
    Returns status, http_status, error message, and ingestion channel.
    """
    token = get_x_bearer_token()

    if token:
        # Check X API v2 connectivity
        headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
        clean_handle = src.handle.lstrip("@")
        api_url = f"https://api.twitter.com/2/users/by/username/{clean_handle}"
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.get(api_url, headers=headers)
                if resp.status_code == 200:
                    return {
                        "status": "ok",
                        "http_status": 200,
                        "mode": "x_api_v2",
                        "error": None,
                    }
                else:
                    return {
                        "status": "failed",
                        "http_status": resp.status_code,
                        "mode": "x_api_v2",
                        "error": f"X API responded with HTTP {resp.status_code}: {resp.text[:200]}",
                    }
        except Exception as e:
            return {
                "status": "failed",
                "http_status": None,
                "mode": "x_api_v2",
                "error": f"Connection to X API failed: {str(e)}",
            }

    # If no token, check if we have a legitimate publisher feed
    if src.publisher_feed:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(src.publisher_feed)
                if resp.status_code == 200:
                    return {
                        "status": "ok",
                        "http_status": 200,
                        "mode": "publisher_feed",
                        "error": None,
                    }
                else:
                    return {
                        "status": "failed",
                        "http_status": resp.status_code,
                        "mode": "publisher_feed",
                        "error": f"Publisher feed HTTP {resp.status_code}",
                    }
        except Exception as e:
            return {
                "status": "failed",
                "http_status": None,
                "mode": "publisher_feed",
                "error": f"Publisher feed connection error: {str(e)}",
            }

    # No token and no legitimate publisher feed:
    # Check public reachability of the profile page to confirm URL is valid,
    # but strictly mark as 'requires_integration' because X requires authentication for tweets.
    try:
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(src.url)
            # Profile exists and responds
            return {
                "status": "requires_integration",
                "http_status": resp.status_code,
                "mode": "profile_verified",
                "error": "يتطلب تكامل X API (Requires Integration - X_BEARER_TOKEN) للوصول المصرح إلى المنشورات.",
            }
    except Exception as e:
        return {
            "status": "failed",
            "http_status": None,
            "mode": "unreachable",
            "error": f"تعذر التحقق من رابط الحساب: {str(e)}",
        }


def classify_post_content(title: str, summary: str, body: str = "") -> Dict[str, Any]:
    """Classify post or article into one of the 6 required categories:
    1. Fraud Incident (confirmed_incident / reported_incident)
    2. Fraud Warning (fraud_warning)
    3. Cyber Threat (general_cybersecurity)
    4. Security Awareness (awareness_campaign)
    5. General News (general_news)
    6. Unrelated (unrelated)
    """
    text = f"{title} {summary} {body}".strip()
    norm_text = fraud_intel.norm(text)

    # Check for fraud signal
    has_fraud = fraud_intel.has_fraud_signal(text)

    # Check for general cyber threat without fraud
    is_cyber = any(term in norm_text for term in ["اختراق", "ثغرة", "برمجية خبيثة", "فدية", "vulnerability", "malware", "ransomware", "exploit", "cve", "zero-day"])

    if has_fraud:
        content_type, reasons = fraud_intel.classify_content_type(title, summary, body)
        if content_type in fraud_intel.INCIDENT_CONTENT_TYPES:
            category_key = "fraud_incident"
            category_label = "حادثة احتيال"
        elif content_type == fraud_intel.CT_WARNING:
            category_key = "fraud_warning"
            category_label = "تحذير احتيالي"
        elif content_type == fraud_intel.CT_AWARENESS:
            category_key = "security_awareness"
            category_label = "توعية أمنية"
        elif content_type == fraud_intel.CT_GENERAL:
            # Distinguish security awareness/training from technical cyber threats
            if any(cue in norm_text for cue in ["توعي", "توعو", "تثقيف", "نصائح", "ارشادات", "مبادره", "حمله", "ورشه", "دوره", "تدريب", "awareness", "guidance", "tips", "training"]):
                category_key = "security_awareness"
                category_label = "توعية أمنية"
            else:
                category_key = "cyber_threat"
                category_label = "تهديد سيبراني"
        else:
            category_key = "unrelated"
            category_label = "غير ذي صلة"
    elif is_cyber:
        category_key = "cyber_threat"
        category_label = "تهديد سيبراني"
        content_type = fraud_intel.CT_GENERAL
        reasons = ["cyber_keywords_matched"]
    else:
        # Check if it has news value or unrelated
        category_key = "unrelated"
        category_label = "غير ذي صلة"
        content_type = fraud_intel.CT_UNRELATED
        reasons = ["no_fraud_or_cyber_signal"]

    # Fraud incidents must NEVER contain general cybersecurity
    eligible_for_incident = category_key == "fraud_incident"
    eligible_for_explore = category_key in ("fraud_incident", "fraud_warning", "security_awareness")

    return {
        "category_key": category_key,
        "category_label": category_label,
        "content_type": content_type,
        "reasons": reasons,
        "eligible_for_incident": eligible_for_incident,
        "eligible_for_explore": eligible_for_explore,
    }


async def ingest_x_source(src_code: str) -> Dict[str, Any]:
    """Run ingestion for a single X source adhering strictly to deduplication and authorization rules."""
    src = X_SOURCES_BY_CODE.get(src_code)
    if not src:
        raise ValueError(f"Unknown X source code: {src_code}")

    conn_info = await check_source_connectivity(src)
    now_iso = datetime.now(timezone.utc).isoformat()

    stats = {
        "source_code": src.code,
        "source_name": src.name,
        "handle": src.handle,
        "status": conn_info["status"],
        "mode": conn_info["mode"],
        "error": conn_info["error"],
        "total_fetched": 0,
        "new_stored": 0,
        "duplicates_linked": 0,
        "classified_categories": {
            "fraud_incident": 0,
            "fraud_warning": 0,
            "cyber_threat": 0,
            "security_awareness": 0,
            "general_news": 0,
            "unrelated": 0,
        },
        "eligible_intelligence_items": 0,
    }

    if conn_info["status"] == "requires_integration":
        con = get_db()
        try:
            con.execute(
                """
                UPDATE source_registry
                SET last_status = 'requires_integration',
                    last_error = ?,
                    last_failure_at = ?
                WHERE code = ?
                """,
                (conn_info["error"], now_iso, src.code),
            )
            con.commit()
        finally:
            con.close()
        return stats

    if conn_info["status"] == "failed":
        con = get_db()
        try:
            con.execute(
                """
                UPDATE source_registry
                SET last_status = 'failed',
                    last_error = ?,
                    last_failure_at = ?,
                    consecutive_failures = consecutive_failures + 1
                WHERE code = ?
                """,
                (conn_info["error"], now_iso, src.code),
            )
            con.commit()
        finally:
            con.close()
        return stats

    # Ingestion via publisher feed (e.g. The Hacker News)
    if conn_info["mode"] == "publisher_feed" and src.publisher_feed:
        try:
            async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(src.publisher_feed)
                parsed = feedparser.parse(resp.text)
                entries = parsed.entries or []
                stats["total_fetched"] = len(entries)

                for e in entries[:25]:  # Batch of latest articles
                    link = getattr(e, "link", "").strip()
                    title = getattr(e, "title", "").strip()
                    summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
                    pub_date = getattr(e, "published", "") or now_iso

                    if not link or not title:
                        continue

                    # Step 4: Deduplication check
                    dup = find_duplicate_article(link, None)
                    classification = classify_post_content(title, summary)
                    cat_key = classification["category_key"]
                    stats["classified_categories"][cat_key] = stats["classified_categories"].get(cat_key, 0) + 1

                    if dup:
                        # Existing article found in MetaView!
                        # Link post/broadcast without duplicating
                        article_id = dup["id"]
                        stats["duplicates_linked"] += 1

                        con = get_db()
                        try:
                            con.execute(
                                """
                                UPDATE threat_intel
                                SET evidence_urls = CASE
                                    WHEN evidence_urls IS NULL OR evidence_urls = '' THEN ?
                                    WHEN evidence_urls LIKE ? THEN evidence_urls
                                    ELSE evidence_urls || ',' || ?
                                END
                                WHERE article_id = ?
                                """,
                                (src.url, f"%{src.handle}%", src.url, article_id),
                            )
                            con.commit()
                        finally:
                            con.close()
                    else:
                        # New article
                        item = {
                            "source": f"{src.name} ({src.handle})",
                            "domain": "thehackernews.com",
                            "country": src.country,
                            "headline": title,
                            "content": "",
                            "article_summary": summary,
                            "url": link,
                            "published_at": pub_date,
                            "created_at": now_iso,
                            "category": "Cybersecurity",
                            "source_code": src.code,
                            "language": src.lang,
                        }
                        article_id, is_new = insert_article_if_new(item)
                        if is_new and article_id:
                            stats["new_stored"] += 1

                            # If classified as fraud warning/awareness/incident, process into threat_intel
                            if classification["eligible_for_explore"]:
                                stats["eligible_intelligence_items"] += 1
                                item_for_intel = item.copy()
                                item_for_intel["article_id"] = article_id
                                intel.process_article_for_intel(item_for_intel)
                            elif classification["category_key"] == "cyber_threat":
                                # Stored as cyber threat, strictly NOT added to incidents
                                pass

                # Update registry to OK with last_success_at
                con = get_db()
                try:
                    con.execute(
                        """
                        UPDATE source_registry
                        SET last_status = 'ok',
                            last_success_at = ?,
                            last_error = NULL,
                            last_entries = ?,
                            consecutive_failures = 0
                        WHERE code = ?
                        """,
                        (now_iso, stats["total_fetched"], src.code),
                    )
                    con.commit()
                finally:
                    con.close()
        except Exception as e:
            stats["status"] = "failed"
            stats["error"] = str(e)
            con = get_db()
            try:
                con.execute(
                    """
                    UPDATE source_registry
                    SET last_status = 'failed',
                        last_failure_at = ?,
                        last_error = ?
                    WHERE code = ?
                    """,
                    (now_iso, str(e), src.code),
                )
                con.commit()
            finally:
                con.close()

    return stats
