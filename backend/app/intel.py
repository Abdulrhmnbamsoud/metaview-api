# backend/app/intel.py
"""Threat Intelligence Engine

This module turns news articles into threat-intelligence records. Classification is
delegated to :mod:`fraud_intel`, a deterministic keyword/heuristic engine for Arabic
and English fraud news (see that module for the taxonomy and the facts-vs-inference
split).

Key responsibilities:
- Classify an article into a fraud/threat type (e.g. financial_fraud, link_phishing).
  A bank or money mention alone never qualifies an article.
- Identify the attack method / technique and channels.
- Determine affected targets and relevance to the Saudi banking sector.
- Extract any CVE identifiers present in the text.
- Persist the result to the existing `threat_intel` table (one row per article).
- Optionally create links between related threat records.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

from . import fraud_intel
from .db import get_db
from .utils import extract_domain

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

_SAUDI_COUNTRY_VALUES = {"SA", "SAUDI ARABIA", "KSA", "السعودية"}


def _extract_cves(text: str) -> List[str]:
    """Find CVE identifiers in the given text."""
    return list({m.upper() for m in CVE_PATTERN.findall(text)})


def _j(value: Any) -> Optional[str]:
    """JSON-encode structured analysis fields (Arabic preserved)."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _store_threat_intel(record: Dict[str, Any]) -> Optional[int]:
    """Insert (or refresh) the threat_intel row of an article and return its id.

    ``threat_intel.article_id`` is UNIQUE, so re-analysing an article updates its
    existing row instead of creating a duplicate.
    """
    con = get_db()
    cur = con.cursor()
    cols = [
        "article_id", "title", "description", "threat_type", "event_type",
        "evidence_confidence", "analysis_confidence", "verified_facts",
        "analytical_assessment", "detection_indicators",
        "preventive_recommendations", "evidence_urls",
        "confidence", "original_url", "related_links", "is_saudi", "is_financial",
        "created_at", "updated_at",
        "fraud_category", "fraud_category_ar", "modus_operandi", "channels", "target_group",
        "bank_relevance", "detection_rule", "required_data", "matched_terms", "content_type",
    ]
    values = [record.get(c) for c in cols]
    placeholders = ", ".join("?" for _ in cols)
    update_cols = [c for c in cols if c not in ("article_id", "created_at")]
    update_sql = ", ".join(f"{c} = excluded.{c}" for c in update_cols)
    threat_id: Optional[int] = None
    try:
        cur.execute(
            f"INSERT INTO threat_intel ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(article_id) DO UPDATE SET {update_sql}",
            values,
        )
        cur.execute("SELECT id FROM threat_intel WHERE article_id = ?", (record.get("article_id"),))
        row = cur.fetchone()
        threat_id = row["id"] if row else None
        con.commit()
    except sqlite3.Error:
        con.rollback()
        threat_id = None
    finally:
        con.close()
    return threat_id


def _store_cve_references(cve_ids: List[str]) -> None:
    """Store CVE references if they are not already present."""
    if not cve_ids:
        return
    con = get_db()
    cur = con.cursor()
    for cve in cve_ids:
        cur.execute(
            "INSERT OR IGNORE INTO cve_reference (cve_id, description, severity) VALUES (?, ?, ?)",
            (cve, "", ""),
        )
    con.commit()
    con.close()


def analyze_and_store(
    article: Dict[str, Any],
    require_saudi: bool = False,
) -> Tuple[Optional[int], Optional[str]]:
    """Analyse an article and persist the threat-intel record.

    Returns ``(threat_id, None)`` when the article qualifies as fraud intelligence, or
    ``(None, reason)`` where reason is ``no_fraud_signal`` | ``non_financial_context``
    | ``not_saudi`` | ``store_failed``.
    """
    title = article.get("headline") or ""
    summary = article.get("article_summary") or ""
    body = article.get("content") or ""
    url = article.get("url") or ""
    country = str(article.get("country") or "").strip().upper()
    source_meta = {
        "domain": article.get("domain") or extract_domain(url),
        "saudi_publisher": bool(article.get("saudi_publisher")) or country in _SAUDI_COUNTRY_VALUES,
        "saudi_scope": bool(article.get("saudi_scope")) or country in _SAUDI_COUNTRY_VALUES,
        "official": bool(article.get("official")),
    }
    analysis, reason = fraud_intel.analyze(title, summary, body, source=source_meta, require_saudi=require_saudi)
    if analysis is None:
        return None, reason

    full_text = f"{title} {summary} {body}"
    cve_ids = _extract_cves(full_text)
    _store_cve_references(cve_ids)

    now_iso = datetime.now(timezone.utc).isoformat()
    facts = analysis["verified_facts"]
    assessment = analysis["analytical_assessment"]
    record = {
        "article_id": article.get("article_id"),
        "title": title or "(no title)",
        "description": facts.get("incident_summary") or summary or title or "(no description)",
        "threat_type": analysis["fraud_category"],
        "event_type": analysis["event_type"],
        "evidence_confidence": analysis["evidence_confidence"],
        "analysis_confidence": analysis["analysis_confidence"],
        "verified_facts": _j(facts),
        "analytical_assessment": _j(assessment),
        "detection_indicators": _j(analysis["detection_indicators"]),
        "preventive_recommendations": _j(analysis["preventive_recommendations"]),
        "evidence_urls": url or None,
        "confidence": analysis["confidence"],
        "original_url": url or None,
        "related_links": ",".join(cve_ids) if cve_ids else None,
        "is_saudi": 1 if analysis["is_saudi"] else 0,
        "is_financial": 1 if analysis["is_financial"] else 0,
        "created_at": now_iso,
        "updated_at": now_iso,
        "fraud_category": analysis["fraud_category"],
        "fraud_category_ar": analysis["fraud_category_ar"],
        "modus_operandi": assessment["modus_operandi"],
        "channels": _j(analysis["channels"]),
        "target_group": _j(analysis["target_group"]),
        "bank_relevance": analysis["bank_relevance"]["level"],
        "detection_rule": _j(analysis["detection_rule"]),
        "required_data": _j(analysis["required_data"]),
        "matched_terms": _j(analysis["matched_terms"]),
        "content_type": analysis["content_type"],
    }
    threat_id = _store_threat_intel(record)
    if threat_id is None:
        return None, "store_failed"
    return threat_id, None


def process_article_for_intel(article: Dict[str, Any]) -> Optional[int]:
    """Analyse an article and store a threat intelligence record.

    Parameters
    ----------
    article: dict
        The article dict as produced by the ingest pipeline (same keys as the
        `Article` Pydantic model). Must contain an ``article_id`` key linking to
        the ``articles`` table.

    Returns
    -------
    int | None
        The newly created ``threat_intel`` row id, or ``None`` if the article
        does not qualify as a fraud-intelligence record.
    """
    threat_id, _reason = analyze_and_store(article, require_saudi=False)
    return threat_id


def link_related_threats(threat_id: int, related_ids: List[int]) -> None:
    """Create incident_links entries linking a threat to other threat ids.
    """
    if not related_ids:
        return
    con = get_db()
    cur = con.cursor()
    for rid in related_ids:
        cur.execute(
            "INSERT INTO incident_links (threat_id, related_threat_id, relation_type) VALUES (?, ?, ?)",
            (threat_id, rid, "related"),
        )
    con.commit()
    con.close()

__all__ = ["process_article_for_intel", "analyze_and_store", "link_related_threats"]
