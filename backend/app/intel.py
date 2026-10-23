# backend/app/intel.py
"""Threat Intelligence Engine

This module provides simple keyword‑based extraction of threat information
from article headlines/content. It is deliberately lightweight – the goal is
to have a working prototype that can be expanded later.

Key responsibilities:
- Classify an article into a threat type (e.g. phishing, ransomware, fraud).
- Identify the attack method / technique.
- Determine affected entities/sectors.
- Extract any CVE identifiers present in the text.
- Produce a `ThreatIntel` Pydantic model instance and persist it to the
  `threat_intel` table.
- Optionally create links between related threat records.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from .db import get_db
from .schemas import ThreatIntel, SourceTrust, CVEReference, IncidentLink

# Simple keyword mapping – can be extended later
THREAT_KEYWORDS = {
    "phishing": "phishing",
    "ransomware": "ransomware",
    "malware": "malware",
    "fraud": "financial_fraud",
    "scam": "financial_fraud",
    "bank": "banking_fraud",
    "cyber": "cyber_attack",
    "attack": "cyber_attack",
    "exploit": "exploit",
    "vulnerability": "vulnerability",
}

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def _detect_threat_type(text: str) -> str:
    """Return a threat type based on keyword matches.
    The first matching keyword wins; fallback to "Other".
    """
    lowered = text.lower()
    for kw, tt in THREAT_KEYWORDS.items():
        if kw in lowered:
            return tt
    return "Other"


def _extract_cves(text: str) -> List[str]:
    """Find CVE identifiers in the given text."""
    return list({m.upper() for m in CVE_PATTERN.findall(text)})


def _store_threat_intel(record: Dict[str, Any]) -> int:
    """Insert a threat_intel row and return its generated id.
    The caller must provide a dict matching the ThreatIntel schema fields.
    """
    con = get_db()
    cur = con.cursor()
    try:
        cur.execute(
            """
            INSERT INTO threat_intel (
                article_id, title, description, threat_type, event_type,
                evidence_confidence, analysis_confidence, verified_facts,
                analytical_assessment, detection_indicators,
                preventive_recommendations, evidence_urls,
                confidence, original_url, related_links, is_saudi, is_financial,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("article_id"),
                record.get("title"),
                record.get("description"),
                record.get("threat_type"),
                record.get("event_type"),
                record.get("evidence_confidence"),
                record.get("analysis_confidence"),
                record.get("verified_facts"),
                record.get("analytical_assessment"),
                record.get("detection_indicators"),
                record.get("preventive_recommendations"),
                record.get("evidence_urls"),
                record.get("confidence"),
                record.get("original_url"),
                record.get("related_links"),
                record.get("is_saudi"),
                record.get("is_financial"),
                record.get("created_at"),
                record.get("updated_at"),
            ),
        )
        threat_id = cur.lastrowid
    except Exception:
        # Duplicate article_id or other constraint violation – fetch existing
        cur.execute("SELECT id FROM threat_intel WHERE article_id = ?", (record.get("article_id"),))
        row = cur.fetchone()
        threat_id = row["id"] if row else None
    con.commit()
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
        does not qualify as a threat record.
    """
    # Combine headline and summary for analysis
    text_to_analyse = f"{article.get('headline', '')} {article.get('article_summary', '')}"
    threat_type = _detect_threat_type(text_to_analyse)
    if threat_type == "Other":
        return None

    # Extract CVE identifiers (if any) and store them separately
    cve_ids = _extract_cves(text_to_analyse)
    _store_cve_references(cve_ids)

    # Determine event type based on heuristics – differentiate incident certainty and guidance
    lowered = text_to_analyse.lower()
    if any(word in lowered for word in ["confirmed", "detected", "breach", "compromise", "attack"]):
        event_type = "confirmed_incident"
    elif any(word in lowered for word in ["reported", "suspected", "possible"]):
        event_type = "reported_incident"
    elif cve_ids:
        event_type = "vulnerability"
    elif threat_type != "Other":
        # If we have a threat type but no strong incident wording, treat as warning
        event_type = "warning"
    elif any(word in lowered for word in ["recommend", "prevent", "guidance", "advice"]):
        event_type = "guidance"
    else:
        event_type = "research"

    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "article_id": article.get("article_id"),
        "title": article.get("headline"),
        "description": article.get("article_summary") or article.get("content", ""),
        "threat_type": threat_type,
        "event_type": event_type,
        # Confidence fields – static placeholders for now; could be refined later
        "evidence_confidence": 0.9,
        "analysis_confidence": 0.9,
        "verified_facts": None,
        "analytical_assessment": None,
        "detection_indicators": None,
        "preventive_recommendations": None,
        "evidence_urls": ",".join(cve_ids) if cve_ids else None,
        "confidence": 0.9,
        "original_url": article.get("url"),
        "related_links": ",".join(cve_ids) if cve_ids else None,
        "is_saudi": 1 if article.get("country") == "SA" else 0,
        "is_financial": 1 if "bank" in lowered else 0,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    return _store_threat_intel(record)


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

__all__ = ["process_article_for_intel", "link_related_threats"]
