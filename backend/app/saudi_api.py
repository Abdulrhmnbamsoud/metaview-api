# backend/app/saudi_api.py
"""REST API for the Saudi Fraud Intelligence section (read endpoints + manual ingest trigger)."""
from __future__ import annotations

import hmac
import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .config import ADMIN_TOKEN, DATA_DIR, DB_PATH, SAUDI_AUTO_INGEST, SAUDI_INGEST_EVERY_HOURS
from .db import get_db
from . import incidents, saudi_ingest
from .fraud_intel import CATEGORY_AR, CATEGORY_ORDER, CONTENT_TYPE_AR, VISIBLE_CONTENT_TYPES
from .auth import validate_session
from .canonical_query import build_canonical_incident_where

router = APIRouter(prefix="/saudi-fraud", tags=["saudi-fraud-intelligence"])

# Saudi item = Saudi-attributed threat that went through the fraud engine (any content type)
_ANY_SAUDI_WHERE = "t.is_saudi = 1 AND t.fraud_category IS NOT NULL"
# Default visibility: only fraud-relevant content. general_cybersecurity / unrelated / unclassified
# articles stay in the DB but never appear in lists or indicators unless explicitly requested.
_VISIBLE_TYPES = "', '".join(VISIBLE_CONTENT_TYPES)
_BASE_WHERE = f"{_ANY_SAUDI_WHERE} AND t.content_type IN ('{_VISIBLE_TYPES}')"

_JSON_FIELDS = (
    "verified_facts", "analytical_assessment", "detection_indicators",
    "preventive_recommendations", "evidence_urls", "channels", "target_group",
    "detection_rule", "required_data", "matched_terms",
)


def _loads(v: Any) -> Any:
    if v is None or v == "":
        return None
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return v


def _item(r, detail: bool = False) -> Dict[str, Any]:
    incident_country = r["incident_country"] if ("incident_country" in r.keys() and r["incident_country"] and r["incident_country"] != "Unknown") else None
    article_country = r["country"] if ("country" in r.keys() and r["country"] and r["country"] != "Global") else None
    resolved_country = incident_country or ("SA" if r["is_saudi"] else (article_country or "Unknown"))

    d: Dict[str, Any] = {
        "id": r["id"],
        "article_id": r["article_id"],
        "title": r["title"],
        "headline": r["headline"],
        "summary": r["description"],
        "fraud_category": r["fraud_category"],
        "fraud_category_ar": r["fraud_category_ar"] or CATEGORY_AR.get(r["fraud_category"]),
        "event_type": r["event_type"],
        "source": r["source"],
        "source_code": r["source_code"],
        "original_url": r["original_url"] or r["url"],
        "published_at": r["published_at"],
        "fetched_at": r["fetched_at"] or r["created_at"],
        "bank_relevance": r["bank_relevance"],
        "confidence": r["confidence"],
        "evidence_confidence": r["evidence_confidence"],
        "analysis_confidence": r["analysis_confidence"],
        "modus_operandi": r["modus_operandi"],
        "channels": _loads(r["channels"]) or [],
        "target_group": _loads(r["target_group"]) or [],
        "detection_rule": _loads(r["detection_rule"]),
        "content_type": r["content_type"] or "unclassified",
        "content_type_ar": CONTENT_TYPE_AR.get(r["content_type"], "غير مصنّف"),
        "incident_id": r["incident_id"],
        "country": resolved_country,
        "incident_country": incident_country,
        "incident_city": r["incident_city"] if ("incident_city" in r.keys() and r["incident_city"]) else None,
        "occurred_at": r["occurred_at"] if ("occurred_at" in r.keys() and r["occurred_at"]) else None,
        "domain": r["domain"] if "domain" in r.keys() else None,
        "is_saudi": bool(r["is_saudi"]),
    }
    if detail:
        for f in ("verified_facts", "analytical_assessment", "detection_indicators",
                  "preventive_recommendations", "evidence_urls", "required_data", "matched_terms"):
            d[f] = _loads(r[f])
        d["language"] = r["language"]
        d["content_excerpt"] = (r["content"] or "")[:1500]
    return d


_SELECT = """
    SELECT t.*, a.headline, a.source, a.url, a.published_at, a.fetched_at, a.source_code,
           a.language, a.content, a.country, a.domain,
           i.incident_country, i.incident_city, i.incident_region, i.occurred_at
    FROM threat_intel t
    JOIN articles a ON a.id = t.article_id
    LEFT JOIN incidents i ON i.id = t.incident_id
"""


def _where(category: Optional[str], source: Optional[str], q: Optional[str],
           date_from: Optional[str], date_to: Optional[str], bank: Optional[str],
           content_type: Optional[str] = None, scope: str = "all", country: Optional[str] = None,
           window_days: Optional[int] = None):
    clauses = []
    params: List[Any] = []

    # Geographic / Scope condition
    if country and country != "all":
        if country.upper() == "SA":
            clauses.append("(t.is_saudi = 1 OR (t.incident_id IS NOT NULL AND i.incident_country = 'SA'))")
        else:
            clauses.append("((t.incident_id IS NOT NULL AND i.incident_country = ?) OR (t.incident_id IS NULL AND a.country = ?))")
            params.extend([country.upper(), country.upper()])
    elif scope == "saudi":
        clauses.append("(t.is_saudi = 1 OR (t.incident_id IS NOT NULL AND i.incident_country = 'SA'))")
    elif scope == "global":
        clauses.append("(t.is_saudi = 0 OR t.is_saudi IS NULL) AND (i.incident_country != 'SA' OR i.incident_country IS NULL)")
    elif scope == "all":
        pass  # Both Saudi and Global included

    clauses.append("t.fraud_category IS NOT NULL")

    if content_type:
        clauses.append("t.content_type = ?")
        params.append(content_type)
    else:
        clauses.append(f"t.content_type IN ('{_VISIBLE_TYPES}')")

    if category and category != "all":
        clauses.append("t.fraud_category = ?")
        params.append(category)

    if source and source != "all":
        clauses.append("(a.source_code = ? OR a.source = ?)")
        params += [source, source]

    if bank and bank != "all":
        clauses.append("t.bank_relevance = ?")
        params.append(bank)

    if q:
        clauses.append("(t.title LIKE ? OR t.description LIKE ? OR a.headline LIKE ? OR t.fraud_category LIKE ? OR t.fraud_category_ar LIKE ? OR t.channels LIKE ?)")
        params += [f"%{q}%"] * 6

    if date_from:
        clauses.append("a.published_at >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("a.published_at <= ?")
        params.append(date_to + "T23:59:59" if len(date_to) == 10 else date_to)

    if isinstance(window_days, int) and window_days > 0 and not date_from:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        clauses.append("a.published_at >= ?")
        params.append(cutoff)

    return (" AND ".join(clauses) if clauses else "1=1"), params


@router.get("/overview")
def overview(
    window_days: Optional[int] = Query(None, ge=1, le=365),
    scope: str = Query("all", pattern="^(saudi|global|all)$"),
    country: Optional[str] = None,
    fraud_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_basis: str = Query("first_reported_at"),
) -> Dict[str, Any]:
    """Headline indicators. The success metric is INDEPENDENT INCIDENTS, not articles: warnings and
    awareness campaigns are counted separately and general-cybersecurity/unrelated articles are
    excluded from every indicator (they remain stored)."""
    con = get_db()
    try:
        inc = incidents.counts(con)

        # 30-day canonical benchmark
        c30_clauses, c30_params, c30_meta = build_canonical_incident_where(geo=scope, country=country, window_days=30)
        c30_dt, c30_dp = c30_meta["make_date_clause"](c30_meta["iso_from"], c30_meta["iso_to"])
        if c30_dt:
            c30_clauses.append(c30_dt)
            c30_params.extend(c30_dp)
        c30_total = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {' AND '.join(c30_clauses)}", c30_params).fetchone()["n"]

        eff_w = window_days if isinstance(window_days, int) and window_days > 0 else 30
        clauses, params, meta = build_canonical_incident_where(
            geo=scope,
            country=country,
            fraud_type=fraud_type,
            date_from=date_from,
            date_to=date_to,
            window_days=eff_w,
            date_basis=date_basis,
        )
        dt_clause, dt_params = meta["make_date_clause"](meta["iso_from"], meta["iso_to"])
        if dt_clause:
            clauses.append(dt_clause)
            params.extend(dt_params)
        where_inc = " AND ".join(clauses)
        canonical_total = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {where_inc}", params).fetchone()["n"]
        canonical_high = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {where_inc} AND LOWER(bank_relevance) = 'high'", params).fetchone()["n"]
        canonical_confirmed = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {where_inc} AND record_type = 'confirmed_incident'", params).fetchone()["n"]
        canonical_reported = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {where_inc} AND record_type = 'reported_incident'", params).fetchone()["n"]

        # 7-day canonical count within this scope
        c7_clauses, c7_params, c7_meta = build_canonical_incident_where(
            geo=scope, country=country, fraud_type=fraud_type, window_days=7, date_basis=date_basis
        )
        c7_dt, c7_dp = c7_meta["make_date_clause"](c7_meta["iso_from"], c7_meta["iso_to"])
        if c7_dt:
            c7_clauses.append(c7_dt)
            c7_params.extend(c7_dp)
        canonical_last7 = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {' AND '.join(c7_clauses)}", c7_params).fetchone()["n"]

        # Total monitored items within current scope
        where_items, items_params = _where(None, None, None, date_from, date_to, None, scope=scope, country=country, window_days=window_days)
        total_items_count = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t JOIN articles a ON a.id=t.article_id LEFT JOIN incidents i ON i.id = t.incident_id WHERE {where_items}",
            items_params).fetchone()["n"]
        srcs = con.execute(
            f"SELECT COUNT(DISTINCT a.source) n FROM threat_intel t JOIN articles a ON a.id=t.article_id LEFT JOIN incidents i ON i.id = t.incident_id WHERE {where_items}",
            items_params).fetchone()["n"]

        # Telemetry metrics
        last_check = con.execute("SELECT MAX(finished_at) m FROM fetch_log").fetchone()["m"]
        last_success = con.execute("SELECT MAX(finished_at) m FROM fetch_log WHERE ok = 1").fetchone()["m"]
        last_item_dt = con.execute(
            f"SELECT MAX(a.published_at) m FROM threat_intel t JOIN articles a ON a.id = t.article_id LEFT JOIN incidents i ON i.id = t.incident_id WHERE {where_items}",
            items_params).fetchone()["m"]
        last_analysis = con.execute("SELECT MAX(updated_at) m FROM threat_intel").fetchone()["m"]

        ok_sources = con.execute("SELECT COUNT(*) n FROM source_registry WHERE last_status='ok'").fetchone()["n"]
        all_sources = con.execute("SELECT COUNT(*) n FROM source_registry").fetchone()["n"]
    finally:
        con.close()
    cats = inc["incident_categories"]
    return {
        # --- success metric -------------------------------------------------
        "incidents_total": canonical_total,
        "incidents_last_30_days": c30_total,
        "incident_articles_total": inc["incident_articles_total"],
        "incidents_last_7_days": canonical_last7,
        "incidents_high_bank_relevance": canonical_high,
        "incidents_confirmed": canonical_confirmed,
        "incidents_reported": canonical_reported,
        "incidents_needs_review": inc["incidents_needs_review"],
        "merge_suggestions_pending": inc["merge_suggestions_pending"],
        # --- separate, NOT counted as incidents -----------------------------
        "warnings_total": inc["warnings_total"],
        "awareness_total": inc["awareness_total"],
        # --- stored but excluded from every fraud indicator ------------------
        "excluded_general_cybersecurity": inc["excluded_general_cybersecurity"],
        "excluded_unrelated": inc["excluded_unrelated"],
        "unclassified": inc["unclassified"],
        "by_content_type": inc["by_content_type"],
        # --- legacy & compatibility field names -----------------------------
        "total_items": total_items_count,
        "last_7_days": canonical_last7,
        "high_bank_relevance": canonical_high,
        "distinct_sources": srcs,
        "categories": [
            {"key": k, "label_ar": CATEGORY_AR.get(k, k), "count": cats.get(k, 0)}
            for k in CATEGORY_ORDER
        ],
        "last_fetch_at": last_check,
        # Separate telemetry indicators
        "last_source_check": last_check,
        "last_successful_fetch": last_success,
        "last_new_item": last_item_dt,
        "last_analysis_update": last_analysis,
        "sources_ok": ok_sources,
        "sources_total": all_sources,
        "refresh_every_hours": SAUDI_INGEST_EVERY_HOURS,
        "auto_ingest_enabled": SAUDI_AUTO_INGEST,
        "ingest_running": bool(saudi_ingest.RUN_STATE.get("running")) or bool(saudi_ingest.lease_status().get("active")),
    }


@router.get("/news")
def news(
    category: Optional[str] = None,
    source: Optional[str] = None,
    q: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    window_days: Optional[int] = Query(None, ge=1, le=365),
    sort_by: Optional[str] = Query("latest", pattern="^(latest|importance)$"),
    bank_relevance: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    content_type: Optional[str] = Query(
        None, pattern="^(confirmed_incident|reported_incident|fraud_warning|awareness_campaign|general_cybersecurity|unrelated)$"),
    scope: str = Query("all", pattern="^(saudi|global|all)$"),
    country: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Articles across Saudi or Global scope. Default is 'all' global scope.
    Strictly orders by original published_at or explainable importance formula."""
    eff_content_type = content_type if isinstance(content_type, str) else None
    eff_bank = bank_relevance if isinstance(bank_relevance, str) else None
    eff_scope = scope if isinstance(scope, str) and scope in ("saudi", "global", "all") else "all"
    eff_country = country if isinstance(country, str) else None
    eff_window = window_days if isinstance(window_days, int) and window_days > 0 else None
    eff_sort = sort_by if isinstance(sort_by, str) and sort_by in ("latest", "importance") else "latest"
    eff_page = page if isinstance(page, int) and page >= 1 else 1
    eff_page_size = page_size if isinstance(page_size, int) and page_size >= 1 else 20

    where, params = _where(
        category, source, q, date_from, date_to, eff_bank, eff_content_type,
        scope=eff_scope, country=eff_country, window_days=eff_window
    )
    con = get_db()
    try:
        total = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t JOIN articles a ON a.id=t.article_id LEFT JOIN incidents i ON i.id = t.incident_id WHERE {where}",
            params).fetchone()["n"]

        if eff_sort == "importance":
            order_clause = (
                "((CASE WHEN LOWER(t.bank_relevance) = 'high' THEN 3 WHEN LOWER(t.bank_relevance) = 'medium' THEN 2 ELSE 1 END) * 10 + "
                "(CASE WHEN t.content_type = 'confirmed_incident' THEN 5 WHEN t.content_type = 'fraud_warning' THEN 3 ELSE 1 END)) DESC, "
                "a.published_at DESC, t.id DESC"
            )
        else:
            order_clause = "a.published_at DESC, t.id DESC"

        rows = con.execute(
            f"{_SELECT} WHERE {where} ORDER BY {order_clause} "
            "LIMIT ? OFFSET ?", params + [eff_page_size, (eff_page - 1) * eff_page_size]).fetchall()
    finally:
        con.close()
    return {"page": page, "page_size": page_size, "total": total,
            "items": [_item(r) for r in rows]}


@router.get("/countries")
def fraud_countries(scope: str = Query("all", pattern="^(saudi|global|all)$")) -> Dict[str, Any]:
    """List countries having fraud intelligence records, with canonical incident counts matching map."""
    con = get_db()
    try:
        where = f"t.fraud_category IS NOT NULL AND t.content_type IN ('{_VISIBLE_TYPES}')"
        if scope == "saudi":
            where += " AND t.is_saudi = 1"
        elif scope == "global":
            where += " AND (t.is_saudi = 0 OR t.is_saudi IS NULL)"
        rows = con.execute(
            f"""
            SELECT COALESCE(NULLIF(a.country, ''), CASE WHEN t.is_saudi=1 THEN 'SA' ELSE 'Global' END) AS country,
                   COUNT(*) as count
            FROM threat_intel t JOIN articles a ON a.id = t.article_id
            WHERE {where}
            GROUP BY country
            ORDER BY count DESC
            """
        ).fetchall()

        # Query canonical incident counts per country (30 days window default)
        c_clauses, c_params, c_meta = build_canonical_incident_where(geo="all", window_days=30)
        c_dt, c_dp = c_meta["make_date_clause"](c_meta["iso_from"], c_meta["iso_to"])
        if c_dt:
            c_clauses.append(c_dt)
            c_params.extend(c_dp)
        inc_rows = con.execute(
            f"SELECT incident_country, COUNT(*) as cnt FROM incidents WHERE {' AND '.join(c_clauses)} GROUP BY incident_country",
            c_params
        ).fetchall()
        inc_map = {r["incident_country"]: r["cnt"] for r in inc_rows}
    finally:
        con.close()

    result = []
    for r in rows:
        c_code = r["country"]
        if not c_code:
            continue
        inc_cnt = inc_map.get(c_code, 0)
        result.append({
            "country": c_code,
            # Prefer real canonical incident count if available, falling back to article count
            "count": inc_cnt if inc_cnt > 0 else r["count"],
            "incidents_count": inc_cnt,
            "articles_count": r["count"],
        })
    return {"countries": result}


@router.get("/news/{threat_id}")
def news_detail(threat_id: int) -> Dict[str, Any]:
    con = get_db()
    try:
        r = con.execute(f"{_SELECT} WHERE t.id = ?", (threat_id,)).fetchone()
    finally:
        con.close()
    if not r:
        raise HTTPException(status_code=404, detail="Fraud item not found")
    return _item(r, detail=True)


@router.get("/sources")
def sources() -> Dict[str, Any]:
    con = get_db()
    try:
        rows = con.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM articles a WHERE a.source_code = s.code) AS stored "
            "FROM source_registry s ORDER BY s.official DESC, s.last_status, s.code").fetchall()
    finally:
        con.close()
    return {"sources": [dict(r) for r in rows]}


@router.get("/incidents")
def incidents_list(
    category: Optional[str] = None,
    fraud_type: Optional[str] = None,
    q: Optional[str] = None,
    bank_relevance: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    record_type: Optional[str] = Query(None, pattern="^(confirmed_incident|reported_incident)$"),
    review_status: Optional[str] = Query(None, pattern="^(auto_merged|single_source|needs_review|human_confirmed)$"),
    scope: str = Query("saudi", pattern="^(saudi|global|all)$"),
    country: Optional[str] = None,
    city: Optional[str] = None,
    region: Optional[str] = None,
    window_days: Optional[int] = Query(None, ge=1, le=365),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_basis: str = Query("first_reported_at"),
    incident_ids: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Independent fraud incidents (one row per real event), each with ALL the articles covering it.
    Uses the canonical query engine to guarantee 100% cross-page reconciliation."""
    raw_ids = [int(x.strip()) for x in str(incident_ids).split(",") if x.strip().isdigit()] if incident_ids else None
    eff_fraud_type = fraud_type or category

    clauses, params, meta = build_canonical_incident_where(
        geo=scope,
        country=country,
        city=city,
        region=region,
        fraud_type=eff_fraud_type,
        date_from=date_from,
        date_to=date_to,
        window_days=window_days,
        date_basis=date_basis,
        record_type=record_type,
        review_status=review_status,
        bank_relevance=bank_relevance,
        q=q,
        incident_ids=raw_ids,
    )

    valid_date_basis = meta["date_basis"]
    order_col = f"COALESCE({valid_date_basis}, created_at)"

    # Apply date clause if date bounds exist
    dt_clause, dt_params = meta["make_date_clause"](meta["iso_from"], meta["iso_to"])
    if dt_clause:
        clauses.append(dt_clause)
        params.extend(dt_params)

    eff_page = page if isinstance(page, int) and page >= 1 else 1
    eff_page_size = page_size if isinstance(page_size, int) and page_size >= 1 else 20

    where = " AND ".join(clauses)
    con = get_db()
    try:
        total = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {where}", params).fetchone()["n"]
        rows = con.execute(
            f"SELECT * FROM incidents WHERE {where} ORDER BY {order_col} DESC, id DESC "
            "LIMIT ? OFFSET ?", params + [eff_page_size, (eff_page - 1) * eff_page_size]).fetchall()
    finally:
        con.close()
    return {
        "page": eff_page,
        "page_size": eff_page_size,
        "total": total,
        "incident_ids": [int(r["id"]) for r in rows],
        "items": [incidents.incident_to_dict(r) for r in rows],
    }


def _check_admin(
    x_admin_token: Optional[str] = None,
    authorization: Optional[str] = None,
    required_permission: str = "incident.approve_merge",
) -> Optional[Any]:
    """Admin gate: accepts legacy X-Admin-Token or Bearer session token with required permission."""
    if x_admin_token and ADMIN_TOKEN and hmac.compare_digest(x_admin_token.encode(), ADMIN_TOKEN.encode()):
        return None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()
        user = validate_session(token)
        if user and user.can(required_permission):
            return user
    if not ADMIN_TOKEN and not authorization:
        raise HTTPException(status_code=503, detail="Disabled: ADMIN_TOKEN is not configured")
    raise HTTPException(status_code=401, detail="Invalid admin token or missing permission")


@router.get("/incidents/merge-suggestions")
def merge_suggestions(status: Optional[str] = Query("pending", pattern="^(pending|approved|rejected|all)$")) -> Dict[str, Any]:
    """Uncertain merges that were NOT applied automatically and wait for a human decision."""
    con = get_db()
    try:
        items = incidents.list_suggestions(con, None if status == "all" else status)
    finally:
        con.close()
    return {"status": status, "total": len(items), "items": items,
            "note": "Pairs share some facts but not enough (or only weakly) to merge safely. "
                    "Approving forces the merge; rejecting forbids it permanently."}


class MergeDecision(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    reviewer: str = Field(default="", max_length=80)
    reason: str = Field(default="", max_length=500)


@router.post("/incidents/merge-suggestions/{suggestion_id}/decision")
def merge_decision(
    suggestion_id: int,
    body: MergeDecision,
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """Record a reviewer decision (admin only). The incident set is rebuilt; no article is ever deleted."""
    user = _check_admin(x_admin_token, authorization, required_permission="incident.approve_merge")
    reviewer_name = body.reviewer or (user.full_name if user else "Admin")
    try:
        return incidents.decide_suggestion(suggestion_id, body.decision, reviewer_name, body.reason)
    except LookupError:
        raise HTTPException(status_code=404, detail="Merge suggestion not found")


@router.get("/incidents/report")
def incidents_report(format: str = Query("json", pattern="^(json|markdown)$")):
    """Articles vs independent incidents, warnings, awareness, review queue - with real links."""
    con = get_db()
    try:
        rep = incidents.build_report(con)
    finally:
        con.close()
    if format == "markdown":
        return PlainTextResponse(incidents.report_markdown(rep), media_type="text/markdown; charset=utf-8")
    return rep


@router.get("/incidents/{incident_id}")
def incident_detail(incident_id: int) -> Dict[str, Any]:
    con = get_db()
    try:
        r = con.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Incident not found")
        d = incidents.incident_to_dict(r, detail=True)
        ids = {a["article_id"] for a in d["articles"]}
        d["pending_merge_suggestions"] = [
            s for s in incidents.list_suggestions(con, "pending")
            if s["article_a"]["id"] in ids or s["article_b"]["id"] in ids]
    finally:
        con.close()
    return d


@router.get("/patterns")
def patterns() -> Dict[str, Any]:
    """Fraud patterns = methods stated in the sources + approved analyst patterns (drafts excluded)."""
    con = get_db()
    try:
        pats = incidents.build_patterns(con)
        # Append approved analyst custom patterns (viewer safe)
        try:
            approved_custom = con.execute("""
                SELECT cp.*, u.full_name as author_name, r.full_name as reviewer_name
                FROM custom_patterns cp
                LEFT JOIN users u ON cp.author_id = u.id
                LEFT JOIN users r ON cp.reviewed_by = r.id
                WHERE cp.status = 'approved'
            """).fetchall()
            for row in approved_custom:
                pats.append({
                    "pattern": row["key"],
                    "pattern_ar": row["label_ar"],
                    "category": "custom",
                    "category_ar": "نمط معتمد من المحللين",
                    "description": row["description"] or "",
                    "modus_operandi": row["modus_operandi"] or "",
                    "channels": _loads(row["channels"]) or [],
                    "target_groups": _loads(row["target_groups"]) or [],
                    "banking_relevance": row["banking_relevance"] or "",
                    "linked_incidents_count": len(_loads(row["linked_incident_ids"]) or []),
                    "status": "approved",
                    "author": row["author_name"],
                    "approved_by": row["reviewer_name"],
                    "approved_at": row["approved_at"],
                    "is_custom": True,
                })
        except Exception:
            pass  # If table not yet queried or empty

        chan: Dict[str, int] = {}
        for r in con.execute("SELECT channels FROM incidents"):
            for ch in _loads(r["channels"]) or []:
                chan[ch] = chan.get(ch, 0) + 1
    finally:
        con.close()
    return {
        "patterns": pats,
        "channels": [{"channel": k, "count": v} for k, v in sorted(chan.items(), key=lambda kv: -kv[1])],
        "note": "Patterns list only methods whose wording appears in a source article or approved analyst patterns.",
    }


@router.get("/patterns/{key}")
def pattern_detail(key: str) -> Dict[str, Any]:
    """Single fraud pattern intelligence detail with linked detection rules."""
    con = get_db()
    try:
        pats = incidents.build_patterns(con)
        matched = next((p for p in pats if p.get("key") == key), None)
        if not matched:
            raise HTTPException(status_code=404, detail="Pattern not found")
        # Find linked rules for this pattern
        rules_list = incidents.build_rules(con)
        linked_rules = [r for r in rules_list if any(p.get("key") == key for p in r.get("evidence", {}).get("patterns", []))]
        matched_copy = dict(matched)
        matched_copy["linked_rules"] = [
            {"rule_id": r["rule_id"], "name": r["name"], "hypothesis": r["hypothesis"], "logic": r.get("logic", "")}
            for r in linked_rules
        ]
        return {"pattern": matched_copy}
    finally:
        con.close()



@router.get("/rules")
def rules() -> Dict[str, Any]:
    """Detection rules linked to incidents/patterns + approved analyst rules (drafts excluded)."""
    con = get_db()
    try:
        items = incidents.build_rules(con)
        try:
            approved_custom = con.execute("""
                SELECT cr.*, u.full_name as author_name, r.full_name as reviewer_name
                FROM custom_rules cr
                LEFT JOIN users u ON cr.author_id = u.id
                LEFT JOIN users r ON cr.reviewed_by = r.id
                WHERE cr.status = 'approved'
            """).fetchall()
            for row in approved_custom:
                items.append({
                    "rule_id": row["rule_id"],
                    "name": row["name"],
                    "hypothesis": row["hypothesis"],
                    "logic": row["logic"],
                    "pattern_key": row["pattern_key"],
                    "required_data": _loads(row["required_data"]) or [],
                    "exceptions": _loads(row["exceptions"]) or [],
                    "false_positive_risk": row["false_positive_risk"],
                    "status": "tested" if row["test_results"] else "approved",
                    "test_results": _loads(row["test_results"]),
                    "author": row["author_name"],
                    "approved_by": row["reviewer_name"],
                    "approved_at": row["approved_at"],
                    "is_custom": True,
                })
        except Exception:
            pass

    finally:
        con.close()
    return {"rules": items,
            "summary": {"total": len(items),
                        "tested": sum(1 for x in items if x["status"] == "tested"),
                        "proposed_untested": sum(1 for x in items if x["status"] in ("proposed_untested", "draft"))},
            "note": "Rules are hypotheses unless a real test result is recorded; drafts are hidden from viewers."}


@router.get("/ingest/runs")
def ingest_runs(limit: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    con = get_db()
    try:
        rows = con.execute(
            "SELECT * FROM fetch_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    finally:
        con.close()
    return {
        "state": {k: v for k, v in saudi_ingest.RUN_STATE.items() if k != "last_report"},
        "last_report_summary": {k: v for k, v in (saudi_ingest.RUN_STATE.get("last_report") or {}).items()
                                if k not in ("sources", "new_examples")} or None,
        "log": [dict(r) for r in rows],
    }


MANUAL_COOLDOWN_SECONDS = 600  # min gap between the end of one run and a manual start


@router.post("/ingest/run")
def ingest_run(
    x_admin_token: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """Manual pull (admin only, fail-closed). Accepts X-Admin-Token or Bearer session token."""
    _check_admin(x_admin_token, authorization, required_permission="ingestion.trigger")
    if saudi_ingest.RUN_STATE.get("running") or saudi_ingest.lease_status().get("active"):
        return saudi_ingest.start_background_run("manual")  # reports already_running
    last = saudi_ingest._last_finished()
    if last is not None:
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        wait = MANUAL_COOLDOWN_SECONDS - (datetime.now(timezone.utc) - last).total_seconds()
        if wait > 0:
            raise HTTPException(
                status_code=429, headers={"Retry-After": str(int(wait) + 1)},
                detail=f"Last ingest finished recently; manual runs are rate-limited. Retry in {int(wait) + 1}s")
    return saudi_ingest.start_background_run("manual")


@router.get("/runtime")
def runtime() -> Dict[str, Any]:
    """Read-only deployment diagnostics (no secrets): deployed commit, DB/disk binding, scheduler state."""
    db_real = os.path.realpath(DB_PATH)
    data_real = os.path.realpath(DATA_DIR)
    in_data = db_real == data_real or db_real.startswith(data_real + os.sep)
    disk: Dict[str, Any] = {"data_dir": DATA_DIR, "db_path": DB_PATH, "db_inside_data_dir": in_data}
    try:
        disk["data_dir_is_mount_point"] = os.path.ismount(data_real)
        disk["data_dir_device"] = os.stat(data_real).st_dev
        disk["root_device"] = os.stat("/").st_dev
        disk["data_dir_on_separate_device"] = disk["data_dir_device"] != disk["root_device"]
        disk["db_size_bytes"] = os.path.getsize(db_real) if os.path.exists(db_real) else None
        disk["free_bytes"] = shutil.disk_usage(data_real).free
    except OSError as e:  # pragma: no cover
        disk["error"] = str(e)
    journal = None
    try:
        con = get_db()
        journal = con.execute("PRAGMA journal_mode").fetchone()[0]
        con.close()
    except Exception as e:  # pragma: no cover
        journal = f"error: {e}"
    return {
        "commit": os.getenv("RENDER_GIT_COMMIT"),
        "branch": os.getenv("RENDER_GIT_BRANCH"),
        "service": os.getenv("RENDER_SERVICE_NAME"),
        "on_render": bool(os.getenv("RENDER")),
        "disk": disk,
        "sqlite_journal_mode": journal,
        "scheduler": saudi_ingest.scheduler_info(),
        "lease": saudi_ingest.lease_status(),
        "admin_token_configured": bool(ADMIN_TOKEN),
        "manual_cooldown_seconds": MANUAL_COOLDOWN_SECONDS,
    }


# --------------------------------------------------------------------------- #
# X (Twitter) Public Intelligence Endpoints
# --------------------------------------------------------------------------- #

class XAnalyzeRequest(BaseModel):
    url: str


@router.post("/x/collect-public")
async def collect_x_public_posts(handle: str = "K_H_AlMuzaini") -> Dict[str, Any]:
    """Fetch and ingest public posts from an X profile without credentials using official public syndication."""
    from . import x_public_collector
    report = await x_public_collector.collect_and_store_x_posts(handle=handle)
    return report


@router.post("/x/analyze-post")
async def analyze_x_post(req: XAnalyzeRequest) -> Dict[str, Any]:
    """Fetch, extract, classify and verify a specific public X post link."""
    from . import x_public_collector
    post, status, error = await x_public_collector.fetch_single_public_post(req.url)
    if not post:
        raise HTTPException(
            status_code=400 if status == x_public_collector.STATUS_NOT_SUPPORTED else 502,
            detail=error or "تعذر جلب المنشور",
        )
    return {
        "status": status,
        "post": post.to_dict(),
    }


@router.get("/x/status")
def x_collector_status() -> Dict[str, Any]:
    """Get the current operational status of the X public collector and official integration options."""
    from . import x_public_collector
    con = get_db()
    try:
        row = con.execute("SELECT * FROM source_registry WHERE code = 'x_k_h_almuzaini'").fetchone()
        count = con.execute("SELECT COUNT(*) n FROM articles WHERE source_code = 'x_k_h_almuzaini'").fetchone()["n"]
    finally:
        con.close()

    status_code = row["last_status"] if row and row["last_status"] else x_public_collector.STATUS_ACTIVE
    return {
        "source_code": "x_k_h_almuzaini",
        "handle": "@K_H_AlMuzaini",
        "status": status_code,
        "status_label": x_public_collector.STATUS_LABELS.get(status_code, status_code),
        "status_arabic": x_public_collector.STATUS_ARABIC_LABELS.get(status_code, status_code),
        "last_success_at": row["last_success_at"] if row else None,
        "last_error": row["last_error"] if row else None,
        "total_posts_stored": count,
        "official_api_token_configured": bool(os.environ.get("X_BEARER_TOKEN")),
    }

