# backend/app/saudi_api.py
"""REST API for the Saudi Fraud Intelligence section (read endpoints + manual ingest trigger)."""
from __future__ import annotations

import hmac
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .config import ADMIN_TOKEN, DATA_DIR, DB_PATH, SAUDI_AUTO_INGEST, SAUDI_INGEST_EVERY_HOURS
from .db import get_db
from . import incidents, saudi_ingest
from .fraud_intel import CATEGORY_AR, CATEGORY_ORDER, CONTENT_TYPE_AR, VISIBLE_CONTENT_TYPES

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
        "country": r["country"] if ("country" in r.keys() and r["country"]) else ("SA" if r["is_saudi"] else "Global"),
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
           a.language, a.content, a.country, a.domain
    FROM threat_intel t JOIN articles a ON a.id = t.article_id
"""


def _where(category: Optional[str], source: Optional[str], q: Optional[str],
           date_from: Optional[str], date_to: Optional[str], bank: Optional[str],
           content_type: Optional[str] = None, scope: str = "saudi", country: Optional[str] = None):
    clauses = []
    params: List[Any] = []

    # Scope condition
    if scope == "saudi":
        clauses.append("t.is_saudi = 1")
    elif scope == "global":
        clauses.append("(t.is_saudi = 0 OR t.is_saudi IS NULL)")
    elif scope == "all":
        pass  # Both Saudi and Global
    else:
        clauses.append("t.is_saudi = 1")

    clauses.append("t.fraud_category IS NOT NULL")

    if content_type:
        clauses.append("t.content_type = ?")
        params.append(content_type)
    else:
        clauses.append(f"t.content_type IN ('{_VISIBLE_TYPES}')")

    if country and country != "all":
        if country.upper() == "SA":
            clauses.append("(t.is_saudi = 1 OR a.country = 'SA')")
        else:
            clauses.append("a.country = ?")
            params.append(country.upper())

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

    return (" AND ".join(clauses) if clauses else "1=1"), params


@router.get("/overview")
def overview() -> Dict[str, Any]:
    """Headline indicators. The success metric is INDEPENDENT INCIDENTS, not articles: warnings and
    awareness campaigns are counted separately and general-cybersecurity/unrelated articles are
    excluded from every indicator (they remain stored)."""
    con = get_db()
    try:
        inc = incidents.counts(con)
        total = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t WHERE {_BASE_WHERE}").fetchone()["n"]
        srcs = con.execute(
            f"SELECT COUNT(DISTINCT a.source) n FROM threat_intel t JOIN articles a ON a.id=t.article_id "
            f"WHERE {_BASE_WHERE}").fetchone()["n"]
        last_run = con.execute(
            "SELECT MAX(finished_at) m FROM fetch_log").fetchone()["m"]
        ok_sources = con.execute(
            "SELECT COUNT(*) n FROM source_registry WHERE last_status='ok'").fetchone()["n"]
        all_sources = con.execute("SELECT COUNT(*) n FROM source_registry").fetchone()["n"]
    finally:
        con.close()
    cats = inc["incident_categories"]
    return {
        # --- success metric -------------------------------------------------
        "incidents_total": inc["incidents_total"],
        "incident_articles_total": inc["incident_articles_total"],
        "incidents_last_7_days": inc["incidents_last_7_days"],
        "incidents_high_bank_relevance": inc["incidents_high_bank_relevance"],
        "incidents_confirmed": inc["incidents_confirmed"],
        "incidents_reported": inc["incidents_reported"],
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
        # --- legacy field names (same values, now incident-based) ------------
        "total_items": total,  # visible articles only: confirmed + warning + awareness (NOT incidents)
        "last_7_days": inc["incidents_last_7_days"],
        "high_bank_relevance": inc["incidents_high_bank_relevance"],
        "distinct_sources": srcs,
        "categories": [
            {"key": k, "label_ar": CATEGORY_AR.get(k, k), "count": cats.get(k, 0)}
            for k in CATEGORY_ORDER
        ],
        "last_fetch_at": last_run,
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
    bank_relevance: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    content_type: Optional[str] = Query(
        None, pattern="^(confirmed_incident|reported_incident|fraud_warning|awareness_campaign|general_cybersecurity|unrelated)$"),
    scope: str = Query("saudi", pattern="^(saudi|global|all)$"),
    country: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Articles across Saudi or Global scope. By default only fraud-relevant content types."""
    where, params = _where(category, source, q, date_from, date_to, bank_relevance, content_type, scope=scope, country=country)
    con = get_db()
    try:
        total = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t JOIN articles a ON a.id=t.article_id WHERE {where}",
            params).fetchone()["n"]
        rows = con.execute(
            f"{_SELECT} WHERE {where} ORDER BY COALESCE(a.published_at, a.created_at) DESC, t.id DESC "
            "LIMIT ? OFFSET ?", params + [page_size, (page - 1) * page_size]).fetchall()
    finally:
        con.close()
    return {"page": page, "page_size": page_size, "total": total,
            "items": [_item(r) for r in rows]}


@router.get("/countries")
def fraud_countries(scope: str = Query("all", pattern="^(saudi|global|all)$")) -> Dict[str, Any]:
    """List countries having fraud intelligence records."""
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
    finally:
        con.close()
    return {"countries": [{"country": r["country"], "count": r["count"]} for r in rows if r["country"]]}


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
    q: Optional[str] = None,
    bank_relevance: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    record_type: Optional[str] = Query(None, pattern="^(confirmed_incident|reported_incident)$"),
    review_status: Optional[str] = Query(None, pattern="^(auto_merged|single_source|needs_review|human_confirmed)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Independent fraud incidents (one row per real event), each with ALL the articles covering it."""
    clauses, params = ["1=1"], []
    if category:
        clauses.append("fraud_category = ?")
        params.append(category)
    if bank_relevance:
        clauses.append("bank_relevance = ?")
        params.append(bank_relevance)
    if record_type:
        clauses.append("record_type = ?")
        params.append(record_type)
    if review_status:
        clauses.append("review_status = ?")
        params.append(review_status)
    if q:
        clauses.append("(title LIKE ? OR summary LIKE ? OR locations LIKE ? OR agencies LIKE ?)")
        params += [f"%{q}%"] * 4
    where = " AND ".join(clauses)
    con = get_db()
    try:
        total = con.execute(f"SELECT COUNT(*) n FROM incidents WHERE {where}", params).fetchone()["n"]
        rows = con.execute(
            f"SELECT * FROM incidents WHERE {where} ORDER BY COALESCE(last_reported_at, created_at) DESC, id DESC "
            "LIMIT ? OFFSET ?", params + [page_size, (page - 1) * page_size]).fetchall()
    finally:
        con.close()
    return {"page": page, "page_size": page_size, "total": total,
            "items": [incidents.incident_to_dict(r) for r in rows]}


def _check_admin(x_admin_token: Optional[str]) -> None:
    """Fail-closed admin gate shared by every state-changing endpoint."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Disabled: ADMIN_TOKEN is not configured")
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Invalid admin token")


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
def merge_decision(suggestion_id: int, body: MergeDecision, x_admin_token: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Record a reviewer decision (admin only). The incident set is rebuilt; no article is ever deleted."""
    _check_admin(x_admin_token)
    try:
        return incidents.decide_suggestion(suggestion_id, body.decision, body.reviewer, body.reason)
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
    """Fraud patterns = methods stated in the sources, aggregated over incidents (warnings = weaker support)."""
    con = get_db()
    try:
        pats = incidents.build_patterns(con)
        chan: Dict[str, int] = {}
        for r in con.execute("SELECT channels FROM incidents"):
            for ch in _loads(r["channels"]) or []:
                chan[ch] = chan.get(ch, 0) + 1
    finally:
        con.close()
    return {
        "patterns": pats,
        "channels": [{"channel": k, "count": v} for k, v in sorted(chan.items(), key=lambda kv: -kv[1])],
        "note": "Patterns list only methods whose wording appears in a source article; counts are INCIDENTS, not articles.",
    }


@router.get("/rules")
def rules() -> Dict[str, Any]:
    """Detection rules linked to the incidents/patterns behind them. ``tested`` requires a recorded backtest."""
    con = get_db()
    try:
        items = incidents.build_rules(con)
    finally:
        con.close()
    return {"rules": items,
            "summary": {"total": len(items),
                        "tested": sum(1 for x in items if x["status"] == "tested"),
                        "proposed_untested": sum(1 for x in items if x["status"] == "proposed_untested")},
            "note": "Rules are system-proposed hypotheses (proposed_untested) unless a real test result is recorded; "
                    "the evidence shows the threat exists, not that the rule works."}


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
def ingest_run(x_admin_token: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Manual pull (admin only, fail-closed). Runs in the background; poll /saudi-fraud/ingest/runs."""
    if not ADMIN_TOKEN:
        # Never fail open: without a configured token the endpoint is disabled.
        raise HTTPException(status_code=503, detail="Manual ingest disabled: ADMIN_TOKEN is not configured")
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Invalid admin token")
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
