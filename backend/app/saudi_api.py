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

from .config import ADMIN_TOKEN, DATA_DIR, DB_PATH, SAUDI_AUTO_INGEST, SAUDI_INGEST_EVERY_HOURS
from .db import get_db
from . import saudi_ingest
from .fraud_intel import CATEGORY_AR, CATEGORY_ORDER

router = APIRouter(prefix="/saudi-fraud", tags=["saudi-fraud-intelligence"])

# Saudi fraud item = Saudi-attributed threat that went through the fraud engine
_BASE_WHERE = "t.is_saudi = 1 AND t.fraud_category IS NOT NULL"

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
           a.language, a.content
    FROM threat_intel t JOIN articles a ON a.id = t.article_id
"""


def _where(category: Optional[str], source: Optional[str], q: Optional[str],
           date_from: Optional[str], date_to: Optional[str], bank: Optional[str]):
    clauses = [_BASE_WHERE]
    params: List[Any] = []
    if category:
        clauses.append("t.fraud_category = ?")
        params.append(category)
    if source:
        clauses.append("(a.source_code = ? OR a.source = ?)")
        params += [source, source]
    if bank:
        clauses.append("t.bank_relevance = ?")
        params.append(bank)
    if q:
        clauses.append("(t.title LIKE ? OR t.description LIKE ? OR a.headline LIKE ?)")
        params += [f"%{q}%"] * 3
    if date_from:
        clauses.append("a.published_at >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("a.published_at <= ?")
        params.append(date_to + "T23:59:59" if len(date_to) == 10 else date_to)
    return " AND ".join(clauses), params


@router.get("/overview")
def overview() -> Dict[str, Any]:
    con = get_db()
    try:
        total = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t WHERE {_BASE_WHERE}").fetchone()["n"]
        cats = {r["fraud_category"]: r["n"] for r in con.execute(
            f"SELECT t.fraud_category, COUNT(*) n FROM threat_intel t WHERE {_BASE_WHERE} "
            "GROUP BY t.fraud_category")}
        last7 = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t JOIN articles a ON a.id=t.article_id "
            f"WHERE {_BASE_WHERE} AND a.published_at >= datetime('now','-7 days')").fetchone()["n"]
        high_bank = con.execute(
            f"SELECT COUNT(*) n FROM threat_intel t WHERE {_BASE_WHERE} AND t.bank_relevance='high'"
        ).fetchone()["n"]
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
    return {
        "total_items": total,
        "last_7_days": last7,
        "high_bank_relevance": high_bank,
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
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    where, params = _where(category, source, q, date_from, date_to, bank_relevance)
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


@router.get("/news/{threat_id}")
def news_detail(threat_id: int) -> Dict[str, Any]:
    con = get_db()
    try:
        r = con.execute(f"{_SELECT} WHERE t.id = ? AND {_BASE_WHERE}", (threat_id,)).fetchone()
    finally:
        con.close()
    if not r:
        raise HTTPException(status_code=404, detail="Saudi fraud item not found")
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


@router.get("/patterns")
def patterns() -> Dict[str, Any]:
    """Aggregated fraud patterns (category / modus operandi / channels) across stored items."""
    con = get_db()
    try:
        rows = con.execute(
            f"SELECT t.fraud_category, t.modus_operandi, t.channels FROM threat_intel t WHERE {_BASE_WHERE}"
        ).fetchall()
    finally:
        con.close()
    by_cat: Dict[str, Dict[str, Any]] = {}
    channels: Dict[str, int] = {}
    for r in rows:
        cat = r["fraud_category"]
        e = by_cat.setdefault(cat, {"key": cat, "label_ar": CATEGORY_AR.get(cat, cat),
                                    "count": 0, "modus_operandi": r["modus_operandi"]})
        e["count"] += 1
        for ch in _loads(r["channels"]) or []:
            channels[ch] = channels.get(ch, 0) + 1
    return {
        "patterns": sorted(by_cat.values(), key=lambda x: -x["count"]),
        "channels": [{"channel": k, "count": v} for k, v in sorted(channels.items(), key=lambda kv: -kv[1])],
    }


@router.get("/rules")
def rules() -> Dict[str, Any]:
    """Proposed (untested) detection rules, one per fraud category, with supporting evidence count."""
    con = get_db()
    try:
        rows = con.execute(
            f"SELECT t.id, t.fraud_category, t.detection_rule, t.required_data, t.title "
            f"FROM threat_intel t WHERE {_BASE_WHERE} ORDER BY t.id DESC"
        ).fetchall()
    finally:
        con.close()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cat = r["fraud_category"]
        e = out.get(cat)
        if e is None:
            e = out[cat] = {
                "category": cat, "category_ar": CATEGORY_AR.get(cat, cat),
                "rule": _loads(r["detection_rule"]), "required_data": _loads(r["required_data"]),
                "supporting_items": 0, "example_threat_ids": [],
            }
        e["supporting_items"] += 1
        if len(e["example_threat_ids"]) < 5:
            e["example_threat_ids"].append(r["id"])
    return {"rules": list(out.values()),
            "note": "Rules are system-proposed hypotheses (status: proposed_untested), not validated detections."}


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
