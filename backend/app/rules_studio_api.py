# backend/app/rules_studio_api.py
"""MetaView 3 — Unified Fraud Detection Intelligence & Rule Deployment Studio API.

Endpoints:
- GET  /rules-studio/overview
- GET  /rules-studio/rules
- GET  /rules-studio/rules/{rule_id}
- POST /rules-studio/parse-scenario
- POST /rules-studio/rules
- POST /rules-studio/rules/{rule_id}/simulate
- GET  /rules-studio/environments
- POST /rules-studio/recommend-deployment
- POST /rules-studio/rules/{rule_id}/governance
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .auth import User, require_permission
from .db import get_db
from .rules_studio_engine import (
    ENV_BRANCH,
    ENV_HYBRID,
    ENV_SAS,
    ENV_SHARED_ROOM,
    SYSTEM_CAPABILITY_REGISTRY,
    advise_deployment,
    discover_intelligence_rules,
    execute_rule_simulation,
    get_all_unified_rules,
    get_rule_by_id,
    parse_natural_language_scenario,
    save_or_update_rule,
)

log = logging.getLogger("metaview.rules_studio_api")
router = APIRouter(tags=["rules-studio"])


# --------------------------------------------------------------------------- #
# Request & Response Schemas
# --------------------------------------------------------------------------- #

class ScenarioParseRequest(BaseModel):
    scenario_text: str = Field(..., min_length=5, max_length=5000, description="Natural language scenario in Arabic or English")


class RuleSaveRequest(BaseModel):
    rule: Dict[str, Any]
    change_summary: Optional[str] = Field(default="تحديث بيانات القاعدة", max_length=500)
    change_reason: Optional[str] = Field(default="مراجعة وتطوير منطق الكشف", max_length=500)


class SimulationRequest(BaseModel):
    custom_cases: Optional[List[Dict[str, Any]]] = None


class GovernanceUpdateRequest(BaseModel):
    new_status: str = Field(..., pattern="^(draft|evidence_review|logic_review|simulation|ready_for_bank_testing|approved_externally|rejected|needs_revision)$")
    reason: str = Field(..., min_length=3, max_length=1000)
    reviewer_name: Optional[str] = Field(default="المراجع المعتمد", max_length=120)
    external_approval: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@router.get("/rules-studio/overview")
def get_rules_overview() -> Dict[str, Any]:
    """Summary metrics of all rules in the unified studio (Public Read-Only)."""
    all_rules = get_all_unified_rules()
    
    total = len(all_rules)
    origin_counts = {"intelligence": 0, "analyst": 0, "hybrid": 0}
    status_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    env_counts: Dict[str, int] = {}
    
    testable_count = 0
    needing_data_count = 0
    awaiting_review_count = 0

    for r in all_rules:
        # Origin
        orig = r.get("origin", "intelligence")
        origin_counts[orig] = origin_counts.get(orig, 0) + 1

        # Status
        st = r.get("status", "draft")
        status_counts[st] = status_counts.get(st, 0) + 1

        # Category
        cat = r.get("fraud_category_ar") or r.get("fraud_category", "غير محدد")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # Recommended Env
        env = r.get("deployment", {}).get("recommended_environment", ENV_SAS)
        env_counts[env] = env_counts.get(env, 0) + 1

        # Testable
        if st in ("simulation", "ready_for_bank_testing", "approved_externally"):
            testable_count += 1

        # Needing data
        missing = r.get("deployment", {}).get("missing_capabilities", [])
        if missing or any(f.get("availability") == "low" for f in r.get("data_requirements", {}).get("required_fields", [])):
            needing_data_count += 1

        # Awaiting review
        if st in ("draft", "evidence_review", "logic_review", "needs_revision"):
            awaiting_review_count += 1

    return {
        "total_rules": total,
        "testable_rules": testable_count,
        "needing_data_rules": needing_data_count,
        "awaiting_review_rules": awaiting_review_count,
        "origin_breakdown": origin_counts,
        "status_breakdown": status_counts,
        "category_breakdown": category_counts,
        "environment_breakdown": env_counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/rules-studio/rules")
def list_rules(
    origin: Optional[str] = Query("all", pattern="^(all|intelligence|analyst|hybrid)$"),
    category: Optional[str] = Query("all"),
    status: Optional[str] = Query("all"),
    environment: Optional[str] = Query("all"),
    q: Optional[str] = None,
) -> Dict[str, Any]:
    """List filtered unified rules (Public Read-Only)."""
    rules = get_all_unified_rules(
        origin_filter=origin if origin != "all" else None,
        category_filter=category if category != "all" else None,
        status_filter=status if status != "all" else None,
    )

    if environment and environment != "all":
        rules = [r for r in rules if r.get("deployment", {}).get("recommended_environment") == environment]

    if q and q.strip():
        query = q.strip().lower()
        rules = [
            r for r in rules
            if query in r.get("name", "").lower()
            or query in r.get("rule_id", "").lower()
            or query in r.get("detection_hypothesis", "").lower()
            or query in r.get("fraud_category_ar", "").lower()
        ]

    return {
        "rules": rules,
        "total": len(rules),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/rules-studio/rules/{rule_id}")
def get_rule_details(rule_id: str) -> Dict[str, Any]:
    """Get full details of a single unified rule (Public Read-Only)."""
    rule = get_rule_by_id(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"قاعدة الكشف [{rule_id}] غير موجودة.")
    return {"rule": rule}


@router.post("/rules-studio/parse-scenario")
def parse_scenario(
    payload: ScenarioParseRequest,
    current_user: User = Depends(require_permission("rule.create")),
) -> Dict[str, Any]:
    """Natural Language Scenario Builder: requires analyst or admin permission."""
    try:
        draft_rule = parse_natural_language_scenario(payload.scenario_text)
        return {
            "status": "success",
            "extracted_scenario": draft_rule.get("scenario_extraction"),
            "proposed_rule": draft_rule,
        }
    except Exception as e:
        log.exception("Scenario parsing error: %s", e)
        raise HTTPException(status_code=400, detail="فشل استخراج عناصر السيناريو. يرجى التحقق من النص المدخل والمحاولة مجدداً.")


@router.post("/rules-studio/rules")
def save_rule(
    payload: RuleSaveRequest,
    current_user: User = Depends(require_permission("rule.create")),
) -> Dict[str, Any]:
    """Save or update a rule draft with versioning (analyst or admin only)."""
    try:
        res = save_or_update_rule(
            rule_dict=payload.rule,
            user_id=current_user.id,
            user_name=current_user.full_name,
            change_summary=payload.change_summary,
            change_reason=payload.change_reason,
        )
        return res
    except Exception as e:
        log.exception("Failed to save rule: %s", e)
        raise HTTPException(status_code=500, detail="تعذر حفظ القاعدة. تم تسجيل الخطأ للمراجعة التقنية.")


@router.post("/rules-studio/rules/{rule_id}/simulate")
def run_simulation(
    rule_id: str,
    payload: Optional[SimulationRequest] = None,
    current_user: User = Depends(require_permission("rule.test")),
) -> Dict[str, Any]:
    """Execute the 6-case logic simulation suite for the rule (analyst or admin only)."""
    rule = get_rule_by_id(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"قاعدة الكشف [{rule_id}] غير موجودة.")

    custom_cases = payload.custom_cases if payload else None
    results = execute_rule_simulation(rule, custom_cases=custom_cases)

    # If all logic passed and status is draft, advance status to simulation
    if results.get("all_logic_passed") and rule.get("status") == "draft":
        rule["status"] = "simulation"
        save_or_update_rule(
            rule,
            user_id=current_user.id,
            user_name=current_user.full_name,
            change_summary="اجتياز المحاكاة المنطقية للسيناريوهات الستة بنجاح."
        )

    return results


@router.get("/rules-studio/environments")
def get_environments_registry() -> Dict[str, Any]:
    """Return the System Capability Registry with real technical capabilities & limitations (Public)."""
    return {
        "environments": SYSTEM_CAPABILITY_REGISTRY,
        "total": len(SYSTEM_CAPABILITY_REGISTRY),
        "disclaimer_ar": "سجل إمكانات الأنظمة المعرف يعكس القدرات الفعلية لكل بيئة؛ أي حقل أو خاصية غير معرّفة تظهر بحالة 'معلومات غير كافية' لمنع التوصيات غير الموثوقة.",
    }


@router.post("/rules-studio/recommend-deployment")
def evaluate_deployment(
    payload: Dict[str, Any],
    current_user: User = Depends(require_permission("rule.create")),
) -> Dict[str, Any]:
    """Advise the best deployment environment based on verified capabilities (analyst or admin)."""
    rule = payload.get("rule", payload)
    advice = advise_deployment(rule)
    return advice


@router.post("/rules-studio/rules/{rule_id}/governance")
def update_rule_governance(
    rule_id: str,
    payload: GovernanceUpdateRequest,
    current_user: User = Depends(require_permission("rule.approve")),
) -> Dict[str, Any]:
    """Update rule lifecycle status with audit justification (Admin only).
    
    CRITICAL INVARIANT:
    'approved_externally' CANNOT be granted automatically without external documented reference.
    """
    rule = get_rule_by_id(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"قاعدة الكشف [{rule_id}] غير موجودة.")

    if payload.new_status == "approved_externally":
        if not payload.external_approval or not payload.external_approval.get("authority") or not payload.external_approval.get("reference_doc"):
            raise HTTPException(
                status_code=400,
                detail="حالة 'Approved Externally' لا تُمنح تلقائياً؛ يجب توفير وثيقة الاعتماد والجهة المخولة (Authority and Reference Doc)."
            )

    old_status = rule.get("status", "draft")
    rule["status"] = payload.new_status
    gov = rule.get("governance", {})
    history = gov.get("version_history", [])
    now = datetime.now(timezone.utc).isoformat()

    history.append({
        "version": gov.get("current_version", "1.0.0"),
        "timestamp": now,
        "author": current_user.full_name or payload.reviewer_name or "المراجع",
        "change_summary": f"تغيير حالة الحوكمة من [{old_status}] إلى [{payload.new_status}].",
        "reason": payload.reason
    })
    gov["version_history"] = history
    gov["updated_at"] = now

    if payload.new_status == "approved_externally":
        gov["external_approval"] = {
            "authority": payload.external_approval.get("authority"),
            "reference_doc": payload.external_approval.get("reference_doc"),
            "approved_at": now,
            "notes": payload.external_approval.get("notes", "اعتماد خارجي موثق")
        }

    rule["governance"] = gov
    save_or_update_rule(
        rule,
        user_id=current_user.id,
        user_name=current_user.full_name,
        change_summary=f"تحديث حالة الحوكمة إلى {payload.new_status}",
        change_reason=payload.reason
    )

    return {
        "status": "success",
        "rule_id": rule_id,
        "old_status": old_status,
        "new_status": payload.new_status,
        "updated_at": now,
        "governance": gov
    }
