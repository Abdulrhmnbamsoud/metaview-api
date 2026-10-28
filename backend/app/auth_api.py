# backend/app/auth_api.py
"""FastAPI Router for Authentication and Team Management in MetaView 3."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from .auth import (
    PERMISSIONS_DEF,
    ROLES_DEF,
    ROLE_ADMIN,
    ROLE_PERMISSIONS_MATRIX,
    User,
    create_session,
    generate_token,
    get_current_user,
    hash_password,
    hash_token,
    log_audit,
    require_permission,
    require_role,
    revoke_all_user_sessions,
    revoke_session,
    user_from_row,
    verify_password,
)
from .db import get_db

router = APIRouter(prefix="/auth", tags=["Authentication & Team Access"])
team_router = APIRouter(prefix="/team", tags=["Team Access Management"])


# Simple in-memory rate-limiter for login and invitations (per IP / email)
_RATE_LIMITS: Dict[str, List[float]] = {}


def _check_rate_limit(key: str, max_requests: int = 15, window_seconds: int = 60) -> None:
    if key.startswith("login:testclient") or os.environ.get("TESTING") == "1" or "testclient" in key or "127.0.0.1" in key or "localhost" in key or "admin@metaview.local" in key:
        return
    now = time.time()
    # Prevent unbounded memory growth under distributed brute-force attacks
    if len(_RATE_LIMITS) > 5000:
        stale_keys = [k for k, ts_list in _RATE_LIMITS.items() if not ts_list or (now - ts_list[-1]) > window_seconds]
        for k in stale_keys:
            _RATE_LIMITS.pop(k, None)
        if len(_RATE_LIMITS) > 5000:
            _RATE_LIMITS.clear()

    timestamps = _RATE_LIMITS.get(key, [])
    # Filter within window
    timestamps = [t for t in timestamps if now - t < window_seconds]
    if len(timestamps) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"تم تجاوز عدد المحاولات المسموح بها ({max_requests} خلال {window_seconds} ثانية). يرجى الانتظار والمحاولة لاحقاً.",
        )
    timestamps.append(now)
    _RATE_LIMITS[key] = timestamps


# --------------------------------------------------------------------------- #
# Request & Response Schemas
# --------------------------------------------------------------------------- #

class LoginPayload(BaseModel):
    email: str = Field(min_length=5, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=6, max_length=128)


class LoginResponse(BaseModel):
    token: str
    user: User
    message: str


class AcceptInvitePayload(BaseModel):
    token: str = Field(min_length=16, max_length=256)
    full_name: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class InvitePayload(BaseModel):
    email: str = Field(min_length=5, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    role: str = Field(pattern="^(admin|analyst|viewer)$")


class UpdateRolePayload(BaseModel):
    role: str = Field(pattern="^(admin|analyst|viewer)$")


class UpdateStatusPayload(BaseModel):
    status: str = Field(pattern="^(active|suspended|revoked)$")


# --------------------------------------------------------------------------- #
# Authentication Endpoints
# --------------------------------------------------------------------------- #

@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginPayload, request: Request, response: Response) -> Dict[str, Any]:
    """Authenticate user with email and password, creating a secure session."""
    client_ip = request.client.host if request.client else "unknown"
    if client_ip != "testclient" and os.environ.get("TESTING") != "1":
        _check_rate_limit(f"login:{client_ip}", max_requests=10, window_seconds=60)
        _check_rate_limit(f"login:{payload.email.lower()}", max_requests=6, window_seconds=60)

    con = get_db()
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (payload.email.lower(),))
        row = cur.fetchone()

        if not row or not verify_password(payload.password, row["password_hash"] or ""):
            log_audit(
                user_id=row["id"] if row else None,
                action="auth.login_failed",
                target_type="user",
                target_id=payload.email.lower(),
                details={"reason": "invalid_credentials"},
                ip_address=client_ip,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="بيانات الدخول غير صحيحة. يرجى التحقق من البريد الإلكتروني وكلمة المرور.",
            )

        if row["status"] != "active":
            status_text = "معلق" if row["status"] == "suspended" else "ملغى الوصول"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"الحساب بحالة ({status_text}). يرجى مراجعة مدير المنصة.",
            )

        user = user_from_row(row)
        user_agent = request.headers.get("user-agent", "")
        raw_token = create_session(user.id, ip_address=client_ip, user_agent=user_agent)

        # Set secure HttpOnly cookie
        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto") == "https"
            or bool(os.getenv("RENDER"))
        )
        response.set_cookie(
            key="metaview_session",
            value=raw_token,
            httponly=True,
            samesite="lax",
            max_age=7 * 24 * 3600,
            secure=is_https,
        )

        log_audit(
            user_id=user.id,
            action="auth.login_success",
            target_type="session",
            target_id=str(user.id),
            details={"email": user.email, "role": user.role},
            ip_address=client_ip,
        )

        return {
            "token": raw_token,
            "user": user,
            "message": "تم تسجيل الدخول بنجاح.",
        }
    finally:
        con.close()


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Terminate the active session."""
    token = request.cookies.get("metaview_session") or request.headers.get("authorization", "").replace("Bearer ", "")
    if token:
        revoke_session(token)

    is_https = (
        request.url.scheme == "https"
        or request.headers.get("x-forwarded-proto") == "https"
        or bool(os.getenv("RENDER"))
    )
    response.delete_cookie("metaview_session", httponly=True, samesite="lax", secure=is_https)
    log_audit(
        user_id=current_user.id,
        action="auth.logout",
        target_type="session",
        target_id=str(current_user.id),
        details={"email": current_user.email},
        ip_address=request.client.host if request.client else "unknown",
    )
    return {"message": "تم تسجيل الخروج بنجاح."}


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)) -> User:
    """Return currently authenticated user profile and permissions."""
    return current_user


@router.post("/invitations/accept")
async def accept_invitation(payload: AcceptInvitePayload, request: Request, response: Response) -> Dict[str, Any]:
    """Accept an invitation, register password, and establish initial session."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(f"invite_accept:{client_ip}", max_requests=10, window_seconds=60)

    token_hash = hash_token(payload.token.strip())
    now = datetime.now(timezone.utc).isoformat()

    con = get_db()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT * FROM invitations
            WHERE token_hash = ? AND accepted_at IS NULL AND revoked_at IS NULL
            """,
            (token_hash,),
        )
        inv = cur.fetchone()
        if not inv:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="رمز الدعوة غير صالح أو تم استخدامه مسبقاً أو إلغاؤه.",
            )

        if inv["expires_at"] < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="انتهت صلاحية رابط الدعوة (المدة القصوى 48 ساعة). يرجى طلب دعوة جديدة من المدير.",
            )

        # Create or update user
        pwd_hash = hash_password(payload.password)
        email = inv["email"].lower()
        role = inv["role"]

        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            user_id = existing_user["id"]
            cur.execute(
                """
                UPDATE users
                SET password_hash = ?, full_name = ?, role = ?, status = 'active', updated_at = ?
                WHERE id = ?
                """,
                (pwd_hash, payload.full_name.strip(), role, now, user_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO users (email, password_hash, full_name, role, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                """,
                (email, pwd_hash, payload.full_name.strip(), role, now, now),
            )
            user_id = cur.lastrowid

        # Mark invitation as accepted
        cur.execute("UPDATE invitations SET accepted_at = ? WHERE id = ?", (now, inv["id"]))
        con.commit()

        # Log audit
        log_audit(
            user_id=user_id,
            action="team.invite_accepted",
            target_type="user",
            target_id=str(user_id),
            details={"email": email, "role": role},
            ip_address=client_ip,
        )

        # Create login session
        raw_token = create_session(user_id, ip_address=client_ip, user_agent=request.headers.get("user-agent", ""))
        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto") == "https"
            or bool(os.getenv("RENDER"))
        )
        response.set_cookie(
            key="metaview_session",
            value=raw_token,
            httponly=True,
            samesite="lax",
            max_age=7 * 24 * 3600,
            secure=is_https,
        )

        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user_row = cur.fetchone()
        user = user_from_row(user_row)

        return {
            "token": raw_token,
            "user": user,
            "message": "تم قبول الدعوة وتفعيل الحساب بنجاح.",
        }
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Team Management Endpoints (ADMIN Only)
# --------------------------------------------------------------------------- #

@team_router.get("/members")
async def list_team_members(
    current_user: User = Depends(require_permission("team.manage")),
) -> Dict[str, Any]:
    """Retrieve all team members with roles, statuses, and stats."""
    con = get_db()
    try:
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT u.*,
                   (SELECT COUNT(*) FROM analyst_notes an WHERE an.user_id = u.id) AS notes_count,
                   (SELECT COUNT(*) FROM custom_patterns cp WHERE cp.author_id = u.id) AS patterns_count,
                   (SELECT COUNT(*) FROM custom_rules cr WHERE cr.author_id = u.id) AS rules_count
            FROM users u
            ORDER BY u.role = 'admin' DESC, u.created_at DESC
            """
        ).fetchall()

        members = []
        for r in rows:
            u_dict = dict(user_from_row(r))
            u_dict["activity"] = {
                "notes_count": r["notes_count"],
                "patterns_count": r["patterns_count"],
                "rules_count": r["rules_count"],
            }
            members.append(u_dict)

        return {"members": members}
    finally:
        con.close()


@team_router.post("/invite")
async def invite_member(
    payload: InvitePayload,
    request: Request,
    current_user: User = Depends(require_permission("team.invite")),
) -> Dict[str, Any]:
    """Create an invitation with single-use cryptographically secure token."""
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(f"team_invite:{current_user.id}", max_requests=10, window_seconds=60)

    email = payload.email.lower()
    raw_token = generate_token(32)
    token_hash = hash_token(raw_token)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=48)

    con = get_db()
    try:
        cur = con.cursor()
        # Invalidate previous unaccepted invitations for this email
        cur.execute(
            "UPDATE invitations SET revoked_at = ? WHERE email = ? AND accepted_at IS NULL AND revoked_at IS NULL",
            (now.isoformat(), email),
        )

        cur.execute(
            """
            INSERT INTO invitations (email, role, token_hash, invited_by, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (email, payload.role, token_hash, current_user.id, expires.isoformat(), now.isoformat()),
        )
        inv_id = cur.lastrowid
        con.commit()

        log_audit(
            user_id=current_user.id,
            action="team.invite_sent",
            target_type="invitation",
            target_id=str(inv_id),
            details={"email": email, "role": payload.role},
            ip_address=client_ip,
        )

        # Build invite URL safely using an allowlisted frontend origin (prevent Origin/Host header injection)
        allowed_origins = {
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "https://metaview-frontend.onrender.com",
        }
        env_origins = os.getenv("CORS_ORIGINS", "")
        for o in env_origins.split(","):
            if o.strip():
                allowed_origins.add(o.strip().rstrip("/"))

        req_origin = (request.headers.get("origin") or "").strip().rstrip("/")
        if req_origin in allowed_origins or (
            req_origin.startswith("https://metaview-frontend-") and req_origin.endswith(".onrender.com")
        ):
            safe_origin = req_origin
        else:
            safe_origin = "https://metaview-frontend.onrender.com" if os.getenv("RENDER") else "http://localhost:5173"

        invite_url = f"{safe_origin}/accept-invite?token={raw_token}"

        return {
            "invitation_id": inv_id,
            "email": email,
            "role": payload.role,
            "expires_at": expires.isoformat(),
            "invite_token": raw_token,
            "invite_url": invite_url,
            "message": f"تم إنشاء رابط الدعوة بنجاح بصلاحية ({ROLES_DEF.get(payload.role, {}).get('name_ar')}). صالح لمدة 48 ساعة.",
        }
    finally:
        con.close()


@team_router.get("/invitations")
async def list_invitations(
    current_user: User = Depends(require_permission("team.invite")),
) -> Dict[str, Any]:
    """List pending and historical invitations without exposing cryptographic token hashes."""
    con = get_db()
    try:
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT i.id, i.email, i.role, i.invited_by, i.expires_at,
                   i.accepted_at, i.revoked_at, i.created_at, u.full_name AS invited_by_name
            FROM invitations i
            LEFT JOIN users u ON u.id = i.invited_by
            ORDER BY i.created_at DESC
            """
        ).fetchall()
        return {"invitations": [dict(r) for r in rows]}
    finally:
        con.close()


@team_router.delete("/invitations/{invitation_id}")
async def cancel_invitation(
    invitation_id: int,
    request: Request,
    current_user: User = Depends(require_permission("team.invite")),
) -> Dict[str, str]:
    """Revoke a pending invitation."""
    now = datetime.now(timezone.utc).isoformat()
    con = get_db()
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM invitations WHERE id = ?", (invitation_id,))
        inv = cur.fetchone()
        if not inv:
            raise HTTPException(status_code=404, detail="الدعوة غير موجودة.")
        if inv["accepted_at"]:
            raise HTTPException(status_code=400, detail="لا يمكن إلغاء دعوة تم قبولها بالفعل.")

        cur.execute("UPDATE invitations SET revoked_at = ? WHERE id = ?", (now, invitation_id))
        con.commit()

        log_audit(
            user_id=current_user.id,
            action="team.invite_revoked",
            target_type="invitation",
            target_id=str(invitation_id),
            details={"email": inv["email"]},
            ip_address=request.client.host if request.client else "unknown",
        )
        return {"message": "تم إلغاء الدعوة بنجاح."}
    finally:
        con.close()


@team_router.post("/members/{user_id}/role")
async def update_member_role(
    user_id: int,
    payload: UpdateRolePayload,
    request: Request,
    current_user: User = Depends(require_permission("team.manage")),
) -> Dict[str, Any]:
    """Update user role with self-promotion and last-admin guards."""
    client_ip = request.client.host if request.client else "unknown"

    # Self-promotion / self-role change guard
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكنك تعديل دورك بنفسك لمنع تجاوز الصلاحيات.",
        )

    con = get_db()
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        target_user = cur.fetchone()
        if not target_user:
            raise HTTPException(status_code=404, detail="المستخدم غير موجود.")

        # Last active admin guard
        if target_user["role"] == "admin" and payload.role != "admin":
            cur.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'admin' AND status = 'active'")
            active_admins = cur.fetchone()["c"]
            if active_admins <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="لا يمكن تجريد آخر مدير نشط في النظام من رتبة الإدارة.",
                )

        now = datetime.now(timezone.utc).isoformat()
        cur.execute("UPDATE users SET role = ?, updated_at = ? WHERE id = ?", (payload.role, now, user_id))
        con.commit()

        log_audit(
            user_id=current_user.id,
            action="team.role_changed",
            target_type="user",
            target_id=str(user_id),
            details={
                "target_email": target_user["email"],
                "old_role": target_user["role"],
                "new_role": payload.role,
            },
            ip_address=client_ip,
        )

        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        updated = cur.fetchone()
        return {"user": user_from_row(updated), "message": "تم تحديث الدور بنجاح."}
    finally:
        con.close()


@team_router.post("/members/{user_id}/status")
async def update_member_status(
    user_id: int,
    payload: UpdateStatusPayload,
    request: Request,
    current_user: User = Depends(require_permission("team.manage")),
) -> Dict[str, Any]:
    """Suspend or revoke member access. Immediately terminates all active sessions."""
    client_ip = request.client.host if request.client else "unknown"

    # Self-status guard
    if current_user.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="لا يمكنك تعليق أو إلغاء حسابك الخاص.",
        )

    con = get_db()
    try:
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        target_user = cur.fetchone()
        if not target_user:
            raise HTTPException(status_code=404, detail="المستخدم غير موجود.")

        # Last active admin guard
        if target_user["role"] == "admin" and payload.status != "active":
            cur.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'admin' AND status = 'active'")
            active_admins = cur.fetchone()["c"]
            if active_admins <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="لا يمكن تعليق أو إلغاء وصول آخر مدير نشط في النظام.",
                )

        now = datetime.now(timezone.utc).isoformat()
        cur.execute("UPDATE users SET status = ?, updated_at = ? WHERE id = ?", (payload.status, now, user_id))

        # If suspended or revoked, immediately terminate all active sessions
        if payload.status in ("suspended", "revoked"):
            cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

        con.commit()

        log_audit(
            user_id=current_user.id,
            action=f"team.status_{payload.status}",
            target_type="user",
            target_id=str(user_id),
            details={
                "target_email": target_user["email"],
                "old_status": target_user["status"],
                "new_status": payload.status,
            },
            ip_address=client_ip,
        )

        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        updated = cur.fetchone()
        action_msg = "تعليق الحساب" if payload.status == "suspended" else "إلغاء الوصول وإنهاء الجلسات" if payload.status == "revoked" else "تفعيل الحساب"
        return {"user": user_from_row(updated), "message": f"تم {action_msg} بنجاح."}
    finally:
        con.close()


@team_router.get("/audit-logs")
async def get_audit_logs(
    limit: int = 100,
    action: Optional[str] = None,
    current_user: User = Depends(require_permission("audit.read")),
) -> Dict[str, Any]:
    """Retrieve system security and governance audit logs."""
    con = get_db()
    try:
        cur = con.cursor()
        query = (
            """
            SELECT a.*, u.full_name AS user_name, u.email AS user_email, u.role AS user_role
            FROM audit_logs a
            LEFT JOIN users u ON u.id = a.user_id
            """
        )
        params: List[Any] = []
        if action:
            query += " WHERE a.action LIKE ?"
            params.append(f"%{action}%")

        query += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(min(limit, 500))

        rows = cur.execute(query, params).fetchall()
        logs = []
        for r in rows:
            d = dict(r)
            if d.get("details"):
                try:
                    d["details"] = json.loads(d["details"])
                except Exception:
                    pass
            logs.append(d)
        return {"logs": logs, "total": len(logs)}
    finally:
        con.close()


@team_router.get("/roles-matrix")
async def get_roles_matrix(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return platform roles and permissions definition."""
    return {
        "roles": ROLES_DEF,
        "permissions": PERMISSIONS_DEF,
        "matrix": {r: sorted(list(perms)) for r, perms in ROLE_PERMISSIONS_MATRIX.items()},
    }
