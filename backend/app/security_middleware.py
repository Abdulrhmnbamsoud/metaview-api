# backend/app/security_middleware.py
"""
MetaView 3 — Banking-Grade Security Middleware & Policy Enforcement.

Features:
1. Standard Banking Security Headers (OWASP / SAMA Recommended):
   - X-Content-Type-Options: nosniff
   - X-Frame-Options: DENY
   - Referrer-Policy: strict-origin-when-cross-origin
   - Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()
   - Strict-Transport-Security: max-age=31536000; includeSubDomains (when HTTPS)
   - Content-Security-Policy: restrict origins, disallow object/frame embeds
2. Request Body Size Limiter:
   - Rejects oversized requests (> 5MB) with HTTP 413 Payload Too Large.
3. In-Memory Sliding Window Rate Limiter:
   - Protects against brute-force, scraping, and DoS.
4. Security Audit Event Logger:
   - Logs security-relevant rejections without sensitive data.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("metaview.security")

# Maximum permitted upload payload (5 MB)
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

# Rate limit defaults
RATE_LIMIT_MUTATING = (30, 60)   # 30 requests per 60 seconds
RATE_LIMIT_GENERAL = (300, 60)   # 300 requests per 60 seconds

_IP_REQUEST_LOGS: Dict[str, List[float]] = {}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces strict HTTP security headers on all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # 1. MIME Sniffing Prevention
        response.headers["X-Content-Type-Options"] = "nosniff"

        # 2. Clickjacking Protection
        response.headers["X-Frame-Options"] = "DENY"

        # 3. Referrer Information Control
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 4. Restrict Hardware/Browser Features
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"

        # 5. Cross-Domain policies & Browser Isolation
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["X-DNS-Prefetch-Control"] = "off"

        # 6. HSTS (Strict-Transport-Security) for HTTPS / Cloud Production
        is_https = (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto") == "https"
            or bool(os.getenv("RENDER"))
        )
        if is_https:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # 7. Content Security Policy (Banking-safe, allowing internal API & static fonts)
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com data:",
            "img-src 'self' data: https: blob:",
            "connect-src 'self' https://metaview-backend.onrender.com https://metaview-frontend.onrender.com http://localhost:8000 http://localhost:5173",
            "object-src 'none'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # 8. Dynamic API Cache-Control (prevent stale 304 or browser caching of analytical or sensitive data)
        path = request.url.path
        if (
            path.startswith("/api")
            or path.startswith("/auth")
            or path.startswith("/team")
            or path.startswith("/workspace")
            or path.startswith("/fraud-intelligence")
            or path.startswith("/saudi-fraud")
            or path.startswith("/rules-studio")
            or path.startswith("/governance")
            or path in [
                "/health",
                "/stats",
                "/sources",
                "/countries",
                "/overview",
                "/news",
                "/incidents",
                "/patterns",
                "/rules",
                "/threats",
                "/search-text",
                "/telemetry",
            ]
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"

        return response



class RequestSizeLimiterMiddleware(BaseHTTPMiddleware):
    """Protects server memory by rejecting oversized request payloads before reading."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > MAX_CONTENT_LENGTH:
                    log.warning(
                        "Payload rejected due to size: %d bytes from IP %s",
                        length,
                        request.client.host if request.client else "unknown"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "حجم الطلب يتجاوز الحد الأقصى المسموح به (5 ميجابايت)."}
                    )
            except ValueError:
                pass

        return await call_next(request)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Sliding-window IP rate limiter distinguishing mutating vs read operations."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Bypass for test clients or health probes
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        if (
            os.environ.get("TESTING") == "1"
            or client_ip in ("testclient", "127.0.0.1", "localhost")
            or path in ("/health", "/favicon.ico")
        ):
            return await call_next(request)

        is_mutating = request.method in ("POST", "PUT", "DELETE", "PATCH")
        max_req, window = RATE_LIMIT_MUTATING if is_mutating else RATE_LIMIT_GENERAL

        key = f"{client_ip}:{'mut' if is_mutating else 'read'}"
        now = time.time()

        # Prevent unbounded memory growth under distributed IP floods
        if len(_IP_REQUEST_LOGS) > 10000:
            stale = [k for k, ts in _IP_REQUEST_LOGS.items() if not ts or (now - ts[-1]) > window]
            for k in stale:
                _IP_REQUEST_LOGS.pop(k, None)
            if len(_IP_REQUEST_LOGS) > 10000:
                _IP_REQUEST_LOGS.clear()

        timestamps = _IP_REQUEST_LOGS.get(key, [])
        # Evict timestamps outside the sliding window
        timestamps = [t for t in timestamps if now - t < window]

        if len(timestamps) >= max_req:
            log.warning("Rate limit exceeded for %s on path %s (%d requests)", key, path, len(timestamps))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"تم تجاوز معدل الطلبات المسموح به ({max_req} طلبات خلال {window} ثانية). يرجى المحاولة لاحقاً."
                },
                headers={"Retry-After": str(window)}
            )

        timestamps.append(now)
        _IP_REQUEST_LOGS[key] = timestamps

        return await call_next(request)


def register_global_exception_handlers(app: FastAPI) -> None:
    """Catches unhandled server exceptions, logs tracebacks internally, and returns safe responses."""

    @app.exception_handler(Exception)
    async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = secrets.token_hex(8)
        log.exception("Unhandled error [correlation_id=%s] on %s %s: %s", correlation_id, request.method, request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "حدث خطأ غير متوقع أثناء معالجة الطلب.",
                "correlation_id": correlation_id,
                "message": "تم تسجيل تفاصيل الخطأ للمراجعة التقنية دون الكشف عن تفاصيل البنية الداخلية."
            }
        )
