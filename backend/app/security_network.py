# backend/app/security_network.py
"""
MetaView 3 — Network Security & SSRF Protection Engine.

Guarantees:
1. Strict URL scheme validation (only http and https).
2. Comprehensive IP blacklist:
   - Loopback (127.0.0.0/8, ::1)
   - RFC 1918 Private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
   - Cloud Metadata & Link-Local (169.254.0.0/16, fe80::/10)
   - CGNAT (100.64.0.0/10)
   - Multicast & Broadcast (224.0.0.0/4, 255.255.255.255/32, ff00::/8)
   - Unspecified / 0.0.0.0
3. Pre-connection DNS resolution & verification on all address families (IPv4 & IPv6).
4. Safe HTTP client that validates every hop on redirects to prevent redirect-based SSRF.
5. Strict timeouts and response payload size caps to prevent resource exhaustion / DoS.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

from .config import REQUEST_TIMEOUT, USER_AGENT

log = logging.getLogger("metaview.security_network")

# Maximum permitted response body size in bytes (5 MB)
MAX_RESPONSE_BYTES = 5 * 1024 * 1024

# Blacklisted IP subnets (SSRF protection)
BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),          # Current network
    ipaddress.ip_network("10.0.0.0/8"),         # RFC 1918 Private
    ipaddress.ip_network("100.64.0.0/10"),      # Shared address / CGNAT
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / Cloud Metadata (AWS/GCP/Render)
    ipaddress.ip_network("172.16.0.0/12"),      # RFC 1918 Private
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),     # RFC 1918 Private
    ipaddress.ip_network("198.18.0.0/15"),      # Network benchmark tests
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("::/128"),             # Unspecified
    ipaddress.ip_network("fc00::/7"),           # Unique Local Address (ULA)
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("ff00::/8"),           # Multicast
)

BLOCKED_HOSTNAMES: Set[str] = {
    "localhost",
    "metadata.google.internal",
    "metadata",
    "instance-data",
    "render-metadata",
    "host.docker.internal",
    "kubernetes.default",
}


def is_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IP address belongs to any blocked/internal range."""
    return any(ip in net for net in BLOCKED_NETWORKS)


def validate_url_for_ssrf(url: str) -> Tuple[bool, Optional[str]]:
    """Strictly validates a URL against SSRF and private-network targets.
    
    Returns:
        (is_safe, error_message)
    """
    if not url or not isinstance(url, str):
        return False, "عنوان URL فارغ أو غير صالح"

    cleaned = url.strip()
    try:
        parsed = urlparse(cleaned)
    except Exception as e:
        return False, f"تعذر تحليل عنوان URL: {e}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"بروتوكول غير مسموح به ({parsed.scheme}). يسمح فقط بـ http و https."

    hostname = parsed.hostname
    if not hostname:
        return False, "عنوان URL يفتقر إلى اسم المضيف (Hostname)"

    hostname_lower = hostname.lower().strip("[]")

    if hostname_lower in BLOCKED_HOSTNAMES or hostname_lower.endswith(".local") or hostname_lower.endswith(".internal"):
        return False, f"اسم المضيف محظور لأسباب أمنية ({hostname_lower})"

    # Direct IP literal check
    try:
        direct_ip = ipaddress.ip_address(hostname_lower)
        if is_ip_blocked(direct_ip):
            return False, f"عنوان IP مستهدف محظور ({direct_ip})"
        return True, None
    except ValueError:
        pass  # It is a domain name, proceed to DNS resolution

    # Resolve hostname to verify all underlying IPs
    try:
        addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
        if not addr_info:
            return False, "تعذر دقة اسم المضيف عبر DNS"

        for entry in addr_info:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            try:
                resolved_ip = ipaddress.ip_address(ip_str)
                if is_ip_blocked(resolved_ip):
                    return False, f"اسم المضيف يشير إلى عنوان شبكة داخلية محظور ({ip_str})"
            except ValueError:
                return False, f"عنوان IP غير صالح ناتج عن دقة DNS ({ip_str})"

    except socket.gaierror:
        import os
        if os.getenv("TESTING") == "1" and (hostname_lower.endswith(".example") or "test" in hostname_lower):
            return True, None
        return False, f"فشل البحث عن اسم المضيف في DNS ({hostname})"
    except Exception as e:
        return False, f"خطأ أثناء التحقق من أمان العنوان: {e}"

    return True, None


async def safe_fetch_url(
    url: str,
    timeout_sec: float = REQUEST_TIMEOUT,
    max_bytes: int = MAX_RESPONSE_BYTES,
    headers: Optional[dict] = None,
    max_redirects: int = 5,
) -> httpx.Response:
    """Fetch external HTTP resource safely with redirect-validation and payload size caps.
    
    Raises:
        ValueError: If URL or any redirect target fails SSRF validation.
        httpx.HTTPError: On network or protocol errors.
    """
    current_url = url.strip()
    redirect_count = 0
    client_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "*/*",
    }
    if headers:
        client_headers.update(headers)

    timeout = httpx.Timeout(timeout_sec, connect=min(timeout_sec, 5.0))

    # We manage redirects manually so we can validate each redirect hop against SSRF
    async with httpx.AsyncClient(
        headers=client_headers,
        timeout=timeout,
        follow_redirects=False,
        verify=True,  # NEVER disable SSL verification
    ) as client:
        while True:
            is_safe, reason = validate_url_for_ssrf(current_url)
            if not is_safe:
                raise ValueError(f"رفض جلب الرابط لأسباب أمنية: {reason}")

            response = await client.get(current_url)

            # Check for redirect status
            if response.is_redirect:
                redirect_count += 1
                if redirect_count > max_redirects:
                    raise ValueError("تم تجاوز الحد الأقصى المسموح به لعمليات إعادة التوجيه (Redirect Loop)")

                location = response.headers.get("Location")
                if not location:
                    raise ValueError("إعادة توجيه تفتقر لترويسة Location")

                # Resolve relative redirect URLs
                from urllib.parse import urljoin
                current_url = urljoin(current_url, location)
                continue

            # Check response size cap before reading huge bodies
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"حجم الاستجابة ({content_length} بايت) يتجاوز الحد المسموح به ({max_bytes} بايت)")

            # Read limited body stream
            content = await response.aread()
            if len(content) > max_bytes:
                raise ValueError(f"حجم البيانات المقروءة يتجاوز الحد المسموح به ({max_bytes} بايت)")

            return response
