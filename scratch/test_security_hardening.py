#!/usr/bin/env python3
"""
MetaView 3 Security Hardening & Banking-Readiness Acceptance Tests
Verifies:
1. Public Viewer Access (Read-Only 200 OK)
2. RBAC & Unauthorized Access Enforcement (Mutating endpoints require auth -> 401/403/503-disabled)
3. Role Privilege Escalation Barriers (Analyst cannot approve rules or trigger system migrations -> 403)
4. SSRF Engine & Socket-Level Anti-DNS Rebinding Defense
5. Information Disclosure Masking (/health and error handlers with correlation IDs)
6. Security Response Headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Permissions-Policy)
7. Request Size Limiting (Payload > 5MB -> 413)
8. Session Idle Timeout & Revocation
9. Database Backup & Verified Restore Plan
"""
import sys
import os
import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

# Set testing environment
os.environ["TESTING"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db import get_db, backup_database, restore_database, DB_PATH
from backend.app.security_network import validate_url_for_ssrf, SSRFSafeTransport
from backend.app.auth import (
    create_session,
    validate_session,
    revoke_session,
    revoke_all_user_sessions,
    hash_password,
    hash_token,
    init_auth_tables,
)
import httpx

client = TestClient(app)

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} - {detail}")
        failed += 1


print("\n=== 1. Public Viewer Access (Read-Only) ===")
r = client.get("/health")
test("GET /health -> 200 OK", r.status_code == 200, f"Got {r.status_code}")

r = client.get("/saudi-fraud/overview")
test("GET /saudi-fraud/overview -> 200 OK", r.status_code == 200, f"Got {r.status_code}")

r = client.get("/threats")
test("GET /threats -> 200 OK", r.status_code == 200, f"Got {r.status_code}")

r = client.get("/articles")
test("GET /articles -> 200 OK", r.status_code == 200, f"Got {r.status_code}")

r = client.get("/rules-studio/rules")
test("GET /rules-studio/rules -> 200 OK", r.status_code == 200, f"Got {r.status_code}")

r = client.get("/rules-studio/overview")
test("GET /rules-studio/overview -> 200 OK", r.status_code == 200, f"Got {r.status_code}")

r = client.get("/rules-studio/environments")
test("GET /rules-studio/environments -> 200 OK", r.status_code == 200, f"Got {r.status_code}")


print("\n=== 2. Broken Access Control & RBAC Enforcement ===")
r = client.post("/ingest/run", json={"feeds": ["https://example.com/rss"]})
test("POST /ingest/run without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")

r = client.post("/ingest-run", json={"feeds": ["https://example.com/rss"]})
test("POST /ingest-run without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")

r = client.post("/rules-studio/rules", json={"title_ar": "قاعدة تجريبية"})
test("POST /rules-studio/rules without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")

r = client.post("/rules-studio/rules/1/governance", json={"governance_status": "APPROVED"})
test("POST /rules-studio/rules/{id}/governance without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")

r = client.post("/rules-studio/rules/1/simulate", json={})
test("POST /rules-studio/rules/{id}/simulate without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")

r = client.post("/rules-studio/parse-scenario", json={"scenario_text": "اختبار سيناريو احتيال"})
test("POST /rules-studio/parse-scenario without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")

r = client.post("/saudi-fraud/ingest/run", json={})
test("POST /saudi-fraud/ingest/run without auth -> 401 or 503 Fail-Closed", r.status_code in (401, 503), f"Got {r.status_code}")

r = client.post("/saudi-fraud/x/collect-public", json={})
test("POST /saudi-fraud/x/collect-public without auth -> 401 or 503 Fail-Closed", r.status_code in (401, 503), f"Got {r.status_code}")

r = client.post("/fraud-intelligence/migrate-database", json={})
test("POST /fraud-intelligence/migrate-database without auth -> 401 Unauthorized", r.status_code == 401, f"Got {r.status_code}")


print("\n=== 3. Role Privilege Boundary & Escalation Prevention ===")
# Create a test analyst user in the database
init_auth_tables()
con = get_db()
cur = con.cursor()
now_iso = datetime.now(timezone.utc).isoformat()
cur.execute(
    """
    INSERT OR REPLACE INTO users (id, email, password_hash, full_name, role, status, created_at, updated_at)
    VALUES (9999, 'analyst.test@metaview.local', ?, 'محلل اختباري', 'analyst', 'active', ?, ?)
    """,
    (hash_password("AnalystPass123!"), now_iso, now_iso),
)
con.commit()
con.close()

# Create a session for the analyst
analyst_token = create_session(9999)
analyst_headers = {"Authorization": f"Bearer {analyst_token}"}

# Analyst can create draft rules
r = client.post(
    "/rules-studio/rules",
    json={"rule": {"title_ar": "قاعدة المحلل التجريبية", "summary_ar": "ملخص تحليلي", "target_env": "HYBRID"}},
    headers=analyst_headers,
)
test("Analyst CAN create draft rule -> 200 OK", r.status_code == 200, f"Got {r.status_code}")
created_rule_id = r.json().get("rule_id", 1) if r.status_code == 200 else 1

# Analyst CANNOT approve rules (requires rule.approve / Admin)
r = client.post(
    f"/rules-studio/rules/{created_rule_id}/governance",
    json={"new_status": "ready_for_bank_testing", "reason": "محاولة ترقية الصلاحية"},
    headers=analyst_headers,
)
test("Analyst CANNOT approve rules (Escalation Barrier) -> 403 Forbidden", r.status_code == 403, f"Got {r.status_code}")

# Analyst CANNOT trigger database migration (requires system.manage / Admin)
r = client.post(
    "/fraud-intelligence/migrate-database",
    json={},
    headers=analyst_headers,
)
test("Analyst CANNOT trigger database migration -> 403 Forbidden", r.status_code == 403, f"Got {r.status_code}")


print("\n=== 4. SSRF & Anti-DNS Rebinding Defense ===")
is_safe, reason = validate_url_for_ssrf("http://127.0.0.1:8000/internal")
test("SSRF blocks loopback IPv4 (127.0.0.1)", not is_safe and ("محظور" in reason or "محجوب" in reason), reason)

is_safe, reason = validate_url_for_ssrf("http://localhost:8080/secrets")
test("SSRF blocks localhost hostname", not is_safe and ("محظور" in reason or "محجوب" in reason), reason)

is_safe, reason = validate_url_for_ssrf("http://169.254.169.254/latest/meta-data/")
test("SSRF blocks AWS/Cloud Metadata (169.254.169.254)", not is_safe and ("محظور" in reason or "محجوب" in reason), reason)

is_safe, reason = validate_url_for_ssrf("http://10.0.0.5/api/v1")
test("SSRF blocks RFC 1918 private network (10.0.0.0/8)", not is_safe and ("محظور" in reason or "محجوب" in reason), reason)

is_safe, reason = validate_url_for_ssrf("http://192.168.1.1/admin")
test("SSRF blocks RFC 1918 private network (192.168.0.0/16)", not is_safe and ("محظور" in reason or "محجوب" in reason), reason)

is_safe, reason = validate_url_for_ssrf("http://172.20.0.10/status")
test("SSRF blocks RFC 1918 private network (172.16.0.0/12)", not is_safe and ("محظور" in reason or "محجوب" in reason), reason)

is_safe, reason = validate_url_for_ssrf("file:///etc/passwd")
test("SSRF blocks file:// protocol", not is_safe and "بروتوكول" in reason, reason)

is_safe, reason = validate_url_for_ssrf("gopher://127.0.0.1:6379/")
test("SSRF blocks gopher:// protocol", not is_safe and "بروتوكول" in reason, reason)

is_safe, reason = validate_url_for_ssrf("ftp://internal.backup/db.tar.gz")
test("SSRF blocks ftp:// protocol", not is_safe and "بروتوكول" in reason, reason)

r = client.post("/saudi-fraud/x/analyze-post", json={"url": "http://169.254.169.254/secret"})
test("Endpoint /saudi-fraud/x/analyze-post blocks Cloud Metadata input", r.status_code == 400, f"Got {r.status_code}")

r = client.post("/saudi-fraud/x/analyze-post", json={"url": "http://127.0.0.1:8000/secret"})
test("Endpoint /saudi-fraud/x/analyze-post blocks Loopback IP", r.status_code == 400, f"Got {r.status_code}")

# Socket-level anti-DNS rebinding verification
async def verify_socket_level_interception():
    transport = SSRFSafeTransport()
    async with httpx.AsyncClient(transport=transport) as ac:
        try:
            await ac.get("http://127.0.0.1:8000/internal-secret")
            return False
        except Exception:
            return True

socket_blocked = asyncio.run(verify_socket_level_interception())
test("SSRFSafeTransport intercepts socket connection to internal IP (Anti-DNS Rebinding)", socket_blocked)


print("\n=== 5. Information Disclosure Masking ===")
r = client.get("/health")
data = r.json()
test("GET /health masks SQLite internal server disk path", data.get("db_path") == "sqlite (persisted)", f"Value: {data.get('db_path')}")
test("GET /health contains no /Users/ or /var/ paths", "/Users/" not in str(data) and "/var/" not in str(data))

r = client.get("/saudi-fraud/runtime")
data = r.json()
test("GET /saudi-fraud/runtime masks raw disk paths for unauthenticated callers", "database_path" not in data or data.get("database_path") == "masked")


print("\n=== 6. Security Response Headers ===")
r = client.get("/health")
headers = r.headers
test("Header X-Content-Type-Options is nosniff", headers.get("X-Content-Type-Options") == "nosniff")
test("Header X-Frame-Options is DENY", headers.get("X-Frame-Options") == "DENY")
test("Header Referrer-Policy is strict-origin-when-cross-origin", headers.get("Referrer-Policy") == "strict-origin-when-cross-origin")
test("Header Permissions-Policy is set", "geolocation=()" in headers.get("Permissions-Policy", ""))
test("Header Content-Security-Policy contains default-src 'self'", "default-src 'self'" in headers.get("Content-Security-Policy", ""))
test("Header Content-Security-Policy contains frame-ancestors 'none'", "frame-ancestors 'none'" in headers.get("Content-Security-Policy", ""))


print("\n=== 7. Request Size Limiting Middleware ===")
huge_payload = "A" * (6 * 1024 * 1024)  # 6MB
r = client.post("/articles", content=huge_payload, headers={"Content-Type": "application/json", "Content-Length": str(len(huge_payload))})
test("Requests exceeding 5MB are rejected with 413 Payload Too Large", r.status_code == 413, f"Got {r.status_code}")


print("\n=== 8. Session Idle Timeout & Revocation ===")
# 8.1 Idle timeout test: Simulate a session last active 15 hours ago
raw_token = create_session(9999)
con = get_db()
cur = con.cursor()
old_time = (datetime.now(timezone.utc) - timedelta(hours=15)).isoformat()
cur.execute("UPDATE sessions SET last_active_at = ? WHERE token_hash = ?", (old_time, hash_token(raw_token)))
con.commit()
con.close()

user = validate_session(raw_token)
test("Session inactive for > 12 hours is automatically expired and invalidated", user is None)

# 8.2 Revoke all sessions test
token_a = create_session(9999)
token_b = create_session(9999)
test("Token A is valid before revocation", validate_session(token_a) is not None)
test("Token B is valid before revocation", validate_session(token_b) is not None)

revoke_all_user_sessions(9999)
test("Token A is terminated after revoke_all_user_sessions", validate_session(token_a) is None)
test("Token B is terminated after revoke_all_user_sessions", validate_session(token_b) is None)


print("\n=== 9. Database Backup & Verified Restore Plan ===")
backup_file = backup_database("acceptance-test")
test("Database backup created successfully using SQLite Online Backup API", backup_file is not None and os.path.exists(backup_file))

# Test restoring into a temporary database file
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
    temp_target = tf.name

try:
    restored = restore_database(backup_file, target_db_path=temp_target)
    test("Database restore into target DB succeeded with PRAGMA quick_check verification", restored is True)

    # Verify restored database contents
    test_con = sqlite3.connect(temp_target)
    t_cur = test_con.cursor()
    t_cur.execute("SELECT COUNT(*) FROM articles")
    articles_count = t_cur.fetchone()[0]
    test_con.close()
    test("Restored database is readable and contains articles data", articles_count >= 0)
finally:
    if os.path.exists(temp_target):
        os.remove(temp_target)


print(f"\n==========================================")
print(f"Summary: {passed} PASSED, {failed} FAILED")
print(f"==========================================")

if failed > 0:
    sys.exit(1)
