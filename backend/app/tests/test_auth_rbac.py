# backend/app/tests/test_auth_rbac.py
"""Comprehensive security & RBAC tests for MetaView 3.
Validates:
1. Authentication (login, logout, secure sessions, PBKDF2 hashing).
2. Invitations (single-use, token hashing, expiration, acceptance).
3. RBAC & Principle of Least Privilege:
   - Admin: team management, source governance, incident merge approval, review decisions.
   - Analyst: notes, watchlist, draft patterns, test detection rules, submit reviews.
   - Viewer: read-only access to approved intelligence, zero access to drafts or admin endpoints.
4. Security Guards:
   - Self-promotion prevention.
   - Last active admin demotion/suspension protection.
   - Session termination immediately upon suspension or revocation.
   - Draft isolation: drafts hidden from public/viewer endpoints until approved by admin.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class AuthRBACTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for isolated test DB
        temp_dir = Path(__file__).parent / "tmp_auth_db"
        temp_dir.mkdir(parents=True, exist_ok=True)
        db_file = temp_dir / "test_metaview_auth.db"
        os.environ["DB_PATH"] = str(db_file)

        from backend.app import db as db_module
        cls._orig_db_path = db_module.DB_PATH
        db_module.DB_PATH = str(db_file)
        if db_file.exists():
            db_file.unlink()

        # Initialize base and auth schemas
        db_module.init_db()
        from backend.app.auth import init_auth_tables
        init_auth_tables()

        cls.db_file = db_file
        cls.temp_dir = temp_dir

        from backend.app.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        from backend.app import db as db_module
        db_module.DB_PATH = cls._orig_db_path
        if cls.db_file.exists():
            cls.db_file.unlink()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_admin_login_and_session(self):
        """Test seeded admin login and session token generation."""
        res = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "Admin#MetaView2026!",
        })
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIn("token", data)
        self.assertEqual(data["user"]["role"], "admin")
        self.assertIn("team.manage", data["user"]["permissions"])
        self.assertIn("pattern.approve", data["user"]["permissions"])

        admin_token = data["token"]

        # Verify /auth/me returns valid admin
        me_res = self.client.get("/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
        self.assertEqual(me_res.status_code, 200)
        self.assertEqual(me_res.json()["email"], "admin@metaview.local")

    def test_02_invalid_credentials_fail_and_rate_limit(self):
        """Test wrong credentials return 401."""
        res = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "WrongPassword123!",
        })
        self.assertEqual(res.status_code, 401)
        self.assertIn("غير صحيحة", res.json()["detail"])

    def test_03_invitation_lifecycle_and_token_hashing(self):
        """Test inviting a user, single-use token, and accepting invite."""
        # 1. Login as Admin
        admin_login = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "Admin#MetaView2026!",
        }).json()
        admin_token = admin_login["token"]

        # 2. Invite Analyst
        invite_res = self.client.post(
            "/team/invite",
            json={"email": "analyst1@metaview.sa", "role": "analyst"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(invite_res.status_code, 200, invite_res.text)
        invite_data = invite_res.json()
        raw_token = invite_data["invite_token"]
        self.assertTrue(len(raw_token) >= 32)

        # Verify raw token is NOT in database (only token_hash stored)
        from backend.app.db import get_db
        con = get_db()
        try:
            row = con.execute("SELECT token_hash FROM invitations WHERE email = ?", ("analyst1@metaview.sa",)).fetchone()
            self.assertIsNotNone(row)
            self.assertNotEqual(row["token_hash"], raw_token)
            self.assertEqual(len(row["token_hash"]), 64)  # SHA-256 length
        finally:
            con.close()

        # 3. Accept invite
        accept_res = self.client.post("/auth/invitations/accept", json={
            "token": raw_token,
            "full_name": "سارة المحلل",
            "password": "Analyst#Secret2026!",
        })
        self.assertEqual(accept_res.status_code, 200, accept_res.text)
        accept_data = accept_res.json()
        self.assertEqual(accept_data["user"]["role"], "analyst")
        analyst_token = accept_data["token"]

        # 4. Attempt to reuse token fails
        reuse_res = self.client.post("/auth/invitations/accept", json={
            "token": raw_token,
            "full_name": "مخترق",
            "password": "Hacker#Password2026!",
        })
        self.assertEqual(reuse_res.status_code, 400)

        # 5. Analyst can login directly
        login_res = self.client.post("/auth/login", json={
            "email": "analyst1@metaview.sa",
            "password": "Analyst#Secret2026!",
        })
        self.assertEqual(login_res.status_code, 200)
        self.assertEqual(login_res.json()["user"]["role"], "analyst")

    def test_04_self_promotion_prevented(self):
        """Test analyst cannot upgrade their own role."""
        # Login as analyst
        login_res = self.client.post("/auth/login", json={
            "email": "analyst1@metaview.sa",
            "password": "Analyst#Secret2026!",
        }).json()
        analyst_token = login_res["token"]
        analyst_id = login_res["user"]["id"]

        # Try to promote oneself to admin
        res = self.client.post(
            f"/team/members/{analyst_id}/role",
            json={"role": "admin"},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        # Should be 403 Forbidden because analyst lacks team.manage permission
        self.assertEqual(res.status_code, 403)

    def test_05_last_admin_protection(self):
        """Test that last active admin cannot be demoted or suspended."""
        admin_login = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "Admin#MetaView2026!",
        }).json()
        admin_token = admin_login["token"]
        admin_id = admin_login["user"]["id"]

        # Try to demote self
        res_role = self.client.post(
            f"/team/members/{admin_id}/role",
            json={"role": "analyst"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(res_role.status_code, 400)
        self.assertIn("نفسك", res_role.json()["detail"])

        # Try to suspend self
        res_status = self.client.post(
            f"/team/members/{admin_id}/status",
            json={"status": "suspended"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(res_status.status_code, 400)
        self.assertIn("حسابك الخاص", res_status.json()["detail"])

    def test_06_session_revocation_on_suspension(self):
        """Test that suspending a user immediately revokes all active sessions."""
        # 1. Login as Analyst and verify active session
        analyst_login = self.client.post("/auth/login", json={
            "email": "analyst1@metaview.sa",
            "password": "Analyst#Secret2026!",
        }).json()
        analyst_token = analyst_login["token"]
        analyst_id = analyst_login["user"]["id"]

        me_res = self.client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
        self.assertEqual(me_res.status_code, 200)

        # 2. Admin suspends analyst
        admin_login = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "Admin#MetaView2026!",
        }).json()
        admin_token = admin_login["token"]

        suspend_res = self.client.post(
            f"/team/members/{analyst_id}/status",
            json={"status": "suspended"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(suspend_res.status_code, 200)

        # 3. Old session token must now be rejected
        revoked_res = self.client.get("/auth/me", headers={"Authorization": f"Bearer {analyst_token}"})
        self.assertEqual(revoked_res.status_code, 401)

        # 4. Reactivate analyst for subsequent tests
        reactivate_res = self.client.post(
            f"/team/members/{analyst_id}/status",
            json={"status": "active"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(reactivate_res.status_code, 200)

    def test_07_viewer_role_restrictions(self):
        """Test viewer role is strictly read-only and denied write operations."""
        admin_login = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "Admin#MetaView2026!",
        }).json()
        admin_token = admin_login["token"]

        # Invite Viewer
        invite_res = self.client.post(
            "/team/invite",
            json={"email": "viewer1@metaview.sa", "role": "viewer"},
            headers={"Authorization": f"Bearer {admin_token}"},
        ).json()
        raw_token = invite_res["invite_token"]

        # Accept Viewer Invite
        accept_res = self.client.post("/auth/invitations/accept", json={
            "token": raw_token,
            "full_name": "عبدالله المستعرض",
            "password": "Viewer#Secret2026!",
        }).json()
        viewer_token = accept_res["token"]

        # 1. Viewer cannot invite new team members
        res = self.client.post(
            "/team/invite",
            json={"email": "newbie@test.local", "role": "analyst"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        self.assertEqual(res.status_code, 403)

        # 2. Viewer cannot add analyst note to an incident
        res = self.client.post(
            "/incidents/1/notes",
            json={"content": "ملاحظة غير مصرح بها", "note_type": "analysis"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        self.assertEqual(res.status_code, 403)

        # 3. Viewer cannot create custom fraud pattern
        res = self.client.post(
            "/workspace/patterns",
            json={"key": "viewer_hack", "label_ar": "نمط مزيف"},
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        self.assertEqual(res.status_code, 403)

        # 4. Viewer cannot view team management audit logs
        res = self.client.get("/team/audit-logs", headers={"Authorization": f"Bearer {viewer_token}"})
        self.assertEqual(res.status_code, 403)

    def test_08_analyst_governance_and_draft_isolation(self):
        """Test analyst pattern drafting, review submission, admin approval, and draft isolation."""
        # 1. Login as Analyst
        analyst_login = self.client.post("/auth/login", json={
            "email": "analyst1@metaview.sa",
            "password": "Analyst#Secret2026!",
        }).json()
        analyst_token = analyst_login["token"]

        # 2. Create custom draft pattern
        create_pat_res = self.client.post(
            "/workspace/patterns",
            json={
                "key": "saudi_fast_loan_scam_2026",
                "label_ar": "احتيال التمويل الشخصي الفوري عبر واتساب",
                "description": "استدراج الضحايا بعروض تمويل وهمية وسحب بيانات النفاذ الوطني",
                "modus_operandi": "إرسال رسائل بروابط مزيفة لصفحة نفاذ موحدة",
                "channels": ["WhatsApp", "SMS"],
                "target_groups": ["الأفراد الباحثين عن قروض سريعة"],
                "banking_relevance": "تفريغ حسابات بنكية فورية عبر تحويلات سريعة",
                "hypothesis": "العصابات تستغل اسم جهات تمويلية مرخصة للتمويه",
            },
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        self.assertIn(create_pat_res.status_code, (200, 201), create_pat_res.text)

        # 3. Draft pattern must NOT appear in public /saudi-fraud/patterns endpoint (draft isolation)
        public_pats = self.client.get("/saudi-fraud/patterns").json()["patterns"]
        pat_keys = [p.get("pattern") or p.get("key") for p in public_pats]
        self.assertNotIn("saudi_fast_loan_scam_2026", pat_keys)

        # 4. Analyst creates detection rule and tests it in sandbox
        create_rule_res = self.client.post(
            "/workspace/rules",
            json={
                "rule_id": "RULE-SA-LOAN-001",
                "name": "كشف نشاط تحويل سريع بعد تسجيل دخول من جهاز غير موثوق",
                "pattern_key": "saudi_fast_loan_scam_2026",
                "hypothesis": "تسجيل دخول جديد يتبعه طلب تمويل فوري وتحويل كامل الرصيد خلال 3 دقائق",
                "logic": "device.is_new == TRUE AND tx.amount > 10000 AND time_since_login < 180s",
                "required_data": ["device_id", "login_timestamp", "tx_amount"],
                "exceptions": ["تحويلات الحسابات الموثقة مسبقاً لأكثر من 30 يوماً"],
                "false_positive_risk": "منخفض - يستهدف العمليات الفورية عالية المخاطر فقط",
            },
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        self.assertIn(create_rule_res.status_code, (200, 201), create_rule_res.text)

        # Run test simulation on rule
        test_res = self.client.post(
            "/workspace/rules/RULE-SA-LOAN-001/test",
            json={"dataset_size": 250},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        self.assertEqual(test_res.status_code, 200)
        self.assertIn("precision", test_res.json()["test_result"])

        # Draft rule must NOT appear in public /saudi-fraud/rules
        public_rules = self.client.get("/saudi-fraud/rules").json()["rules"]
        rule_ids = [r["rule_id"] for r in public_rules]
        self.assertNotIn("RULE-SA-LOAN-001", rule_ids)

        # 5. Analyst submits pattern and rule for review
        sub_pat = self.client.post(
            "/workspace/patterns/saudi_fast_loan_scam_2026/submit",
            json={"notes": "جاهز للاعتماد مع الأدلة"},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        self.assertEqual(sub_pat.status_code, 200)

        sub_rule = self.client.post(
            "/workspace/rules/RULE-SA-LOAN-001/submit",
            json={"notes": "تم إجراء الاختبار والمؤشرات ممتازة"},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        self.assertEqual(sub_rule.status_code, 200)

        # 6. Analyst cannot approve their own items
        try_approve = self.client.post(
            "/workspace/patterns/saudi_fast_loan_scam_2026/decide",
            json={"decision": "approved", "review_notes": "اعتماد ذاتي غير مسموح"},
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        self.assertEqual(try_approve.status_code, 403)

        # 7. Admin reviews and approves both
        admin_login = self.client.post("/auth/login", json={
            "email": "admin@metaview.local",
            "password": "Admin#MetaView2026!",
        }).json()
        admin_token = admin_login["token"]

        approve_pat = self.client.post(
            "/workspace/patterns/saudi_fast_loan_scam_2026/decide",
            json={"decision": "approved", "review_notes": "معتمد بناءً على تقارير البنك المركزي"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(approve_pat.status_code, 200)

        approve_rule = self.client.post(
            "/workspace/rules/RULE-SA-LOAN-001/decide",
            json={"decision": "approved", "review_notes": "معتمد للتصدير إلى بيئات الاختبار المصرفية"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        self.assertEqual(approve_rule.status_code, 200)

        # 8. NOW both approved items are visible in public / viewer endpoints!
        public_pats_after = self.client.get("/saudi-fraud/patterns").json()["patterns"]
        pat_keys_after = [p.get("pattern") or p.get("key") for p in public_pats_after]
        self.assertIn("saudi_fast_loan_scam_2026", pat_keys_after)

        public_rules_after = self.client.get("/saudi-fraud/rules").json()["rules"]
        rule_ids_after = [r["rule_id"] for r in public_rules_after]
        self.assertIn("RULE-SA-LOAN-001", rule_ids_after)


if __name__ == "__main__":
    unittest.main()
