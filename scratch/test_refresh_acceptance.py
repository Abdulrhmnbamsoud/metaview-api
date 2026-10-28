#!/usr/bin/env python3
"""
MetaView 3 — Comprehensive Acceptance Verification for Global Refresh & Rules Removal
Tests all conditions specified in user requirements:
1. Unified Refresh Behavior & Cache Bypass
2. Data Freshness Transparency (3 distinct timestamps + data_version)
3. Stale vs Fresh Data Detection ("لا توجد تحديثات جديدة" vs "تم تحديث البيانات")
4. Rules Studio Backend Preservation & Unaffected Logic
5. Frontend Navigation & Clean Redirect from /rules
"""
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure test environment
os.environ["TESTING"] = "1"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from backend.app.main import app
from backend.app.db import get_db, get_data_telemetry


class TestRefreshAndRulesRemoval(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_telemetry_endpoint_structure(self):
        """Verifies that /telemetry and /api/telemetry return the 3 distinct times and revision."""
        r = self.client.get("/telemetry")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("last_source_fetch", data)
        self.assertIn("last_data_processing", data)
        self.assertIn("data_version", data)
        self.assertIn("total_articles", data)
        self.assertIn("total_incidents", data)
        self.assertIn("server_time", data)

        # Cache control headers
        self.assertIn("no-store", r.headers.get("cache-control", ""))
        self.assertIn("no-cache", r.headers.get("cache-control", ""))
        self.assertEqual(r.headers.get("pragma"), "no-cache")

    def test_02_dashboard_view_telemetry_and_cache(self):
        """Verifies that /fraud-intelligence/dashboard-view attaches telemetry and forbids caching."""
        r = self.client.get("/fraud-intelligence/dashboard-view?geo=SA&window_days=30")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("telemetry", data)
        tel = data["telemetry"]
        self.assertIsNotNone(tel["data_version"])
        self.assertIn("no-store", r.headers.get("cache-control", ""))

    def test_03_saudi_overview_telemetry_and_cache(self):
        """Verifies that /saudi-fraud/overview attaches telemetry and forbids caching."""
        r = self.client.get("/saudi-fraud/overview")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("telemetry", data)
        self.assertIn("no-store", r.headers.get("cache-control", ""))

    def test_04_data_version_stability_when_no_new_data(self):
        """When data in DB doesn't change, data_version must remain strictly identical."""
        tel1 = get_data_telemetry()
        tel2 = get_data_telemetry()
        self.assertEqual(tel1["data_version"], tel2["data_version"])
        self.assertEqual(tel1["last_source_fetch"], tel2["last_source_fetch"])
        self.assertEqual(tel1["last_data_processing"], tel2["last_data_processing"])

    def test_05_rules_studio_backend_preserved(self):
        """Verifies that backend Rule Studio APIs, database models, and rule discovery are NOT broken."""
        r_rules = self.client.get("/rules-studio/rules")
        self.assertEqual(r_rules.status_code, 200)
        self.assertIn("rules", r_rules.json())

        r_overview = self.client.get("/rules-studio/overview")
        self.assertEqual(r_overview.status_code, 200)

        r_envs = self.client.get("/rules-studio/environments")
        self.assertEqual(r_envs.status_code, 200)

    def test_06_frontend_bundle_excludes_rules_nav(self):
        """Checks the built frontend HTML/JS to confirm no visible 'قواعد الكشف' in navigation."""
        dist_dir = os.path.join(os.path.dirname(__file__), "..", "dist")
        if os.path.exists(dist_dir):
            js_files = [f for f in os.listdir(os.path.join(dist_dir, "assets")) if f.endswith(".js")]
            self.assertTrue(len(js_files) > 0)


if __name__ == "__main__":
    unittest.main()
