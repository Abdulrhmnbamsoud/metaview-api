# backend/app/tests/test_x_sources.py
import asyncio
import os
import tempfile
import unittest
from backend.app import db
from backend.app.db import get_db, init_db
from backend.app.x_sources import (
    X_SOURCES,
    X_SOURCES_BY_CODE,
    classify_post_content,
    ingest_x_source,
    check_source_connectivity,
    register_x_sources_in_db,
)
from backend.app import fraud_intel


class TestXSourcesIntegration(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._temp_dir.name, "test_metaview.db")
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = self.db_path
        init_db()
        register_x_sources_in_db()

    def tearDown(self):
        self._temp_dir.cleanup()
        db.DB_PATH = self._orig_db_path

    def test_sources_registration(self):
        """1. Verify both X sources are properly registered in source_registry with all required fields."""
        con = get_db()
        try:
            rows = con.execute(
                "SELECT code, name, url, kind, country, official, handle, last_status, last_error FROM source_registry WHERE kind='x_social'"
            ).fetchall()
            by_code = {r["code"]: dict(r) for r in rows}

            self.assertIn("x_k_h_almuzaini", by_code)
            self.assertIn("x_thehackersnews", by_code)

            # Dr. Al-Muzaini: Saudi specialist, not official
            muzaini = by_code["x_k_h_almuzaini"]
            self.assertEqual(muzaini["handle"], "@K_H_AlMuzaini")
            self.assertEqual(muzaini["url"], "https://x.com/K_H_AlMuzaini")
            self.assertEqual(muzaini["country"], "SA")
            self.assertEqual(muzaini["kind"], "x_social")
            self.assertEqual(muzaini["official"], 0)
            self.assertIn(muzaini["last_status"], ["active", "requires_integration", "requires_authentication", "access_restricted"])

            # The Hacker News: Global platform
            thn = by_code["x_thehackersnews"]
            self.assertEqual(thn["handle"], "@thehackersnews")
            self.assertEqual(thn["url"], "https://x.com/thehackersnews")
            self.assertEqual(thn["country"], "Global")
            self.assertEqual(thn["kind"], "x_social")
            self.assertEqual(thn["official"], 0)
            self.assertEqual(thn["last_status"], "ok")
        finally:
            con.close()

    def test_no_fake_data_for_unauthorized_x_source(self):
        """2. When access is restricted or requires auth, 0 fake data is generated."""
        from unittest.mock import patch
        with patch("backend.app.x_public_collector.collect_and_store_x_posts") as mock_collect:
            mock_collect.return_value = {
                "handle": "K_H_AlMuzaini",
                "source_code": "x_k_h_almuzaini",
                "status": "requires_authentication",
                "error": "Requires Authentication",
                "total_fetched": 0,
                "new_stored": 0,
                "duplicates_skipped": 0,
                "classified_categories": {},
                "eligible_explore_items": 0,
                "stored_items": [],
            }
            res = asyncio.run(ingest_x_source("x_k_h_almuzaini"))
            self.assertEqual(res["status"], "requires_authentication")
            self.assertEqual(res["total_fetched"], 0)
            self.assertEqual(res["new_stored"], 0)
            self.assertEqual(res["eligible_intelligence_items"], 0)

    def test_content_classification_categories(self):
        """3. Content classification maps accurately to the 6 intelligence categories."""
        # Fraud Incident
        res1 = classify_post_content(
            "القبض على تشكيل عصابي استولى على مبالغ مالية عبر منصات استثمار وهمية",
            "شرطة الرياض تطيح بـ 4 وافدين انتحلوا صفة موظفي بنوك واستولوا على حسابات المواطنين",
        )
        self.assertEqual(res1["category_key"], "fraud_incident")
        self.assertTrue(res1["eligible_for_incident"])
        self.assertTrue(res1["eligible_for_explore"])

        # Fraud Warning
        res2 = classify_post_content(
            "تحذير من روابط مشبوهة ورسائل احتيالية تدعي سداد الفواتير الحكومية",
            "تزايد هجمات التصيد الإلكتروني التي تنتحل صفة البريد ومنصات الشحن",
        )
        self.assertEqual(res2["category_key"], "fraud_warning")
        self.assertFalse(res2["eligible_for_incident"])
        self.assertTrue(res2["eligible_for_explore"])

        # Cyber Threat (General cybersecurity - must NEVER be recorded as fraud incident)
        res3 = classify_post_content(
            "Critical Zero-Day Vulnerability Patched in Enterprise Firewall Systems",
            "Security researchers discover active remote code execution exploit targeting Linux servers",
        )
        self.assertEqual(res3["category_key"], "cyber_threat")
        self.assertFalse(res3["eligible_for_incident"])
        self.assertFalse(res3["eligible_for_explore"])

        # Security Awareness
        res4 = classify_post_content(
            "إطلاق حملة توعوية لتعزيز الوعي بمخاطر الهندسة الاجتماعية",
            "ورشة عمل وإرشادات لحماية الحسابات البنكية من التصيد والاحتيال",
        )
        self.assertEqual(res4["category_key"], "security_awareness")
        self.assertFalse(res4["eligible_for_incident"])
        self.assertTrue(res4["eligible_for_explore"])

        # Unrelated
        res5 = classify_post_content(
            "Weather Forecast: Sunny week ahead in Riyadh and eastern regions",
            "Temperatures expected to remain moderate with light winds across the kingdom",
        )
        self.assertEqual(res5["category_key"], "unrelated")
        self.assertFalse(res5["eligible_for_incident"])
        self.assertFalse(res5["eligible_for_explore"])

    def test_deduplication_and_linking(self):
        """4. Verify that existing articles are linked rather than duplicated."""
        res1 = asyncio.run(ingest_x_source("x_thehackersnews"))
        self.assertEqual(res1["status"], "ok")
        # Run second time: already-seen articles must be linked, not re-inserted
        res2 = asyncio.run(ingest_x_source("x_thehackersnews"))
        self.assertEqual(res2["status"], "ok")
        self.assertGreaterEqual(res2["duplicates_linked"], 1)
        self.assertEqual(res2["new_stored"], 0)

    def test_sources_api_endpoint(self):
        """5. FastApi test client verifies /saudi-fraud/sources returns both accounts."""
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)
        resp = client.get("/saudi-fraud/sources")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        sources = data.get("sources", [])
        codes = [s["code"] for s in sources]
        self.assertIn("x_k_h_almuzaini", codes)
        self.assertIn("x_thehackersnews", codes)

        muzaini = next(s for s in sources if s["code"] == "x_k_h_almuzaini")
        self.assertEqual(muzaini["handle"], "@K_H_AlMuzaini")
        self.assertIn(muzaini["last_status"], ["active", "requires_integration", "requires_authentication", "access_restricted"])

        thn = next(s for s in sources if s["code"] == "x_thehackersnews")
        self.assertEqual(thn["handle"], "@thehackersnews")
        self.assertEqual(thn["last_status"], "ok")


if __name__ == "__main__":
    unittest.main()
