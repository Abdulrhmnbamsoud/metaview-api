# backend/app/tests/test_x_public_collector.py
"""Test suite for MetaView 3 X Public Posts Collector.

Validates:
1. Public timeline post extraction and field completeness.
2. Accurate classification into the 5 target categories (confirmed_incident,
   reported_incident, fraud_warning, awareness_campaign, unrelated).
3. Invariant: Social posts from X are NEVER promoted to confirmed incidents
   without verified police/court enforcement facts.
4. Robust deduplication based on canonical tweet ID and URL.
5. Correct status handling (Active, Requires Authentication, Access Restricted, Failed, Not Supported).
6. Single post link analysis via oEmbed / syndication.
7. API endpoints integration (/saudi-fraud/x/collect-public, /saudi-fraud/x/analyze-post, /saudi-fraud/x/status).
"""
import asyncio
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app import db, saudi_api, x_public_collector
from backend.app.x_public_collector import (
    XPublicPost,
    clean_tweet_text,
    classify_x_post,
    collect_and_store_x_posts,
    fetch_public_posts_syndication,
    fetch_single_public_post,
    parse_twitter_date,
    STATUS_ACTIVE,
    STATUS_REQUIRES_AUTH,
    STATUS_ACCESS_RESTRICTED,
    STATUS_FAILED,
    STATUS_NOT_SUPPORTED,
)


class TestXPublicCollector(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._temp_dir.name, "test_metaview.db")
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = self.db_path

        # Initialize schema
        db.init_db()

        # Seed source_registry
        con = sqlite3.connect(self.db_path)
        con.execute(
            """
            INSERT OR REPLACE INTO source_registry (code, name, url, kind, country, lang, official, enabled, last_status, handle)
            VALUES ('x_k_h_almuzaini', 'د. خالد بن حمد المزيني', 'https://x.com/K_H_AlMuzaini', 'x_social', 'SA', 'ar', 0, 1, 'active', '@K_H_AlMuzaini')
            """
        )
        con.commit()
        con.close()

        # Setup test client
        self.app = FastAPI()
        self.app.include_router(saudi_api.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self._temp_dir.cleanup()
        db.DB_PATH = self._orig_db_path

    def test_clean_tweet_text(self):
        raw = "تحذير من روابط التصيد الاحتيالي https://t.co/XyZ12345"
        cleaned = clean_tweet_text(raw)
        self.assertEqual(cleaned, "تحذير من روابط التصيد الاحتيالي")

    def test_parse_twitter_date(self):
        d_str = "Wed Sep 02 18:28:17 +0000 2026"
        iso = parse_twitter_date(d_str)
        self.assertIsNotNone(iso)
        self.assertTrue(iso.startswith("2026-09-02T18:28:17"))

    def test_classification_categories_separation(self):
        # 1. Unrelated greeting/holiday
        res1 = classify_x_post("تقبل الله طاعاتكم ومن العايدين الفايزين وكل عام وأنتم بخير")
        self.assertEqual(res1["category_key"], "unrelated")
        self.assertFalse(res1["eligible_for_incident"])
        self.assertFalse(res1["eligible_for_explore"])

        # 2. Fraud warning (warning about fake apps or marriage scam groups)
        res2 = classify_x_post("#تحذير احذروا من روابط التصيد الاحتيالي والمواقع الوهمية التي تنتحل صفة جهات رسمية")
        self.assertEqual(res2["category_key"], "fraud_warning")
        self.assertFalse(res2["eligible_for_incident"])  # Invariant: warning is NOT an incident
        self.assertTrue(res2["eligible_for_explore"])

        # 3. Awareness campaign (#خلك_حريص)
        res3 = classify_x_post("#خلك_حريص نصائح أمنية وإرشادات لتجنب مشاركة رمز التحقق OTP مع أي شخص")
        self.assertEqual(res3["category_key"], "awareness_campaign")
        self.assertFalse(res3["eligible_for_incident"])
        self.assertTrue(res3["eligible_for_explore"])

        # 4. Reported incident (victim complaints on social media)
        res4 = classify_x_post("مواطن وقع ضحية عصابة احتيال مالي وسحبوا من حسابي 50 ألف ريال بعد اختراق الهاتف")
        self.assertEqual(res4["category_key"], "reported_incident")
        self.assertFalse(res4["eligible_for_incident"])  # Stays as report/threat intel, not verified judicial verdict
        self.assertTrue(res4["eligible_for_explore"])

        # 5. Confirmed incident (strict enforcement facts cited)
        res5 = classify_x_post("أعلنت شرطة الرياض القبض على تشكيل عصابي للاحتيال المالي وإحالتهم للنيابة العامة")
        self.assertEqual(res5["category_key"], "confirmed_incident")
        self.assertTrue(res5["eligible_for_incident"])

    def test_live_syndication_field_extraction(self):
        """Test fetching real public posts from syndication without credentials."""
        posts, status, error = asyncio.run(fetch_public_posts_syndication("K_H_AlMuzaini"))
        # In live environments, Twitter syndication returns ACTIVE or ACCESS_RESTRICTED if rate-limited
        self.assertIn(status, [STATUS_ACTIVE, STATUS_ACCESS_RESTRICTED])
        if status == STATUS_ACTIVE:
            self.assertIsNone(error)
            self.assertGreater(len(posts), 0)
            p = posts[0]
            self.assertTrue(len(p.post_id) > 5)
            self.assertTrue(p.url.startswith("https://x.com/"))
            self.assertEqual(p.author_handle, "@K_H_AlMuzaini")
            self.assertIn("المزيني", p.author_name)
            self.assertIsNotNone(p.text)
            self.assertIn(p.category_key, ["fraud_warning", "awareness_campaign", "unrelated", "reported_incident", "confirmed_incident"])

    @patch("backend.app.x_public_collector.fetch_public_posts_syndication")
    def test_deduplication_and_persistence(self, mock_fetch):
        """Verify that collecting posts multiple times prevents duplicate insertions."""
        sample_posts = [
            XPublicPost(
                post_id="9990001",
                text="تحذير من روابط تصيد احتيالي تنتحل صفة البريد السعودي",
                published_at="2026-10-26T10:00:00+00:00",
                url="https://x.com/K_H_AlMuzaini/status/9990001",
                author_handle="@K_H_AlMuzaini",
                author_name="د. خالد بن حمد المزيني",
                source_code="x_k_h_almuzaini",
                outbound_urls=["https://phishing-site-example.com"],
                media_urls=["https://pbs.twimg.com/media/sample1.jpg"],
                category_key="fraud_warning",
                category_label="تحذير احتيالي",
                content_type="fraud_warning",
                classification_reasons=["warning_about_phishing"],
                is_eligible_explore=True,
                is_eligible_incident=False,
            ),
            XPublicPost(
                post_id="9990002",
                text="#خلك_حريص نصائح لحماية بيانات البطاقة المصرفية ورمز التحقق",
                published_at="2026-10-26T11:00:00+00:00",
                url="https://x.com/K_H_AlMuzaini/status/9990002",
                author_handle="@K_H_AlMuzaini",
                author_name="د. خالد بن حمد المزيني",
                source_code="x_k_h_almuzaini",
                category_key="awareness_campaign",
                category_label="توعية أمنية",
                content_type="awareness_campaign",
                classification_reasons=["awareness_guidance"],
                is_eligible_explore=True,
                is_eligible_incident=False,
            ),
        ]
        mock_fetch.return_value = (sample_posts, STATUS_ACTIVE, None)

        # First collection
        report1 = asyncio.run(collect_and_store_x_posts(handle="K_H_AlMuzaini", max_posts=10))
        self.assertEqual(report1["status"], STATUS_ACTIVE)
        self.assertEqual(report1["new_stored"], 2)
        self.assertEqual(report1["duplicates_skipped"], 0)

        con = sqlite3.connect(self.db_path)
        count1 = con.execute("SELECT COUNT(*) FROM articles WHERE source_code = 'x_k_h_almuzaini'").fetchone()[0]
        con.close()
        self.assertEqual(count1, 2)

        # Second collection (all should be detected as duplicates)
        report2 = asyncio.run(collect_and_store_x_posts(handle="K_H_AlMuzaini", max_posts=10))
        self.assertEqual(report2["status"], STATUS_ACTIVE)
        self.assertEqual(report2["new_stored"], 0)
        self.assertEqual(report2["duplicates_skipped"], 2)

        con = sqlite3.connect(self.db_path)
        count2 = con.execute("SELECT COUNT(*) FROM articles WHERE source_code = 'x_k_h_almuzaini'").fetchone()[0]
        threat_count = con.execute("SELECT COUNT(*) FROM threat_intel").fetchone()[0]
        incident_count = con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        con.close()
        self.assertEqual(count2, 2)
        self.assertEqual(threat_count, 2)
        # Golden invariant: No X posts promoted to confirmed incidents!
        self.assertEqual(incident_count, 0)

    def test_single_post_oembed_analysis(self):
        """Test analyzing a specific tweet link via public oEmbed."""
        test_url = "https://x.com/K_H_AlMuzaini/status/1558848332893753350"
        post, status, error = asyncio.run(fetch_single_public_post(test_url))
        self.assertIn(status, [STATUS_ACTIVE, STATUS_ACCESS_RESTRICTED])
        if status == STATUS_ACTIVE:
            self.assertIsNone(error)
            self.assertIsNotNone(post)
            self.assertEqual(post.post_id, "1558848332893753350")
            self.assertIn("المزيني", post.author_name)
            self.assertEqual(post.category_key, "fraud_warning")

    def test_status_codes_and_labels(self):
        """Verify that source status maps to the 5 official states."""
        self.assertEqual(x_public_collector.STATUS_LABELS[STATUS_ACTIVE], "Active")
        self.assertEqual(x_public_collector.STATUS_LABELS[STATUS_REQUIRES_AUTH], "Requires Authentication")
        self.assertEqual(x_public_collector.STATUS_LABELS[STATUS_ACCESS_RESTRICTED], "Access Restricted")
        self.assertEqual(x_public_collector.STATUS_LABELS[STATUS_FAILED], "Failed")
        self.assertEqual(x_public_collector.STATUS_LABELS[STATUS_NOT_SUPPORTED], "Not Supported")

        # Test invalid handle yields Not Supported
        posts, status, error = asyncio.run(fetch_public_posts_syndication(""))
        self.assertEqual(status, STATUS_NOT_SUPPORTED)
        self.assertIn("غير صالح", error)

    @patch("backend.app.x_public_collector.fetch_single_public_post")
    def test_api_endpoints(self, mock_single):
        """Test FastAPI endpoints for public X collection and status."""
        # 1. Status endpoint
        resp1 = self.client.get("/saudi-fraud/x/status")
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertEqual(data1["source_code"], "x_k_h_almuzaini")
        self.assertIn(data1["status_label"], ["Active", "Requires Authentication", "Access Restricted", "Failed"])

        # 2. Analyze post endpoint
        mock_post = XPublicPost(
            post_id="1558848332893753350",
            text="تحذير من روابط التصيد الاحتيالي",
            published_at="2026-10-26T12:00:00+00:00",
            url="https://x.com/K_H_AlMuzaini/status/1558848332893753350",
            author_handle="@K_H_AlMuzaini",
            author_name="د. خالد بن حمد المزيني",
            source_code="x_k_h_almuzaini",
            category_key="fraud_warning",
            category_label="تحذير احتيالي",
            content_type="fraud_warning",
        )
        mock_single.return_value = (mock_post, STATUS_ACTIVE, None)

        resp2 = self.client.post(
            "/saudi-fraud/x/analyze-post",
            json={"url": "https://x.com/K_H_AlMuzaini/status/1558848332893753350"},
        )
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertEqual(data2["status"], STATUS_ACTIVE)
        self.assertEqual(data2["post"]["post_id"], "1558848332893753350")


if __name__ == "__main__":
    unittest.main()
