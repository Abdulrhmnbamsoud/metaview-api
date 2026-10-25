# backend/app/tests/test_global_fraud.py
import json
import os
import shutil
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

class GlobalFraudTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_dir = Path(__file__).parent / "tmp_db_global"
        temp_dir.mkdir(parents=True, exist_ok=True)
        db_file = temp_dir / "test_global_metaview.db"
        os.environ["DB_PATH"] = str(db_file)

        from backend.app import db as db_module
        cls._orig_db_path = db_module.DB_PATH
        db_module.DB_PATH = str(db_file)
        if db_file.exists():
            db_file.unlink()
        db_module.init_db()
        cls.db_file = db_file
        cls.temp_dir = temp_dir

        # Insert 1 Saudi fraud threat and 1 US global fraud threat
        con = db_module.get_db()
        try:
            # 1. Saudi article
            con.execute(
                """
                INSERT INTO articles (id, headline, content, url, source, source_code, country, domain, published_at, created_at, language)
                VALUES (101, 'الاحتيال المالي في البنوك السعودية', 'تفاصيل احتيال مالي', 'https://example.com/sa-fraud', 'جريدة الرياض', 'alriyadh', 'SA', 'alriyadh.com', '2026-10-24T10:00:00Z', '2026-10-24T10:00:00Z', 'ar')
                """
            )
            con.execute(
                """
                INSERT INTO threat_intel (id, article_id, title, description, is_saudi, threat_type, fraud_category, fraud_category_ar, content_type, bank_relevance, confidence, channels, target_group)
                VALUES (201, 101, 'الاحتيال المالي في البنوك السعودية', 'ملخص احتيال', 1, 'financial_fraud', 'financial_fraud', 'احتيال مالي ومصرفي', 'fraud_warning', 'high', 0.9, '[]', '[]')
                """
            )

            # 2. US Global article
            con.execute(
                """
                INSERT INTO articles (id, headline, content, url, source, source_code, country, domain, published_at, created_at, language)
                VALUES (102, 'FBI warns of phishing scam targeting major banks', 'Phishing emails mimicking banks', 'https://example.com/us-phishing', 'Reuters', 'reuters', 'US', 'reuters.com', '2026-10-24T11:00:00Z', '2026-10-24T11:00:00Z', 'en')
                """
            )
            con.execute(
                """
                INSERT INTO threat_intel (id, article_id, title, description, is_saudi, threat_type, fraud_category, fraud_category_ar, content_type, bank_relevance, confidence, channels, target_group)
                VALUES (202, 102, 'FBI warns of phishing scam targeting major banks', 'Phishing attack description', 0, 'phishing', 'link_phishing', 'احتيال الروابط والتصيد', 'fraud_warning', 'high', 0.85, '[]', '[]')
                """
            )

            # 3. UK Global article
            con.execute(
                """
                INSERT INTO articles (id, headline, content, url, source, source_code, country, domain, published_at, created_at, language)
                VALUES (103, 'London police arrest suspects in investment fraud', 'Investment crypto scam', 'https://example.com/uk-fraud', 'BBC', 'bbc', 'GB', 'bbc.co.uk', '2026-10-24T12:00:00Z', '2026-10-24T12:00:00Z', 'en')
                """
            )
            con.execute(
                """
                INSERT INTO threat_intel (id, article_id, title, description, is_saudi, threat_type, fraud_category, fraud_category_ar, content_type, bank_relevance, confidence, channels, target_group)
                VALUES (203, 103, 'London police arrest suspects in investment fraud', 'Investment scheme', 0, 'financial_fraud', 'investment_fraud', 'احتيال الاستثمار', 'reported_incident', 'medium', 0.8, '[]', '[]')
                """
            )
            con.commit()
        finally:
            con.close()

        from backend.app.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        from backend.app import db as db_module
        db_module.DB_PATH = cls._orig_db_path
        if cls.db_file.exists():
            cls.db_file.unlink()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_scope_saudi(self):
        resp = self.client.get("/saudi-fraud/news?scope=saudi")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["id"], 201)
        self.assertTrue(data["items"][0]["is_saudi"])
        self.assertEqual(data["items"][0]["country"], "SA")

    def test_scope_global(self):
        resp = self.client.get("/saudi-fraud/news?scope=global")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 2)
        ids = {item["id"] for item in data["items"]}
        self.assertEqual(ids, {202, 203})
        for item in data["items"]:
            self.assertFalse(item["is_saudi"])

    def test_scope_all(self):
        resp = self.client.get("/saudi-fraud/news?scope=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 3)

    def test_country_filter(self):
        resp = self.client.get("/saudi-fraud/news?scope=global&country=US")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["country"], "US")
        self.assertEqual(data["items"][0]["id"], 202)

    def test_countries_distribution_api(self):
        resp = self.client.get("/saudi-fraud/countries?scope=global")
        self.assertEqual(resp.status_code, 200)
        countries = {c["country"]: c["count"] for c in resp.json()["countries"]}
        self.assertEqual(countries.get("US"), 1)
        self.assertEqual(countries.get("GB"), 1)
        self.assertNotIn("SA", countries)

        resp_all = self.client.get("/saudi-fraud/countries?scope=all")
        self.assertEqual(resp_all.status_code, 200)
        all_countries = {c["country"]: c["count"] for c in resp_all.json()["countries"]}
        self.assertEqual(all_countries.get("SA"), 1)
        self.assertEqual(all_countries.get("US"), 1)
        self.assertEqual(all_countries.get("GB"), 1)

    def test_search_on_scope(self):
        # Searching "phishing" on global
        resp = self.client.get("/saudi-fraud/news?scope=global&q=phishing")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["id"], 202)

        # Searching "phishing" on saudi should return 0
        resp_saudi = self.client.get("/saudi-fraud/news?scope=saudi&q=phishing")
        self.assertEqual(resp_saudi.status_code, 200)
        self.assertEqual(resp_saudi.json()["total"], 0)
