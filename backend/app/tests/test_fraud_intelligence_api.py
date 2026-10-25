# backend/app/tests/test_fraud_intelligence_api.py
"""Unit tests for Fraud Intelligence Analytics Endpoints."""
import unittest
from fastapi.testclient import TestClient

from backend.app.db import get_db
from backend.app.main import app


class TestFraudIntelligenceApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Seed test incident into test database
        con = get_db()
        con.execute(
            """
            INSERT INTO incidents (
                title, summary, fraud_type, fraud_type_ar, fraud_type_en,
                record_type, incident_country, incident_city, incident_region,
                targeted_sector, targeted_sector_ar, impersonated_entity,
                channel, channel_ar, financial_loss_amount, financial_loss_currency,
                evidence_text, evidence_sources, is_emerging_threat, emerging_rationale,
                created_at, updated_at
            ) VALUES (
                'Chase Bank Phishing Incident', 'Details of scam', 'bank_impersonation', 'انتحال صفة بنوك ومصارف', 'Bank Impersonation',
                'confirmed_incident', 'US', 'New York', 'New York',
                'banking', 'القطاع المصرفي والمالي', 'Chase Bank (JPMorgan Chase)',
                'email', 'بريد إلكتروني (Email)', 50000.0, 'USD',
                'Victims lost $50,000 to Chase bank email scam', '["https://example.com/test"]', 1, 'استخدام برمجيات تصيد متقدمة',
                '2026-10-20T10:00:00Z', '2026-10-20T10:00:00Z'
            )
            """
        )
        con.commit()
        con.close()

    def test_summary_kpis(self):
        resp = self.client.get("/api/fraud-intelligence/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_unique_incidents", data)
        self.assertIn("confirmed_incidents", data)
        self.assertIn("reported_incidents", data)
        self.assertIn("financial_loss_documented", data)
        self.assertIn("excluded_content_breakdown", data)
        self.assertGreaterEqual(data["total_unique_incidents"], 1)

    def test_trends_equal_window(self):
        resp = self.client.get("/api/fraud-intelligence/trends?window_days=30")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["window_days"], 30)
        self.assertIn("current_observed_incidents", data)
        self.assertIn("current_source_coverage_volume", data)
        self.assertIn("neutral_disclaimer", data)
        self.assertIn("series", data)

    def test_typologies_breakdown(self):
        resp = self.client.get("/api/fraud-intelligence/typologies")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        first = data[0]
        self.assertIn("key", first)
        self.assertIn("name_ar", first)
        self.assertIn("count", first)
        self.assertIn("incident_ids", first)

    def test_geography_distribution(self):
        resp = self.client.get("/api/fraud-intelligence/geography")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("countries", data)
        self.assertIn("cities", data)
        self.assertIn("documented_geo_incidents", data)

    def test_entities_and_sectors(self):
        resp = self.client.get("/api/fraud-intelligence/entities-and-sectors")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("sectors", data)
        self.assertIn("impersonated_entities", data)

    def test_channels_distribution(self):
        resp = self.client.get("/api/fraud-intelligence/channels")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

    def test_emerging_threats(self):
        resp = self.client.get("/api/fraud-intelligence/emerging-threats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        if data:
            self.assertIn("emerging_rationale", data[0])
            self.assertIn("evidence_text", data[0])

    def test_banking_risk(self):
        resp = self.client.get("/api/fraud-intelligence/banking-risk")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_banking_incidents", data)
        self.assertIn("estimated_banking_loss_usd", data)

    def test_data_quality(self):
        resp = self.client.get("/api/fraud-intelligence/data-quality")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("field_completion_stats", data)
        self.assertIn("sources", data)

    def test_drilldown_incidents(self):
        resp = self.client.get("/api/fraud-intelligence/incidents?page=1&page_size=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("items", data)
        self.assertIn("total", data)
        if data["items"]:
            first_id = data["items"][0]["id"]
            single_resp = self.client.get(f"/api/fraud-intelligence/incidents/{first_id}")
            self.assertEqual(single_resp.status_code, 200)
            single_data = single_resp.json()
            self.assertIn("evidence_text", single_data)


if __name__ == "__main__":
    unittest.main()
