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
        # Seed test incidents into test database
        con = get_db()
        test_rows = [
            (
                'Chase Bank Phishing Incident', 'Details of scam', 'bank_impersonation', 'انتحال صفة بنوك ومصارف', 'Bank Impersonation',
                'confirmed_incident', 'US', 'New York', 'New York',
                'banking', 'القطاع المصرفي والمالي', 'Chase Bank (JPMorgan Chase)',
                'email', 'بريد إلكتروني (Email)', 50000.0, 'USD',
                'Victims lost $50,000 to Chase bank email scam', '["https://example.com/test"]', 1, 'استخدام برمجيات تصيد متقدمة',
                '2026-10-20T10:00:00Z', '2026-10-20T10:00:00Z', '2026-10-20T10:00:00Z'
            ),
            (
                'شرطة الرياض تطيح بتشكيل عصابي', 'تفاصيل القضية', 'gov_impersonation', 'انتحال صفة جهات حكومية', 'Government Impersonation',
                'confirmed_incident', 'SA', 'الرياض', 'منطقة الرياض',
                'banking', 'القطاع المصرفي والمالي', 'وزارة الداخلية',
                'phone_call', 'مكالمات هاتفية', 100000.0, 'SAR',
                'شرطة الرياض تطيح بتشكيل عصابي احتيالي', '["https://spa.gov.sa/1"]', 0, None,
                '2026-10-15T10:00:00Z', '2026-10-15T10:00:00Z', '2026-10-15T10:00:00Z'
            ),
            (
                'شرطة الرياض تضبط شبكة تحويلات', 'تفاصيل القضية', 'gov_impersonation', 'انتحال صفة جهات حكومية', 'Government Impersonation',
                'confirmed_incident', 'SA', 'الرياض', 'منطقة الرياض',
                'banking', 'القطاع المصرفي والمالي', 'النيابة العامة',
                'sms', 'رسائل نصية (SMS)', 80000.0, 'SAR',
                'شرطة الرياض تضبط شبكة تحويلات غير مشروعة', '["https://spa.gov.sa/2"]', 0, None,
                '2026-10-16T10:00:00Z', '2026-10-16T10:00:00Z', '2026-10-16T10:00:00Z'
            ),
            (
                'ضبط شبكة احتيال مالي بالرياض', 'تفاصيل القضية', 'other', 'أخرى / غير محدد', 'Other',
                'confirmed_incident', 'SA', 'الرياض', 'منطقة الرياض',
                'banking', 'القطاع المصرفي والمالي', 'Unknown',
                'sms', 'رسائل نصية (SMS)', 40000.0, 'SAR',
                'ضبط شبكة احتيال مالي بالرياض', '["https://spa.gov.sa/3"]', 0, None,
                '2026-10-17T10:00:00Z', '2026-10-17T10:00:00Z', '2026-10-17T10:00:00Z'
            ),
            (
                'الإطاحة بشبكة احتيال في الرياض', 'تفاصيل القضية', 'other', 'أخرى / غير محدد', 'Other',
                'confirmed_incident', 'SA', 'الرياض', 'منطقة الرياض',
                'banking', 'القطاع المصرفي والمالي', 'Unknown',
                'phone_call', 'مكالمات هاتفية', 25000.0, 'SAR',
                'الإطاحة بشبكة احتيال في الرياض', '["https://spa.gov.sa/4"]', 0, None,
                '2026-10-18T10:00:00Z', '2026-10-18T10:00:00Z', '2026-10-18T10:00:00Z'
            ),
            (
                'السعودية تعلن الإطاحة بعصابة مالية', 'تفاصيل القضية', 'gov_impersonation', 'انتحال صفة جهات حكومية', 'Government Impersonation',
                'confirmed_incident', 'SA', 'Unknown', 'المملكة العربية السعودية',
                'banking', 'القطاع المصرفي والمالي', 'الأمن العام',
                'phone_call', 'مكالمات هاتفية', 90000.0, 'SAR',
                'السعودية تعلن الإطاحة بعصابة مالية', '["https://spa.gov.sa/5"]', 0, None,
                '2026-10-19T10:00:00Z', '2026-10-19T10:00:00Z', '2026-10-19T10:00:00Z'
            ),
            (
                'الأمن السعودي يقبض على شبكة احتيال', 'تفاصيل القضية', 'other', 'أخرى / غير محدد', 'Other',
                'confirmed_incident', 'SA', 'Unknown', 'المملكة العربية السعودية',
                'ecommerce', 'التجارة الإلكترونية', 'Unknown',
                'phone_call', 'مكالمات هاتفية', 150000.0, 'SAR',
                'الأمن السعودي يقبض على شبكة احتيال تضم 9 مقيمين', '["https://spa.gov.sa/6"]', 0, None,
                '2026-10-23T10:00:00Z', '2026-10-23T10:00:00Z', '2026-10-23T10:00:00Z'
            ),
        ]
        con.executemany(
            """
            INSERT INTO incidents (
                title, summary, fraud_type, fraud_type_ar, fraud_type_en,
                record_type, incident_country, incident_city, incident_region,
                targeted_sector, targeted_sector_ar, impersonated_entity,
                channel, channel_ar, financial_loss_amount, financial_loss_currency,
                evidence_text, evidence_sources, is_emerging_threat, emerging_rationale,
                created_at, updated_at, first_reported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            test_rows,
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


    def test_scenario_a_saudi_all_30d(self):
        """Test A: Country = Saudi Arabia, Fraud Type = All, Date = Last 30 Days."""
        resp = self.client.get("/api/fraud-intelligence/dashboard-view?geo=SA&window_days=30&fraud_type=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_empty"])
        self.assertEqual(data["total_scope_incidents"], 6)
        self.assertEqual(data["mapped_incidents_count"], 4)
        self.assertEqual(data["unknown_location_count"], 2)
        self.assertEqual(data["mapped_incidents_count"] + data["unknown_location_count"], data["total_scope_incidents"])
        self.assertEqual(len(data["map_points"]), 1)
        self.assertEqual(data["map_points"][0]["name"], "الرياض")
        self.assertEqual(data["map_points"][0]["count"], 4)
        self.assertGreater(len(data["top_patterns"]), 0)
        self.assertGreater(len(data["latest_incidents"]), 0)

    def test_scenario_b_saudi_phishing_7d(self):
        """Test B: Country = Saudi Arabia, Fraud Type = Phishing, Date = Last 7 Days (Expect Empty State)."""
        resp = self.client.get("/api/fraud-intelligence/dashboard-view?geo=SA&window_days=7&fraud_type=phishing")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_empty"])
        self.assertEqual(data["total_scope_incidents"], 0)
        self.assertEqual(data["mapped_incidents_count"], 0)
        self.assertEqual(data["unknown_location_count"], 0)
        self.assertEqual(len(data["map_points"]), 0)
        self.assertEqual(len(data["top_patterns"]), 0)

    def test_scenario_c_global_all_90d(self):
        """Test C: Country = All, Fraud Type = All, Date = Last 90 Days (Global Data Restored)."""
        resp = self.client.get("/api/fraud-intelligence/dashboard-view?geo=ALL&window_days=90&fraud_type=all")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_empty"])
        self.assertGreaterEqual(data["total_scope_incidents"], 7)
        self.assertEqual(data["mapped_incidents_count"] + data["unknown_location_count"], data["total_scope_incidents"])
        self.assertGreater(len(data["map_points"]), 1)

    def test_scenario_d_saudi_custom_range(self):
        """Test D: Country = Saudi Arabia, Fraud Type = All, Date = Custom Range."""
        resp = self.client.get(
            "/api/fraud-intelligence/dashboard-view?geo=SA&fraud_type=all&date_from=2026-10-01T00:00:00Z&date_to=2026-10-25T23:59:59Z"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_empty"])
        self.assertEqual(data["total_scope_incidents"], 6)

    def test_scenario_e_empty_period(self):
        """Test E: Period with zero incidents returns proper graceful Empty State."""
        resp = self.client.get(
            "/api/fraud-intelligence/dashboard-view?geo=SA&date_from=2025-01-01T00:00:00Z&date_to=2025-01-10T23:59:59Z"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_empty"])
        self.assertIsNotNone(data["empty_message"])
        self.assertEqual(data["total_scope_incidents"], 0)
        self.assertEqual(data["mapped_incidents_count"], 0)
        self.assertEqual(data["unknown_location_count"], 0)

    def test_scenario_f_filter_consistency_across_sections(self):
        """Test F: Cross-Component Validation: Total incidents strictly match across all sections."""
        resp = self.client.get("/api/fraud-intelligence/dashboard-view?geo=SA&window_days=30")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        total = data["total_scope_incidents"]
        self.assertEqual(data["mapped_incidents_count"] + data["unknown_location_count"], total)
        pattern_total = sum(p["incidents_count"] for p in data["top_patterns"])
        self.assertEqual(pattern_total, total)

    def test_scenario_g_reset_to_default(self):
        """Test G: Default endpoint returns default Saudi Arabia 30 days active filter metadata."""
        resp = self.client.get("/api/fraud-intelligence/dashboard-view")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["active_filters"]["geo"], "SA")
        self.assertEqual(data["active_filters"]["fraud_type"], "all")
        self.assertEqual(data["active_filters"]["window_days"], 30)


if __name__ == "__main__":
    unittest.main()

