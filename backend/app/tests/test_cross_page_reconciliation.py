# backend/app/tests/test_cross_page_reconciliation.py
"""
MetaView 3 — Cross-Page Reconciliation & Incident Set Invariance Test Suite

Verifies:
1. Strict cross-page reconciliation: Incident ID sets (not just counts) must match 100%
   between /fraud-intelligence/dashboard-view and /saudi-fraud/incidents.
2. Map country number == dashboard total == incidents list count for the same filters.
3. Sum of city counts + unknown location count == total country incidents (no double count, no invented cities).
4. Drilldown on any country, city, or hotspot returns the exact matching incidents.
5. Strict separation between incidents, warnings, awareness, and patterns (Requirement 7).
"""

import unittest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db import get_db


class TestCrossPageReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_saudi_30_days_reconciliation(self):
        """Dashboard View and Incidents List must return identical Incident IDs for Saudi Arabia (30 days)."""
        res_dash = self.client.get("/fraud-intelligence/dashboard-view?geo=SA&window_days=30")
        self.assertEqual(res_dash.status_code, 200)
        dash_data = res_dash.json()

        res_inc = self.client.get("/saudi-fraud/incidents?scope=saudi&window_days=30&page_size=100")
        self.assertEqual(res_inc.status_code, 200)
        inc_data = res_inc.json()

        dash_ids = set(dash_data.get("incident_ids", []))
        inc_ids = set(i["id"] for i in inc_data.get("items", []))

        # Check total count
        self.assertEqual(dash_data["total_scope_incidents"], inc_data["total"])
        # Check exact set equality of IDs
        self.assertEqual(dash_ids, inc_ids)

        # Check that Overview endpoint matches the 30-day canonical total
        res_ov = self.client.get("/saudi-fraud/overview?scope=saudi&window_days=30")
        self.assertEqual(res_ov.status_code, 200)
        ov_data = res_ov.json()
        self.assertEqual(ov_data["incidents_total"], dash_data["total_scope_incidents"])

    def test_city_and_unknown_location_invariant(self):
        """Sum of mapped cities + unknown location incidents must equal total country incidents."""
        res_dash = self.client.get("/fraud-intelligence/dashboard-view?geo=SA&window_days=30")
        self.assertEqual(res_dash.status_code, 200)
        data = res_dash.json()

        total = data["total_scope_incidents"]
        city_sum = sum(p["count"] for p in data["map_points"])
        unknown_cnt = data["unknown_location_count"]

        self.assertEqual(city_sum + unknown_cnt, total, "Invariant broken: city sum + unknown != total")

        # Verify city map points do not contain 'Unknown' or invented locations
        for pt in data["map_points"]:
            self.assertNotIn(pt["name"], ["Unknown", "غير محدد", "None", ""])
            self.assertGreater(pt["count"], 0)

        # Verify unknown location incidents are explicitly provided
        self.assertEqual(len(data.get("unknown_location_incident_ids", [])), unknown_cnt)

    def test_world_scope_reconciliation(self):
        """All-world scope returns identical ID sets and sum of top countries equals total."""
        res_dash = self.client.get("/fraud-intelligence/dashboard-view?geo=all&window_days=30")
        self.assertEqual(res_dash.status_code, 200)
        dash_data = res_dash.json()

        res_inc = self.client.get("/saudi-fraud/incidents?scope=all&window_days=30&page_size=100")
        self.assertEqual(res_inc.status_code, 200)
        inc_data = res_inc.json()

        dash_ids = set(dash_data.get("incident_ids", []))
        inc_ids = set(i["id"] for i in inc_data.get("items", []))

        self.assertEqual(dash_data["total_scope_incidents"], inc_data["total"])
        self.assertEqual(dash_ids, inc_ids)

        # Sum of country breakdown equals total
        top_countries = dash_data.get("top_countries", [])
        sum_countries = sum(c["count"] for c in top_countries)
        self.assertEqual(sum_countries, dash_data["total_scope_incidents"])

        # World hotspots counts match top countries
        hotspots = dash_data.get("world_hotspots", [])
        for h in hotspots:
            matching_top = [c for c in top_countries if c["code"].lower() == h["id"].lower()]
            if matching_top:
                self.assertEqual(h["count"], matching_top[0]["count"])

    def test_drilldown_exact_incident_count_and_ids(self):
        """Clicking on a country drilldown returns the exact incidents with matching count."""
        res_dash = self.client.get("/fraud-intelligence/dashboard-view?geo=all&window_days=30")
        dash_data = res_dash.json()

        for c_item in dash_data.get("top_countries", [])[:3]:
            c_code = c_item["code"]
            expected_count = c_item["count"]
            expected_ids = set(c_item["incident_ids"])

            # Call drilldown endpoint passing the country code and window_days
            res_drill = self.client.get(
                f"/fraud-intelligence/incidents?country={c_code}&window_days=30&page_size=100"
            )
            self.assertEqual(res_drill.status_code, 200)
            drill_data = res_drill.json()
            drill_ids = set(i["id"] for i in drill_data.get("items", []))

            # The drilldown items count must equal the displayed number
            self.assertEqual(drill_data["total"], expected_count)
            self.assertEqual(drill_ids, expected_ids)

    def test_separation_of_warnings_and_patterns(self):
        """Explore / Overview must keep warnings and awareness separate from incident counts."""
        res_ov = self.client.get("/saudi-fraud/overview")
        self.assertEqual(res_ov.status_code, 200)
        ov = res_ov.json()

        # Incidents total must NOT equal total articles (which includes warnings and awareness)
        # Verify warnings and awareness fields exist independently
        self.assertIn("incidents_total", ov)
        self.assertIn("warnings_total", ov)
        self.assertIn("awareness_total", ov)
        self.assertIn("excluded_general_cybersecurity", ov)

        # Check /saudi-fraud/news returns warnings separately
        res_news = self.client.get("/saudi-fraud/news?content_type=fraud_warning")
        self.assertEqual(res_news.status_code, 200)
        news_data = res_news.json()
        for item in news_data.get("items", []):
            self.assertEqual(item["content_type"], "fraud_warning")

    def test_fraud_type_filter_reconciliation(self):
        """Filtering by specific fraud type produces identical incident sets across dashboard and list."""
        for ft in ["gov_impersonation", "bank_impersonation", "phishing"]:
            res_dash = self.client.get(f"/fraud-intelligence/dashboard-view?geo=all&fraud_type={ft}&window_days=90")
            self.assertEqual(res_dash.status_code, 200)
            dash_ids = set(res_dash.json().get("incident_ids", []))

            res_inc = self.client.get(f"/saudi-fraud/incidents?scope=all&fraud_type={ft}&window_days=90&page_size=100")
            self.assertEqual(res_inc.status_code, 200)
            inc_ids = set(i["id"] for i in res_inc.json().get("items", []))

            self.assertEqual(dash_ids, inc_ids, f"ID mismatch for fraud_type={ft}")


if __name__ == "__main__":
    unittest.main()
