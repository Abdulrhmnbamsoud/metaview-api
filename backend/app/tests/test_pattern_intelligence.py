# backend/app/tests/test_pattern_intelligence.py
"""Hermetic Tests for Fraud Pattern Classification, Evidence Integrity, and Epistemic Tiers."""
import json
import unittest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db import get_db
from backend.app import incidents


class TestPatternIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        con = get_db()
        try:
            # 1. Seed warning/awareness intelligence for 'fraudulent_ads' (0 incidents, 1 warning, 1 source)
            con.execute("""
                INSERT OR IGNORE INTO articles (id, source, domain, country, headline, content, article_summary, url, published_at, created_at, language)
                VALUES (
                    9001, 'الشرق الأوسط', 'aawsat.com', 'SA',
                    '«ميتا» لـ«الشرق الأوسط»: الاحتيال الرقمي يعبر المنصات ويتطلب تعاوناً أوسع لمواجهته',
                    'أزالت ميتا أكثر من 159 مليون إعلان احتيالي حول العالم وتواجه إعلانات احتيالية أو وهمية.',
                    'إعلانات احتيالية أو وهمية عبر المنصات الرقمية وشبكات التواصل.',
                    'https://aawsat.com/tech/5331466-meta-fraud',
                    '2026-10-21T22:18:42+00:00', '2026-10-22T00:00:00Z', 'ar'
                )
            """)

            vf = {
                "origin": "stated_in_source",
                "method_tags": [
                    {
                        "tag": "fraudulent_ads",
                        "label": "إعلانات احتيالية أو وهمية",
                        "quote": "أزالت شركة ميتا أكثر من 159 مليون إعلان احتيالي حول العالم خلال 2025 وحذفتها قبل إبلاغ المستخدمين."
                    }
                ]
            }

            con.execute("""
                INSERT OR IGNORE INTO threat_intel (
                    id, article_id, title, description, threat_type, content_type, is_saudi, fraud_category, fraud_category_ar,
                    verified_facts, created_at
                )
                VALUES (
                    9001, 9001,
                    '«ميتا» لـ«الشرق الأوسط»: الاحتيال الرقمي يعبر المنصات ويتطلب تعاوناً أوسع لمواجهته',
                    'تفاصيل تحذير ميتا فيو عن إعلانات احتيالية', 'financial_fraud',
                    'awareness_campaign', 1, 'financial_fraud', 'احتيال مالي ومصرفي',
                    ?, '2026-10-22T00:00:00Z'
                )
            """, (json.dumps(vf, ensure_ascii=False),))

            # 2. Seed 2 confirmed incidents for 'bank_impersonation'
            con.execute("""
                INSERT OR IGNORE INTO incidents (
                    id, title, summary, fraud_category, fraud_category_ar, first_reported_at, last_reported_at,
                    article_count, source_count, record_type, methods, channels, bank_relevance, agencies
                )
                VALUES (
                    9002, 'ضبط تشكيل عصابي انتحل صفة بنوك', 'تفاصيل الواقعة', 'bank_impersonation', 'انتحال صفة بنوك ومصارف',
                    '2026-10-18T10:00:00Z', '2026-10-18T10:00:00Z', 3, 2, 'confirmed_incident',
                    '["انتحال صفة بنوك ومصارف"]', '["مكالمات هاتفية"]', 'high', '["شرطة الرياض"]'
                )
            """)

            con.execute("""
                INSERT OR IGNORE INTO incidents (
                    id, title, summary, fraud_category, fraud_category_ar, first_reported_at, last_reported_at,
                    article_count, source_count, record_type, methods, channels, bank_relevance, agencies
                )
                VALUES (
                    9003, 'الإطاحة بمحتالين استولوا على حسابات مصرفية', 'تفاصيل الواقعة', 'bank_impersonation', 'انتحال صفة بنوك ومصارف',
                    '2026-10-19T10:00:00Z', '2026-10-19T10:00:00Z', 2, 2, 'confirmed_incident',
                    '["انتحال صفة بنوك ومصارف"]', '["sms"]', 'high', '["النيابة العامة"]'
                )
            """)

            # 3. Seed 1 reported incident for 'ecommerce_fraud'
            con.execute("""
                INSERT OR IGNORE INTO incidents (
                    id, title, summary, fraud_category, fraud_category_ar, first_reported_at, last_reported_at,
                    article_count, source_count, record_type, methods, channels, bank_relevance, agencies
                )
                VALUES (
                    9004, 'بلاغ عن متجر إلكتروني وهمي', 'تفاصيل الواقعة', 'ecommerce_fraud', 'احتيال تجارة ومتاجر إلكترونية',
                    '2026-10-20T10:00:00Z', '2026-10-20T10:00:00Z', 1, 1, 'reported_incident',
                    '["احتيال تجارة ومتاجر إلكترونية"]', '["متاجر إلكترونية"]', 'medium', '["مصادر الرصد"]'
                )
            """)

            con.commit()
        finally:
            con.close()

    def test_fraudulent_ads_pattern_integrity(self):
        """Acceptance test specifically for 'إعلانات احتيالية أو وهمية' (0 incidents, 1 warning, 1 source)."""
        con = get_db()
        try:
            pats = incidents.build_patterns(con)
        finally:
            con.close()

        ad_pat = next((p for p in pats if p["key"] == "fraudulent_ads"), None)
        self.assertIsNotNone(ad_pat, "fraudulent_ads pattern must be extracted from intelligence sources")

        # 1. Metric counts
        self.assertEqual(ad_pat["incident_count"], 0, "Must not fabricate incidents")
        self.assertEqual(ad_pat["confirmed_incident_count"], 0)
        self.assertGreaterEqual(ad_pat["warning_article_count"], 1, "Must capture warning article")
        self.assertGreaterEqual(ad_pat["source_count"], 1, "Must capture independent source")
        self.assertEqual(len(ad_pat["incidents"]), 0, "No linked criminal incidents allowed")

        # 2. Epistemic tier classification
        self.assertEqual(
            ad_pat["pattern_tier"], "warning_based_intelligence",
            f"Expected warning_based_intelligence tier, got {ad_pat['pattern_tier']}"
        )
        self.assertIn("تحذير", ad_pat["pattern_tier_ar"], "Arabic label must reflect warning status")
        self.assertNotIn("متكرر", ad_pat["pattern_tier_ar"], "0 incidents must NEVER be labeled recurring")
        self.assertNotIn("متكرر", ad_pat["summary"], "Summary must NOT label 0-incident pattern as recurring")

        # 3. Bank relevance with explicit rationale
        self.assertEqual(ad_pat["bank_relevance"], "medium", "Ads without direct bank breach must be analytical medium relevance")
        self.assertTrue("تقدير تحليلي" in ad_pat["bank_relevance_ar"] or "متوسطة" in ad_pat["bank_relevance_ar"])
        self.assertGreater(len(ad_pat["bank_relevance_rationale"]), 10, "Must provide clear justification for bank relevance")

        # 4. Confidence level and rationale
        self.assertIn(ad_pat["confidence_level"], ("medium", "high"))
        self.assertGreater(len(ad_pat["confidence_rationale"]), 10)

        # 5. Evidence and source quote
        self.assertGreaterEqual(len(ad_pat["evidence"]), 1)
        quote = ad_pat["evidence"][0]["quote"]
        self.assertTrue("إعلان" in quote or "ميتا" in quote or "احتيال" in quote, "Evidence must contain verbatim source quote")
        self.assertGreater(len(ad_pat["evidence"][0]["url"]), 0, "Evidence must link to source URL")

        # 6. Structured how_it_works & banking_analysis
        self.assertIn("steps", ad_pat["how_it_works"])
        self.assertIn("potential_bank_risks", ad_pat["banking_analysis"])
        self.assertIn("uncertainties_and_limitations", ad_pat["banking_analysis"])

    def test_confirmed_fraud_pattern_tier(self):
        """Test pattern with multiple confirmed incidents (e.g. bank_impersonation)."""
        con = get_db()
        try:
            pats = incidents.build_patterns(con)
        finally:
            con.close()

        conf_pat = next((p for p in pats if p["key"] == "bank_impersonation"), None)
        self.assertIsNotNone(conf_pat)
        self.assertEqual(conf_pat["pattern_tier"], "confirmed_fraud_pattern")
        self.assertIn("مؤكد", conf_pat["pattern_tier_ar"])
        self.assertGreaterEqual(len(conf_pat["incidents"]), 2)

    def test_single_incident_reported_method_tier(self):
        """Test pattern with exactly 1 incident (e.g. ecommerce_fraud)."""
        con = get_db()
        try:
            pats = incidents.build_patterns(con)
        finally:
            con.close()

        single_pat = next((p for p in pats if p["key"] == "ecommerce_fraud"), None)
        self.assertIsNotNone(single_pat)
        self.assertEqual(single_pat["pattern_tier"], "reported_fraud_method")
        self.assertIn("مبلّغ عنه", single_pat["pattern_tier_ar"])

    def test_pattern_detail_api_endpoints(self):
        """Test GET /saudi-fraud/patterns and /saudi-fraud/patterns/{key} API routes."""
        # List endpoint
        res = self.client.get("/saudi-fraud/patterns")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("patterns", data)
        self.assertTrue(any(p["key"] == "fraudulent_ads" for p in data["patterns"]))

        # Detail endpoint for fraudulent_ads
        res_detail = self.client.get("/saudi-fraud/patterns/fraudulent_ads")
        self.assertEqual(res_detail.status_code, 200)
        p = res_detail.json()["pattern"]
        self.assertEqual(p["key"], "fraudulent_ads")
        self.assertEqual(p["pattern_tier"], "warning_based_intelligence")
        self.assertEqual(p["incident_count"], 0)
        self.assertGreaterEqual(p["warning_article_count"], 1)
        self.assertIn("how_it_works", p)
        self.assertIn("banking_analysis", p)

        # Non-existent pattern returns 404
        res_404 = self.client.get("/saudi-fraud/patterns/non_existent_key_xyz_123")
        self.assertEqual(res_404.status_code, 404)
