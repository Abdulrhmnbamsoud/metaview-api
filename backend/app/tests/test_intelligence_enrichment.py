# backend/app/tests/test_intelligence_enrichment.py
"""Unit tests for Intelligence Enrichment Layer and Incident Rebuilder."""
import os
import shutil
import sqlite3
import tempfile
import unittest

from backend.app.intelligence_enrichment import (
    FRAUD_TYPES,
    enrich_intelligence,
    extract_financial_loss,
    norm,
)
from backend.app.intelligence_builder import rebuild_fraud_intelligence
from backend.app.db import init_db


class TestIntelligenceEnrichment(unittest.TestCase):
    def test_normalization_and_morphology(self):
        t1 = norm("عصابة «الخصم المفاجئ» الاحتيالية في قبضة شرطة دبي")
        self.assertIn("الاحتيالي", t1)
        self.assertIn("دبي", t1)

    def test_content_type_separation(self):
        # 1. Confirmed incident (arrest/conviction)
        res_conf = enrich_intelligence(headline="Police arrest suspect in $73K Chase bank scam")
        self.assertEqual(res_conf["content_type"], "confirmed_incident")
        self.assertEqual(res_conf["fraud_type"], "bank_impersonation")

        # 2. Reported incident (victim loss)
        res_rep = enrich_intelligence(headline="City of Mobile victim of $14.7 million financial fraud")
        self.assertEqual(res_rep["content_type"], "reported_incident")

        # 3. Fraud warning (official warning/alert)
        res_warn = enrich_intelligence(headline="Report Fraud alert: Police forces being impersonated in scam calls")
        self.assertEqual(res_warn["content_type"], "fraud_warning")

        # 4. Awareness campaign
        res_aware = enrich_intelligence(headline="Interactive: How to spot a banking scam before you lose everything")
        self.assertEqual(res_aware["content_type"], "awareness_campaign")

        # 5. General cybersecurity
        res_cyber = enrich_intelligence(headline="Critical zero-day vulnerability discovered in VPN routers")
        self.assertEqual(res_cyber["content_type"], "general_cybersecurity")

        # 6. Unrelated
        res_unrel = enrich_intelligence(headline="Premier League: Arsenal secures late victory against Chelsea")
        self.assertEqual(res_unrel["content_type"], "unrelated")

    def test_financial_loss_extraction(self):
        # $57.7M
        amt, curr, raw = extract_financial_loss("Canadian spear-phishing losses hit $57.7M in 6 mos")
        self.assertEqual(amt, 57_700_000.0)
        self.assertEqual(curr, "USD")

        # SAR with Arabic text
        amt_sar, curr_sar, _ = extract_financial_loss("استولى التشكيل العصابي على 1,250,000 ريال سعودي")
        self.assertEqual(amt_sar, 1_250_000.0)
        self.assertEqual(curr_sar, "SAR")

        # EUR
        amt_eur, curr_eur, _ = extract_financial_loss("cost victims €380,000 in AI fraud")
        self.assertEqual(amt_eur, 380_000.0)
        self.assertEqual(curr_eur, "EUR")

        # GBP
        amt_gbp, curr_gbp, _ = extract_financial_loss("defrauded of £15,000 in roofing scam")
        self.assertEqual(amt_gbp, 15_000.0)
        self.assertEqual(curr_gbp, "GBP")

        # Non-financial text returns None and 'Unknown'
        amt_none, curr_none, _ = extract_financial_loss("General news with no monetary mentions")
        self.assertIsNone(amt_none)
        self.assertEqual(curr_none, "Unknown")

    def test_strict_unknown_for_unstated_fields(self):
        # Text with only fraud mention but no city, no channel, no entity
        res = enrich_intelligence(headline="Police investigate reported investment fraud scheme")
        self.assertEqual(res["fraud_type"], "investment_fraud")
        self.assertEqual(res["incident_country"], "Unknown")
        self.assertEqual(res["incident_city"], "Unknown")
        self.assertEqual(res["victim_location"], "Unknown")
        self.assertEqual(res["suspect_location"], "Unknown")
        self.assertEqual(res["channel"], "Unknown")
        self.assertEqual(res["impersonated_entity"], "Unknown")
        self.assertIsNone(res["financial_loss_amount"])

    def test_emerging_threats_rationale(self):
        # AI-powered scam
        res_ai = enrich_intelligence(headline="Police arrest two suspects in AI-powered bank fraud")
        self.assertEqual(res_ai["is_emerging_threat"], 1)
        self.assertIsNotNone(res_ai["emerging_rationale"])
        self.assertIn("الذكاء الاصطناعي", res_ai["emerging_rationale"])

        # Novel physical collection / gold scam
        res_gold = enrich_intelligence(headline="Norwegians Leave Gold in Forest as Some Fall for Scam")
        self.assertEqual(res_gold["is_emerging_threat"], 1)
        self.assertIn("التسليم الميداني", res_gold["emerging_rationale"])


class TestIntelligenceBuilderMigration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_metaview.db")
        import backend.app.db as db_mod
        self.orig_db_path = db_mod.DB_PATH
        db_mod.DB_PATH = self.db_path
        os.environ["METAVIEW_DB_PATH"] = self.db_path
        db_mod.init_db()

    def tearDown(self):
        import backend.app.db as db_mod
        db_mod.DB_PATH = self.orig_db_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_safe_migration_and_deduplication(self):
        con = sqlite3.connect(self.db_path)
        # Seed 3 articles: two covering Operation Confrontation Ghost, one unrelated
        con.execute(
            """
            INSERT INTO articles (id, headline, article_summary, content, url, source, country, published_at)
            VALUES 
            (1, 'UAE-led Operation Confrontation Ghost identifies $39B in fraud', 'Police announced details', 'full text', 'https://wam.ae/1', 'wam', 'AE', '2026-10-01T10:00:00Z'),
            (2, 'عملية "مواجهة الشبح".. الإمارات تعلن إحباط احتيال بقيمة 39 مليار دولار', 'تفاصيل إحباط العملية', 'نص الخبر', 'https://wam.ae/2', 'wam', 'AE', '2026-10-01T12:00:00Z'),
            (3, 'Local library celebrates 50th anniversary', 'Community gathering', 'full content', 'https://local.test/3', 'local', 'US', '2026-10-01T08:00:00Z')
            """
        )
        con.commit()

        # Run rebuild
        result = rebuild_fraud_intelligence(con=con, user_tag="test_runner")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_articles_evaluated"], 3)
        # Articles 1 and 2 should merge into exactly 1 independent incident!
        self.assertEqual(result["total_incidents_created"], 1)
        self.assertEqual(result["multi_source_deduplications"], 1)

        # Verify audit log was recorded
        audit = con.execute("SELECT * FROM enrichment_audit_log").fetchone()
        self.assertIsNotNone(audit)
        con.close()


if __name__ == "__main__":
    unittest.main()
