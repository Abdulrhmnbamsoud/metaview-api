"""Unit tests for the fraud-intelligence engine (no network, no database)."""
import unittest

from backend.app import fraud_intel as fi


class FraudEngineTests(unittest.TestCase):
    def test_saudi_financial_fraud_network(self):
        a, why = fi.analyze(
            "السعودية: الأمن يعلن الإطاحة بشبكة احتيال مالي تضم 9 يمنيين وسودانيين",
            "أعلنت وزارة الداخلية القبض على 9 أشخاص نفذوا عمليات احتيال مالي عبر الاتصال بالضحايا وانتحال صفة موظفي البنوك.",
            require_saudi=True,
        )
        self.assertIsNone(why)
        self.assertTrue(a["is_saudi"])
        self.assertEqual(a["fraud_category"], "financial_fraud")
        self.assertEqual(a["event_type"], "confirmed_incident")
        self.assertIn("9 أشخاص", " ".join(a["verified_facts"]["numbers"]["counts"]))
        # facts and inferences are kept apart
        self.assertEqual(a["verified_facts"]["origin"], "stated_in_source")
        self.assertEqual(a["analytical_assessment"]["origin"], "system_inference")
        self.assertTrue(a["detection_rule"]["is_inference"])
        self.assertEqual(a["detection_rule"]["status"], "proposed_untested")

    def test_bank_news_without_fraud_is_excluded(self):
        a, why = fi.analyze("بنك الرياض يعلن ارتفاع أرباح الربع الثالث", "أعلن البنك عن نمو الأرباح وزيادة التمويل العقاري للعملاء.")
        self.assertIsNone(a)
        self.assertEqual(why, "no_fraud_signal")

    def test_money_words_alone_are_not_fraud(self):
        a, why = fi.analyze("ساما تصدر تعليمات جديدة للبنوك", "البنك المركزي السعودي يطلب من البنوك تحديث بيانات الحسابات وتحويل الأموال.")
        self.assertIsNone(a)
        self.assertEqual(why, "no_fraud_signal")

    def test_sports_fraud_wording_is_excluded(self):
        a, why = fi.analyze("اتهامات باحتيال في مباراة الدوري", "اتهم النادي الحكم بالتلاعب في المباراة.")
        self.assertIsNone(a)
        self.assertEqual(why, "non_financial_context")

    def test_phishing_sms_with_saudi_publisher(self):
        a, why = fi.analyze(
            "هيئة الاتصالات تحذر من رسائل نصية تحتوي روابط مشبوهة",
            "حذرت من رسائل نصية تنتحل صفة جهات حكومية وتطلب تحديث البيانات عبر رابط مشبوه.",
            source={"saudi_publisher": True, "saudi_scope": True, "domain": "okaz.com.sa"},
            require_saudi=True,
        )
        self.assertIsNone(why)
        self.assertIn(a["fraud_category"], ("link_phishing", "gov_impersonation"))
        self.assertEqual(a["saudi_basis"], "saudi_domestic_feed")
        self.assertEqual(a["event_type"], "warning")
        self.assertIn("رسائل نصية (SMS)", a["channels"])

    def test_non_saudi_fraud_rejected_when_required(self):
        a, why = fi.analyze("Police bust phone scam call centre in India", "Officers arrested 12 people who defrauded victims by phone.", require_saudi=True)
        self.assertIsNone(a)
        self.assertEqual(why, "not_saudi")

    def test_united_kingdom_is_not_saudi(self):
        a, why = fi.analyze("المملكة المتحدة: احتيال في الاستثمار عبر الإنترنت", "تحذير من منصات تداول وهمية تستهدف المستثمرين.", require_saudi=True)
        self.assertIsNone(a)
        self.assertEqual(why, "not_saudi")

    def test_investment_fraud_cma(self):
        a, why = fi.analyze(
            "هيئة السوق المالية تحذر من منصات تداول وهمية",
            "حذرت هيئة السوق المالية المستثمرين من منصات تداول وهمية تعد بأرباح مضمونة وغير مرخصة.",
            require_saudi=True,
        )
        self.assertIsNone(why)
        self.assertEqual(a["fraud_category"], "investment_fraud")
        self.assertIn("هيئة السوق المالية", a["verified_facts"]["agencies_mentioned"])

    def test_saudi_outlet_alone_is_not_enough(self):
        a, why = fi.analyze("احتيال مالي عبر منصات تداول وهمية", "حذرت جهات من منصات وهمية تستهدف المستثمرين.",
                            source={"saudi_publisher": True, "domain": "alarabiya.net"}, require_saudi=True)
        self.assertIsNone(a)
        self.assertEqual(why, "not_saudi")

    def test_saudi_outlet_with_domestic_marker(self):
        a, why = fi.analyze("القبض على مقيم نفذ عمليات احتيال عبر رسائل نصية", "ضبطت الجهات المختصة مقيمًا أرسل رسائل احتيالية لسرقة بيانات البطاقات.",
                            source={"saudi_publisher": True, "domain": "sabq.org"}, require_saudi=True)
        self.assertIsNone(why)
        self.assertEqual(a["saudi_basis"], "saudi_outlet_domestic_marker")

    def test_password_phrase_does_not_trigger_gov_category(self):
        a, _ = fi.analyze("احتيال: سرقة كلمة المرور من عملاء بنك", "تلقى العملاء اتصالات تطلب كلمة المرور ورمز التحقق في السعودية.", require_saudi=True)
        self.assertNotEqual(a["fraud_category"], "gov_impersonation")

    def test_whole_word_city_matching(self):
        # "الرياضة" (sports) must not be read as the city "الرياض"
        terms = fi._hits(fi.SAUDI_TERMS, fi.norm("احتيال في الرياضة الإلكترونية"))
        self.assertEqual(terms, [])
        self.assertIn("رياض", fi._hits(fi.SAUDI_TERMS, fi.norm("احتيال في الرياض")))

    def test_normalisation(self):
        self.assertEqual(fi.norm("الاحتيَال الإلكتروني ٢٠٢٦"), fi.norm("الاحتيال الالكتروني 2026"))

    # --- regressions from the first live run -------------------------------------------------
    def test_foreign_fraud_news_is_not_saudi_even_if_page_body_mentions_saudi(self):
        body = "أخبار السعودية سعودي المملكة السعودية قائمة الأقسام"  # menu / related-links boilerplate
        for title, summary in [
            ("الكويت تحجب 790 موقعًا احتياليًا ضمن جهود مكافحة الجرائم الإلكترونية",
             "كشفت دولة الكويت عن تسجيل 493 قضية نصب واحتيال إلكتروني"),
            ("تحذيرات مصرفية من تنامي أساليب الاحتيال الإلكتروني في الأردن",
             "مؤسسات أمنية ومالية أردنية تكثف حملات التوعية من الاحتيال الإلكتروني"),
        ]:
            a, why = fi.analyze(title, summary, body, source={"domain": "example.com"}, require_saudi=True)
            self.assertIsNone(a, title)
            self.assertEqual(why, "not_saudi")

    def test_us_celebrity_fraud_on_saudi_aggregator_is_excluded(self):
        a, why = fi.analyze(
            "ترافيس كيلسي يقع ضحية احتيال مالي بـ 35 مليون دولار",
            "سقوط نجم الرياضة في فخ عملية احتيال مالي قادت مستشاره المالي إلى السجن في أمريكا.",
            "أخبار السعودية سعودي المملكة السعودية",
            source={"domain": "news50.sa", "saudi_publisher": True}, require_saudi=True)
        self.assertIsNone(a)
        self.assertEqual(why, "not_saudi")

    def test_bare_kingdom_from_non_saudi_publisher_is_not_saudi_evidence(self):
        a, why = fi.analyze(
            "عصابات تداول وهمي تتصيد الضحايا",
            "تحذير من منصات استثمار وهمية تستهدف أموال المستثمرين في المملكة",
            source={"domain": "albiladpress.com"}, require_saudi=True)
        self.assertIsNone(a)
        self.assertEqual(why, "not_saudi")

    def test_weak_term_alone_and_sports_are_excluded(self):
        a, why = fi.analyze("ورش طلاء المركبات.. فواتير ملتهبة ومخاوف من الغش",
                            "تتزايد مخاوف المستهلكين من تحايل بعض الورش في السعودية على أسعار الدهان والفواتير")
        self.assertIsNone(a)
        self.assertEqual(why, "no_fraud_signal")
        a, why = fi.analyze("القادسية يلغي 2409 تذاكر قبل مواجهة الهلال",
                            "ألغت إدارة النادي التذاكر لمباراة دوري روشن السعودي بعد رصد عمليات احتيال في بيعها")
        self.assertIsNone(a)
        self.assertEqual(why, "non_financial_context")

    def test_saudi_outlet_body_terms_need_two_hits_and_no_foreign_headline(self):
        src = {"domain": "alwatan.com.sa", "saudi_publisher": True}
        a, why = fi.analyze("الاحتيال الإلكتروني يطال رسوم المدارس الأهلية",
                            "رصدت شكاوى من رسائل احتيالية تطلب تحويل الرسوم إلى حسابات مجهولة",
                            "تلقى أولياء الأمور في الرياض وجدة رسائل ...", source=src, require_saudi=True)
        self.assertIsNone(why)
        self.assertEqual(a["saudi_basis"], "saudi_outlet_body_terms")


if __name__ == "__main__":
    unittest.main()
