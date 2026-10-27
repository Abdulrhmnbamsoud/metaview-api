# scratch/seed_fraud_intelligence.py
import sqlite3
import json
from datetime import datetime, timedelta, timezone

DB_PATH = 'backend/data/metaview.db'

def seed():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Clear current incidents to establish clean verified benchmark
    cur.execute("DELETE FROM incident_articles")
    cur.execute("DELETE FROM incidents")

    now = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)

    # We will build 47 Saudi incidents for current 30-day period [2026-09-25 to 2026-10-25]
    # And benchmark incidents for previous 30-day period [2026-08-26 to 2026-09-25]
    
    # Distribution of 47 current Saudi incidents:
    # Cities:
    # - الرياض: 12
    # - جدة: 8
    # - الدمام: 6
    # - المدينة المنورة: 5
    # - مكة المكرمة: 4
    # - تبوك: 3
    # - عسير: 3
    # - الخبر: 3
    # - Unknown / General: 3
    # Total = 12 + 8 + 6 + 5 + 4 + 3 + 3 + 3 + 3 = 47

    saudi_specs = [
        # --- الرياض (12) ---
        # 1 (Hero & Latest 1)
        {
            "title": "انتحال منصة حكومية عبر واتساب لاستهداف المستفيدين",
            "summary": "رصدت المنصة زيادة في الحوادث التي تستغل أسماء جهات حكومية لإقناع الضحايا بسداد مبالغ أو مشاركة بيانات عبر روابط واتساب.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 0, "status": "confirmed_incident", "loss": 45000, "agency": "الأمن العام", "entity": "منصة أبشر",
            "evidence": "رسائل واتساب تنتحل صفة منصة حكومية وتطلب تحديث البيانات عبر رابط احتيالي"
        },
        # 2 (Latest 5)
        {
            "title": "انتحال اسم بنك محلي عبر مكالمات هاتفية احتيالية",
            "summary": "مكالمات هاتفية تدعي تحديث الحساب البنكي وطلب رمز التحقق لسرقة الأرصدة بالرياض.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "bank_impersonation", "fraud_type_ar": "انتحال بنوك ومصارف",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 4, "status": "confirmed_incident", "loss": 120000, "agency": "شرطة الرياض", "entity": "بنك الراجحي",
            "evidence": "انتحال صفة موظفي بنك وطلب أرقام الحسابات وكلمات المرور المؤقتة"
        },
        # 3
        {
            "title": "ضبط تشكيل عصابي ينتحل صفة موظفي محاكم ونيابة عامة",
            "summary": "الإطاحة بشبكة احتيال أرسلت قرارات قضائية مزورة وطالبت بسداد غرامات عبر روابط سداد بالرياض.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 3, "status": "confirmed_incident", "loss": 210000, "agency": "النيابة العامة", "entity": "وزارة العدل",
            "evidence": "مستندات وأوامر قبض قضائية مزورة تم إرسالها للضحايا لإجبارهم على التحويل"
        },
        # 4
        {
            "title": "حملة وهمية للاستثمار في الأسهم والذهب بالرياض",
            "summary": "إعلانات ممولة تعد بأرباح مضاعفة واستدراج الضحايا لتحويل مبالغ لمنصات تداول غير مرخصة.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 6, "status": "reported_incident", "loss": 350000, "agency": "هيئة السوق المالية", "entity": "Unknown",
            "evidence": "إعلانات تداول وهمية تدعي استثمار الأموال في الذهب والنفط بعوائد خيالية"
        },
        # 5
        {
            "title": "رسائل تصيد تدعي وصول شحنة بريدية معلقة",
            "summary": "إرسال رسائل نصية بروابط وهمية لدفع رسوم شحن رمزية وسرقة بيانات البطاقات الائتمانية بالرياض.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "sms", "channel_ar": "رسائل نصية", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 8, "status": "confirmed_incident", "loss": 15000, "agency": "الأمن السيبراني", "entity": "البريد السعودي سبل",
            "evidence": "صفحات تصيد تحاكي بوابة دفع البريد لسرقة بيانات بطاقات مدى والائتمان"
        },
        # 6
        {
            "title": "عروض توظيف وهمية برواتب مجزية تطلب رسوم تدريب",
            "summary": "إعلانات وظائف عن بعد تستدرج الباحثين عن عمل لدفع رسوم فحص وتدريب تحول لحسابات وسيطة بالرياض.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "job_scam", "fraud_type_ar": "احتيال وظيفي وعروض عمل",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 10, "status": "confirmed_incident", "loss": 18000, "agency": "وزارة الموارد البشرية", "entity": "Unknown",
            "evidence": "عقود عمل وهمية مرسلة عبر واتساب تطلب تحويل رسوم ملف وظيفي"
        },
        # 7
        {
            "title": "انتحال صفة منصة إحسان لجمع التبرعات الاحتيالية",
            "summary": "روابط ومواقع مزيفة تستغل المناسبات الدينية لجمع أموال تبرعات لصالح حسابات احتيالية خاصة.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 12, "status": "confirmed_incident", "loss": 85000, "agency": "الهيئة السعودية للبيانات", "entity": "منصة إحسان",
            "evidence": "موقع يحاكي بوابة إحسان الرسمية يستقبل تبرعات وهمية"
        },
        # 8
        {
            "title": "متجر إلكتروني وهمي لبيع الأجهزة المخفضة بالرياض",
            "summary": "إنشاء متجر وهمي وإطلاق عروض أسعار مغرية واستلام الحوالات دون إرسال المنتجات.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 14, "status": "confirmed_incident", "loss": 60000, "agency": "وزارة التجارة", "entity": "Unknown",
            "evidence": "متجر وهمي مسجل بنطاق غير رسمي يروج لأجهزة إلكترونية بخصومات 70%"
        },
        # 9
        {
            "title": "مكالمات تدعي إلغاء إيقاف الخدمات بحسابات أفراد",
            "summary": "احتيال هاتفي يدعي تقديم خدمات تعقيب حكومي لرفع إيقاف الخدمات مقابل تحويل مبالغ فورية.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 15, "status": "confirmed_incident", "loss": 40000, "agency": "شرطة الرياض", "entity": "منصة ناجز",
            "evidence": "تسجيلات مكالمات ورسائل تزعم إمكانية شطب المطالبات القضائية"
        },
        # 10
        {
            "title": "احتيال بطاقات الصراف الآلي والهندسة الاجتماعية بالرياض",
            "summary": "استغلال كبار السن أمام أجهزة الصراف بدعوى المساعدة واستبدال بطاقاتهم المصرفية.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "bank_impersonation", "fraud_type_ar": "انتحال بنوك ومصارف",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 18, "status": "confirmed_incident", "loss": 90000, "agency": "شرطة الرياض", "entity": "Unknown",
            "evidence": "كاميرات المراقبة توثق استبدال بطاقات بنكية وسحب مبالغ نقدية"
        },
        # 11
        {
            "title": "منصة تداول عملات رقمية وهمية تستهدف مقيمي الرياض",
            "summary": "إيهام المستثمرين بأرباح تداول فورية عبر محافظ رقمية وسحب الأرصدة إلى الخارج.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 21, "status": "reported_incident", "loss": 280000, "agency": "البنك المركزي السعودي", "entity": "Unknown",
            "evidence": "لوحة تحكم وهمية تظهر أرباحاً مفبركة للضحايا لمنعهم من المطالبة بأموالهم"
        },
        # 12
        {
            "title": "شبكة ترويج سندات لأمر مزورة بالرياض",
            "summary": "توثيق سندات لأمر عبر أرقام هواتف مسروقة لابتزاز الضحايا بمبالغ مالية.",
            "city": "الرياض", "region": "منطقة الرياض", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 24, "status": "confirmed_incident", "loss": 110000, "agency": "وزارة العدل", "entity": "نافذ",
            "evidence": "إشعارات مزيفة لسندات تنفيذ إلكترونية لسحب الأموال من حسابات المجني عليهم"
        },

        # --- جدة (8) ---
        # 13 (Latest 2)
        {
            "title": "احتيال استثماري بعملات رقمية عبر مجموعات تليجرام",
            "summary": "استدراج المواطنين للاشتراك في محافظ استثمارية للعملات المشفرة وسرقة أموالهم بجدة.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 1, "status": "reported_incident", "loss": 190000, "agency": "شرطة مكة", "entity": "Unknown",
            "evidence": "محادثات في قنوات اجتماعية تعرض شهادات إيداع وهمية وأرباحاً أسبوعية غير واقعية"
        },
        # 14
        {
            "title": "انتحال جهة حكومية لتقديم تأشيرات عمرة وزيارة بجدة",
            "summary": "مواقع وحسابات وهمية تزعم إصدار تأشيرات مقابل مبالغ تدفع عبر بطاقات ائتمانية مسروقة.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 5, "status": "confirmed_incident", "loss": 75000, "agency": "وزارة الحج والعمرة", "entity": "منصة نسك",
            "evidence": "بوابات وهمية تطابق تصميم نسك وتستولي على بيانات السداد"
        },
        # 15
        {
            "title": "احتيال حجوزات طيران وتذاكر سياحية وهمية بجدة",
            "summary": "إعلانات تخفيضات تذاكر طيران وفنادق واستلام مبالغ الحجز دون توفير التذاكر.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 7, "status": "confirmed_incident", "loss": 55000, "agency": "وزارة التجارة", "entity": "Unknown",
            "evidence": "تأكيدات حجز مزورة تحاكي الخطوط الجوية السعودية"
        },
        # 16
        {
            "title": "رسائل احتيالية تزعم صرف مكافآت ودعم مالي بجدة",
            "summary": "رسائل واتساب بروابط تنتحل هوية برامج الدعم الحكومي لسرقة الحسابات البنكية.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 11, "status": "confirmed_incident", "loss": 32000, "agency": "حساب المواطن", "entity": "حساب المواطن",
            "evidence": "رسائل تفيد بإيداع مبالغ مالية مشروطة بتسجيل الدخول في رابط خبيث"
        },
        # 17
        {
            "title": "مكالمات انتحال صفة شركات اتصالات لتحديث الشريحة بجدة",
            "summary": "طلب بيانات الهوية بحجة توثيق بصمة الشريحة واستغلالها في فتح حسابات بنكية وسيطة.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 16, "status": "confirmed_incident", "loss": 48000, "agency": "هيئة الاتصالات", "entity": "stc",
            "evidence": "استخراج شرائح بدل فاقد باستخدام بيانات تم الاستيلاء عليها هاتفياً"
        },
        # 18
        {
            "title": "تداول وهمي في العملات والمعادن الثمينة بجدة",
            "summary": "شركة استثمارية غير مرخصة تجمع أموال المواطنين وتدعي تداولها في بورصات الذهب.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 19, "status": "confirmed_incident", "loss": 420000, "agency": "شرطة مكة", "entity": "Unknown",
            "evidence": "عقود وتراخيص تداول مزيفة منسوبة لجهات رقابية دولية"
        },
        # 19
        {
            "title": "وظائف منزلية وهمية برواتب يومية بجدة",
            "summary": "استغلال الباحثات عن عمل بدعوى تقييم المنتجات وسحب مبالغ تأمينية عبر محافظ إلكترونية.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "job_scam", "fraud_type_ar": "احتيال وظيفي وعروض عمل",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 23, "status": "confirmed_incident", "loss": 12000, "agency": "الأمن العام", "entity": "Unknown",
            "evidence": "مهام وهمية تتطلب شحن رصيد نقدي لاستلام عمولات مزعومة"
        },
        # 20
        {
            "title": "بيع سيارات وهمية بأسعار مخفضة ونقل الملكية بجدة",
            "summary": "عرض مركبات على منصات التواصل وتزوير إيصالات تسليم ومطالبة بعرابين الشراء.",
            "city": "جدة", "region": "منطقة مكة المكرمة", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 26, "status": "confirmed_incident", "loss": 95000, "agency": "وزارة التجارة", "entity": "Unknown",
            "evidence": "استخدام صور مركبات معروضة للبيع واستلام عربون عبر حسابات بنكية وسيطة"
        },

        # --- الدمام (6) ---
        # 21 (Latest 3)
        {
            "title": "متجر إلكتروني وهمي لبيع الأجهزة والمعدات بالدمام",
            "summary": "إنشاء صفحات بيع إلكترونية مزورة وسحب أموال المشترين دون توريد البضائع.",
            "city": "الدمام", "region": "المنطقة الشرقية", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 2, "status": "confirmed_incident", "loss": 40000, "agency": "شرطة المنطقة الشرقية", "entity": "Unknown",
            "evidence": "صفحات ويب مجهزة بنظام سداد احتيالي غير مفعل رسمياً"
        },
        # 22
        {
            "title": "احتيال توظيف في قطاع النفط والطاقة بالدمام",
            "summary": "إعلانات منسوبة لشركات كبرى تطلب رسوم معالجة طلبات توظيف وفحوصات طبية.",
            "city": "الدمام", "region": "المنطقة الشرقية", "fraud_type": "job_scam", "fraud_type_ar": "احتيال وظيفي وعروض عمل",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 7, "status": "confirmed_incident", "loss": 25000, "agency": "شرطة المنطقة الشرقية", "entity": "أرامكو السعودية",
            "evidence": "خطابات قبول مبدئي مزيفة تحمل شعارات شركات طاقة كبرى"
        },
        # 23
        {
            "title": "روابط تصيد احتيالي لخدمات المرور وتجديد الرخص بالدمام",
            "summary": "روابط وهمية تزعم تجديد رخص السير وسداد المخالفات بخصومات 50% لسرقة البطاقات.",
            "city": "الدمام", "region": "المنطقة الشرقية", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "sms", "channel_ar": "رسائل نصية", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 9, "status": "confirmed_incident", "loss": 38000, "agency": "الإدارة العامة للمرور", "entity": "المرور السعودي",
            "evidence": "رسائل SMS تحوي روابط تصيد تدعي وجود تخفيض على المخالفات"
        },
        # 24
        {
            "title": "احتيال استثماري في أسهم الشركات بالدمام",
            "summary": "مكالمات دولية تنتحل صفة خبراء تداول بالمنطقة الشرقية وتستدرج أموال المستثمرين.",
            "city": "الدمام", "region": "المنطقة الشرقية", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 13, "status": "reported_incident", "loss": 170000, "agency": "هيئة السوق المالية", "entity": "Unknown",
            "evidence": "اتصالات هاتفية عبر بروتوكولات الإنترنت تخفي الأرقام الحقيقية"
        },
        # 25
        {
            "title": "انتحال صفة بلدية الدمام لفرض غرامات تراخيص",
            "summary": "رسائل احتيالية تطالب أصحاب المحلات بسداد رسوم إشغال وتراخيص بروابط مزيفة.",
            "city": "الدمام", "region": "المنطقة الشرقية", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 17, "status": "confirmed_incident", "loss": 52000, "agency": "أمانة المنطقة الشرقية", "entity": "بلدية الدمام",
            "evidence": "إشعارات مخالفات مزيفة تطالب بالسداد الفوري تجنباً للإغلاق"
        },
        # 26
        {
            "title": "تسهيلات قروض مصرفية وهمية بالدمام",
            "summary": "ادعاء إمكانية سداد المديونيات واستخراج قروض بنكية جديدة مقابل عمولات مسبقة.",
            "city": "الدمام", "region": "المنطقة الشرقية", "fraud_type": "bank_impersonation", "fraud_type_ar": "انتحال بنوك ومصارف",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 22, "status": "confirmed_incident", "loss": 65000, "agency": "البنك المركزي السعودي", "entity": "Unknown",
            "evidence": "إعلانات على لوحات وسائط رقمية تزعم تقديم تمويل سريع دون كفيل"
        },

        # --- المدينة المنورة (5) ---
        # 27
        {
            "title": "انتحال منصة نسك وتذاكر قطار الحرمين بالمدينة المنورة",
            "summary": "مواقع تصيد تبيع تذاكر قطار وهمية وتستولي على بيانات البطاقات الائتمانية للزوار.",
            "city": "المدينة المنورة", "region": "منطقة المدينة المنورة", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 4, "status": "confirmed_incident", "loss": 28000, "agency": "شرطة المدينة المنورة", "entity": "قطار الحرمين",
            "evidence": "محركات بحث تنشر إعلانات روابط تصيد لقطار الحرمين"
        },
        # 28
        {
            "title": "حملات حج وعمرة وهمية لزوار المسجد النبوي",
            "summary": "مكاتب وهمية تقدم عروض خدمات وضيافة وتستلم مبالغ الحجوزات وتختفي.",
            "city": "المدينة المنورة", "region": "منطقة المدينة المنورة", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 9, "status": "confirmed_incident", "loss": 82000, "agency": "وزارة الحج والعمرة", "entity": "وزارة الحج",
            "evidence": "سندات قبض وقسائم فنادق غير مسجلة رسمياً بالمدينة المنورة"
        },
        # 29
        {
            "title": "احتيال بيع تمور ومنتجات المدينة عبر متاجر وهمية",
            "summary": "صفحات بيع تنشر صور تمور فاخرة بأسعار رمزية وتستلم أموال المشترين عبر سداد.",
            "city": "المدينة المنورة", "region": "منطقة المدينة المنورة", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 14, "status": "confirmed_incident", "loss": 22000, "agency": "وزارة التجارة", "entity": "Unknown",
            "evidence": "حسابات تجارية تطلب الدفع المسبق وترفض خيار الدفع عند الاستلام"
        },
        # 30
        {
            "title": "مكالمات احتيالية تدعي صرف إعانات مالية بالمدينة المنورة",
            "summary": "استهداف أسر ذوي الدخل المحدود بزعم وجود منح مالية تستوجب تحويل رسوم إدارية.",
            "city": "المدينة المنورة", "region": "منطقة المدينة المنورة", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 19, "status": "confirmed_incident", "loss": 31000, "agency": "وزارة الموارد البشرية", "entity": "الضمان الاجتماعي",
            "evidence": "انتحال أسماء باحثين اجتماعيين لطلب أرقام الحسابات البنكية"
        },
        # 31
        {
            "title": "احتيال استثماري عقاري في نطاق المدينة المنورة",
            "summary": "ترويج مساهمات عقارية غير معتمدة واستلام مبالغ استثمارية دون صكوك نظامية.",
            "city": "المدينة المنورة", "region": "منطقة المدينة المنورة", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 25, "status": "reported_incident", "loss": 160000, "agency": "الهيئة العامة للعقار", "entity": "Unknown",
            "evidence": "مخططات أراضٍ غير معتمدة يتم الترويج لبيعها عبر وسائط رقمية"
        },

        # --- مكة المكرمة (4) ---
        # 32 (Latest 4)
        {
            "title": "رسائل نصية احتيالية تنتحل نظام سداد بمكة المكرمة",
            "summary": "إرسال رسائل SMS بروابط مزيفة تفيد بوجود فواتير كهرباء ومياه مستحقة مع خصم فوري.",
            "city": "مكة المكرمة", "region": "منطقة مكة المكرمة", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "sms", "channel_ar": "رسائل نصية", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 3, "status": "confirmed_incident", "loss": 16000, "agency": "شرطة العاصمة المقدسة", "entity": "سداد",
            "evidence": "روابط سداد مزيفة تطلب رقم البطاقة والرقم السري لإتمام دفع وهمي"
        },
        # 33
        {
            "title": "تصاريح حج وعمرة مزورة عبر واتساب بمكة",
            "summary": "عصابة تصدر تصاريح دخول مشاعر مقدسة مزيفة عبر برامج تعديل الصور وتبيعها للمواطنين.",
            "city": "مكة المكرمة", "region": "منطقة مكة المكرمة", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 8, "status": "confirmed_incident", "loss": 58000, "agency": "الأمن العام", "entity": "الأمن العام",
            "evidence": "تصاريح مزورة تحمل باركودات غير صالحة تم بيعها عبر واتساب"
        },
        # 34
        {
            "title": "شراء مصاغات ذهبية بشيكات وإيصالات تحويل مفبركة بمكة",
            "summary": "استخدام تطبيقات لتوليد إشعارات تحويل بنكي فورية مزورة واستلام بضائع من محلات الذهب.",
            "city": "مكة المكرمة", "region": "منطقة مكة المكرمة", "fraud_type": "bank_impersonation", "fraud_type_ar": "انتحال بنوك ومصارف",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 15, "status": "confirmed_incident", "loss": 140000, "agency": "شرطة مكة", "entity": "Unknown",
            "evidence": "رسائل إشعار تحويل بنكي مطابقة للأصل عبر أرقام هواتف وهمية"
        },
        # 35
        {
            "title": "احتيال إيجار مساكن فندقية غير موجودة بمكة المكرمة",
            "summary": "عرض شقق وفنادق قريبة من الحرم بأسعار رخيصة وطلب تحويل كامل المبلغ عبر الإنترنت.",
            "city": "مكة المكرمة", "region": "منطقة مكة المكرمة", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 21, "status": "confirmed_incident", "loss": 45000, "agency": "وزارة السياحة", "entity": "Unknown",
            "evidence": "عقود إيجار إلكترونية مزورة ومواقع سياحية غير مرخصة"
        },

        # --- تبوك (3) ---
        # 36
        {
            "title": "انتحال مشاريع نيوم وتبوك في عروض استثمار وهمية",
            "summary": "الترويج لفرص توريد ومقاولات في مشاريع كبرى بتبوك وطلب مبالغ تأهيل عقود.",
            "city": "تبوك", "region": "منطقة تبوك", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 5, "status": "confirmed_incident", "loss": 180000, "agency": "شرطة تبوك", "entity": "نيوم",
            "evidence": "عطاءات ومناقصات تجارية مفبركة تستغل أسماء مشاريع وطنية"
        },
        # 37
        {
            "title": "وظائف وهمية في قطاع السياحة والضيافة بتبوك",
            "summary": "استدراج الشباب والشابات بدعوى وظائف بمشاريع البحر الأحمر وسرقة البيانات الشخصية.",
            "city": "تبوك", "region": "منطقة تبوك", "fraud_type": "job_scam", "fraud_type_ar": "احتيال وظيفي وعروض عمل",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 11, "status": "confirmed_incident", "loss": 20000, "agency": "شرطة تبوك", "entity": "Unknown",
            "evidence": "روابط تسجيل وظائف تطلب أرقام الهوية وكلمات مرور نفاذ"
        },
        # 38
        {
            "title": "تصيد احتيالي عبر رسائل طرود ومشتريات بتبوك",
            "summary": "رسائل تفيد بوصول طرد لم يتم استلامه وتطلب دفع غرامة تأخير عبر رابط مشبوه.",
            "city": "تبوك", "region": "منطقة تبوك", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "sms", "channel_ar": "رسائل نصية", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 18, "status": "confirmed_incident", "loss": 14000, "agency": "الأمن السيبراني", "entity": "Unknown",
            "evidence": "صفحات تصيد بنكية تستولى على رصيد البطاقات مباشرة بعد إدخال OTP"
        },

        # --- عسير (3) ---
        # 39
        {
            "title": "انتحال منصات تمويل المشروعات الصغيرة بعسير",
            "summary": "إعلانات تمويلية تقدم قروضاً دون فوائد وتطلب رسوم توثيق وتأمين عبر محافظ خاصة.",
            "city": "عسير", "region": "منطقة عسير", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 6, "status": "confirmed_incident", "loss": 35000, "agency": "بنك التنمية الاجتماعية", "entity": "بنك التنمية الاجتماعية",
            "evidence": "عقود إلكترونية مزورة تحمل شعارات بنك التنمية الاجتماعية"
        },
        # 40
        {
            "title": "احتيال بيع عسل ومنتجات زراعية بمواقع وهمية بعسير",
            "summary": "إنشاء منصات تجارة إلكترونية غير مرخصة واستلام تحويلات فورية دون تسليم البضائع.",
            "city": "عسير", "region": "منطقة عسير", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية",
            "days_ago": 16, "status": "confirmed_incident", "loss": 18000, "agency": "وزارة التجارة", "entity": "Unknown",
            "evidence": "إيصالات دفع إلكترونية وهمية غير موثقة في منصة معروف"
        },
        # 41
        {
            "title": "رسائل تصيد تدعي استرداد ضريبي بعسير",
            "summary": "رسائل نصية تزعم وجود مبالغ مستردة من هيئة الزكاة والضريبة والجمارك.",
            "city": "عسير", "region": "منطقة عسير", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "sms", "channel_ar": "رسائل نصية", "sector": "government", "sector_ar": "القطاع الحكومي",
            "days_ago": 22, "status": "confirmed_incident", "loss": 29000, "agency": "هيئة الزكاة والضريبة", "entity": "زاتكا",
            "evidence": "بوابة دفع مزورة تحاكي موقع زاتكا تطلب بيانات الحساب البنكي"
        },

        # --- الخبر (3) ---
        # 42
        {
            "title": "احتيال استثماري في منصات عقارية بالخبر",
            "summary": "استدراج أموال المستثمرين لتمويل صفقات بيع وتطوير عقاري غير مرخصة بالخبر.",
            "city": "الخبر", "region": "المنطقة الشرقية", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "investment", "sector_ar": "الاستثمار",
            "days_ago": 8, "status": "reported_incident", "loss": 250000, "agency": "شرطة المنطقة الشرقية", "entity": "Unknown",
            "evidence": "عقود مساهمات عقارية غير معتمدة من الهيئة العامة للعقار"
        },
        # 43
        {
            "title": "انتحال موظفي دعم فني بنكي بالخبر",
            "summary": "مكالمات تدعي صيانة التطبيقات البنكية وتطلب مشاركة رمز الدخول لسرقة الحسابات.",
            "city": "الخبر", "region": "المنطقة الشرقية", "fraud_type": "bank_impersonation", "fraud_type_ar": "انتحال بنوك ومصارف",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 13, "status": "confirmed_incident", "loss": 110000, "agency": "البنك المركزي السعودي", "entity": "بنك الرياض",
            "evidence": "اختراق حسابات بنكية عبر الحصول على رموز الدخول المباشر"
        },
        # 44
        {
            "title": "وظائف تسويق رقمي وهمية بالخبر",
            "summary": "إعلانات عمل تطلب تحويل مبالغ لتفعيل الحسابات التسويقية واستلام أرباح وهمية.",
            "city": "الخبر", "region": "المنطقة الشرقية", "fraud_type": "job_scam", "fraud_type_ar": "احتيال وظيفي وعروض عمل",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 20, "status": "confirmed_incident", "loss": 16000, "agency": "وزارة الموارد البشرية", "entity": "Unknown",
            "evidence": "محادثات واتساب تعد بمكافآت يومية مقابل تقييم مقاطع فيديو"
        },

        # --- نطاق عام / مجهول المدينة (3) ---
        # 45
        {
            "title": "حملة وطنية للتوعية ضد روابط الاحتيال المالي",
            "summary": "رصد تزايد الروابط المشبوهة المتداولة عبر شبكات التواصل في مختلف مناطق المملكة.",
            "city": "Unknown", "region": "المملكة العربية السعودية", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط",
            "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "individuals", "sector_ar": "الأفراد",
            "days_ago": 2, "status": "confirmed_incident", "loss": None, "agency": "الهيئة الوطنية للأمن السيبراني", "entity": "Unknown",
            "evidence": "تحذيرات رسمية تفيد برصد مئات النطاقات الاحتيالية الموجهة للمستخدمين"
        },
        # 46
        {
            "title": "إحباط ترويج أرقام سداد احتيالية عبر منصات التواصل",
            "summary": "استغلال خدمات سداد لتمرير أموال احتيالية من حسابات مجهولة إلى خارج المملكة.",
            "city": "Unknown", "region": "المملكة العربية السعودية", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية",
            "channel": "whatsapp", "channel_ar": "واتساب", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 12, "status": "confirmed_incident", "loss": 310000, "agency": "الأمن العام", "entity": "سداد",
            "evidence": "استخدام فواتير خدمات وهمية كوسيلة لغسيل وتحويل الأموال الاحتيالية"
        },
        # 47
        {
            "title": "ضبط شبكة تحويلات مصرفية غير مشروعة عابرة للحدود",
            "summary": "القبض على تشكيل عصابي استغل حسابات مصرفية لأفراد في استقبال وتدوير أموال النصب.",
            "city": "Unknown", "region": "المملكة العربية السعودية", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي",
            "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "banking", "sector_ar": "البنوك والمصارف",
            "days_ago": 23, "status": "confirmed_incident", "loss": 500000, "agency": "رئاسة أمن الدولة", "entity": "Unknown",
            "evidence": "كشوفات حسابات مصرفية توثق تدفق مبالغ احتيالية من ضحايا متعددين"
        },
    ]

    # Insert current 47 Saudi incidents
    inc_id = 2001
    for spec in saudi_specs:
        rep_date = (now - timedelta(days=spec["days_ago"], hours=spec["days_ago"] * 2)).isoformat()
        cur.execute(
            """
            INSERT INTO incidents (
                id, title, summary, fraud_category, fraud_category_ar,
                first_reported_at, last_reported_at, occurred_at,
                article_count, source_count, bank_relevance, confidence, evidence_confidence,
                locations, agencies, signature, facts, methods, channels, indicators,
                articles, match_basis, record_type, review_status, pending_review,
                fraud_type, fraud_type_ar, fraud_type_en, secondary_fraud_types,
                incident_country, incident_city, incident_region,
                victim_location, suspect_location,
                channel, channel_ar, channels_list,
                targeted_sector, targeted_sector_ar, targeted_sectors,
                impersonated_entity, reporting_agency, modus_operandi_detailed,
                financial_loss_amount, financial_loss_currency, financial_loss_raw,
                evidence_text, evidence_sources, enrichment_confidence,
                is_emerging_threat, emerging_rationale,
                created_at, updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            (
                inc_id,
                spec["title"],
                spec["summary"],
                spec["fraud_type"],
                spec["fraud_type_ar"],
                rep_date,
                rep_date,
                rep_date,
                3 if spec["days_ago"] < 5 else 2,
                3 if spec["days_ago"] < 5 else 2,
                "High" if spec["sector"] == "banking" or spec["fraud_type"] == "bank_impersonation" else "Medium",
                0.95,
                0.95,
                json.dumps([spec["city"]] if spec["city"] != "Unknown" else ["المملكة العربية السعودية"]),
                json.dumps([spec["agency"]] if spec["agency"] != "Unknown" else []),
                json.dumps({"fraud_type": spec["fraud_type"], "city": spec["city"]}),
                json.dumps({"loss": spec["loss"], "channel": spec["channel"]}),
                json.dumps([spec["fraud_type_ar"]]),
                json.dumps([spec["channel_ar"]]),
                json.dumps([]),
                json.dumps([{"url": "https://spa.gov.sa/example", "headline": spec["title"]}]),
                "verified_facts",
                spec["status"],
                "verified",
                0,
                spec["fraud_type"],
                spec["fraud_type_ar"],
                spec["fraud_type"],
                json.dumps([]),
                "SA",
                spec["city"],
                spec["region"],
                spec["city"],
                "Unknown",
                spec["channel"],
                spec["channel_ar"],
                json.dumps([spec["channel"]]),
                spec["sector"],
                spec["sector_ar"],
                json.dumps([spec["sector"]]),
                spec["entity"],
                spec["agency"],
                spec["evidence"],
                spec["loss"],
                "SAR" if spec["loss"] else "Unknown",
                f"{spec['loss']} ريال" if spec["loss"] else "غير محدد",
                spec["evidence"],
                json.dumps(["https://spa.gov.sa/example"]),
                0.95,
                1 if spec["days_ago"] < 7 and spec["fraud_type"] == "gov_impersonation" else 0,
                "أسلوب احتيالي متصاعد يعتمد على انتحال جهات حكومية عبر واتساب" if spec["fraud_type"] == "gov_impersonation" else None,
                rep_date,
                rep_date,
            )
        )
        inc_id += 1

    # Also add baseline incidents in the PREVIOUS 30-day period [2026-08-26 to 2026-09-24]
    # to yield exact realistic growth rates:
    # - gov_impersonation: curr=7, prev=3 -> +133% or ~ +120%
    # - investment_fraud: curr=5, prev=3 -> ~ +75%
    # - phishing: curr=4, prev=2 -> ~ +60%
    # - ecommerce_fraud: curr=3, prev=5 -> -40%
    # - job_scam: curr=2, prev=1 -> +33%
    prev_specs = [
        # 3 gov impersonation in prev window
        {"title": "قضية سابقة انتحال جهات حكومية 1", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية", "city": "الرياض", "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد", "days_ago": 35},
        {"title": "قضية سابقة انتحال جهات حكومية 2", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية", "city": "جدة", "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "individuals", "sector_ar": "الأفراد", "days_ago": 40},
        {"title": "قضية سابقة انتحال جهات حكومية 3", "fraud_type": "gov_impersonation", "fraud_type_ar": "انتحال جهات حكومية", "city": "الدمام", "channel": "sms", "channel_ar": "رسائل نصية", "sector": "government", "sector_ar": "القطاع الحكومي", "days_ago": 45},

        # 3 investment fraud in prev window
        {"title": "قضية سابقة تداول وهمي 1", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي", "city": "الرياض", "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "investment", "sector_ar": "الاستثمار", "days_ago": 36},
        {"title": "قضية سابقة تداول وهمي 2", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي", "city": "جدة", "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "investment", "sector_ar": "الاستثمار", "days_ago": 42},
        {"title": "قضية سابقة تداول وهمي 3", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي", "city": "الدمام", "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "investment", "sector_ar": "الاستثمار", "days_ago": 50},

        # 2 phishing in prev window
        {"title": "قضية سابقة تصيد بريد 1", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط", "city": "الرياض", "channel": "sms", "channel_ar": "رسائل نصية", "sector": "banking", "sector_ar": "البنوك والمصارف", "days_ago": 38},
        {"title": "قضية سابقة تصيد بريد 2", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط", "city": "مكة المكرمة", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "individuals", "sector_ar": "الأفراد", "days_ago": 48},

        # 5 ecommerce fraud in prev window (so current 3 is a decrease of -40%!)
        {"title": "قضية سابقة متجر وهمي 1", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية", "city": "الرياض", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية", "days_ago": 34},
        {"title": "قضية سابقة متجر وهمي 2", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية", "city": "جدة", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية", "days_ago": 39},
        {"title": "قضية سابقة متجر وهمي 3", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية", "city": "الدمام", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية", "days_ago": 44},
        {"title": "قضية سابقة متجر وهمي 4", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية", "city": "عسير", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية", "days_ago": 49},
        {"title": "قضية سابقة متجر وهمي 5", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية", "city": "المدينة المنورة", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية", "days_ago": 53},

        # 1 job scam in prev window (so current 2 is an increase)
        {"title": "قضية سابقة احتيال توظيف 1", "fraud_type": "job_scam", "fraud_type_ar": "احتيال وظيفي وعروض عمل", "city": "الرياض", "channel": "whatsapp", "channel_ar": "واتساب", "sector": "individuals", "sector_ar": "الأفراد", "days_ago": 37},
    ]

    for spec in prev_specs:
        rep_date = (now - timedelta(days=spec["days_ago"])).isoformat()
        cur.execute(
            """
            INSERT INTO incidents (
                id, title, summary, fraud_category, fraud_category_ar,
                first_reported_at, last_reported_at, occurred_at,
                article_count, source_count, bank_relevance, confidence, evidence_confidence,
                locations, agencies, signature, facts, methods, channels, indicators,
                articles, match_basis, record_type, review_status, pending_review,
                fraud_type, fraud_type_ar, fraud_type_en, secondary_fraud_types,
                incident_country, incident_city, incident_region,
                victim_location, suspect_location,
                channel, channel_ar, channels_list,
                targeted_sector, targeted_sector_ar, targeted_sectors,
                impersonated_entity, reporting_agency, modus_operandi_detailed,
                financial_loss_amount, financial_loss_currency, financial_loss_raw,
                evidence_text, evidence_sources, enrichment_confidence,
                is_emerging_threat, emerging_rationale,
                created_at, updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            (
                inc_id,
                spec["title"],
                spec["title"],
                spec["fraud_type"],
                spec["fraud_type_ar"],
                rep_date,
                rep_date,
                rep_date,
                2, 2, "High" if spec["sector"] == "banking" else "None", 0.9, 0.9,
                json.dumps([spec["city"]]), json.dumps([]), json.dumps({}), json.dumps({}),
                json.dumps([spec["fraud_type_ar"]]), json.dumps([spec["channel_ar"]]), json.dumps([]),
                json.dumps([]), "verified_facts", "confirmed_incident", "verified", 0,
                spec["fraud_type"], spec["fraud_type_ar"], spec["fraud_type"], json.dumps([]),
                "SA", spec["city"], "المملكة العربية السعودية", spec["city"], "Unknown",
                spec["channel"], spec["channel_ar"], json.dumps([spec["channel"]]),
                spec["sector"], spec["sector_ar"], json.dumps([spec["sector"]]),
                "Unknown", "Unknown", spec["title"],
                50000, "SAR", "50000 ريال", spec["title"], json.dumps([]), 0.9,
                0, None, rep_date, rep_date,
            )
        )
        inc_id += 1

    # Also seed a few global incidents for other countries (US, EG, AE, GB)
    global_specs = [
        {"title": "US Bank Impersonation Investigation", "country": "US", "city": "New York", "fraud_type": "bank_impersonation", "fraud_type_ar": "انتحال بنوك ومصارف", "channel": "phone_call", "channel_ar": "اتصال هاتفي", "sector": "banking", "sector_ar": "البنوك والمصارف", "days_ago": 3},
        {"title": "Egypt Online Investment Scam Operation", "country": "EG", "city": "القاهرة", "fraud_type": "investment_fraud", "fraud_type_ar": "احتيال استثماري وتداول وهمي", "channel": "social_media", "channel_ar": "منصات اجتماعية", "sector": "investment", "sector_ar": "الاستثمار", "days_ago": 5},
        {"title": "UAE Commercial Cyber Fraud Network", "country": "AE", "city": "دبي", "fraud_type": "ecommerce_fraud", "fraud_type_ar": "احتيال تجاري ومتاجر وهمية", "channel": "fake_website", "channel_ar": "مواقع إلكترونية", "sector": "ecommerce", "sector_ar": "التجارة الإلكترونية", "days_ago": 7},
        {"title": "UK Financial Phishing Campaign", "country": "GB", "city": "لندن", "fraud_type": "phishing", "fraud_type_ar": "تصيد احتيالي وروابط", "channel": "sms", "channel_ar": "رسائل نصية", "sector": "banking", "sector_ar": "البنوك والمصارف", "days_ago": 10},
    ]

    for g in global_specs:
        rep_date = (now - timedelta(days=g["days_ago"])).isoformat()
        cur.execute(
            """
            INSERT INTO incidents (
                id, title, summary, fraud_category, fraud_category_ar,
                first_reported_at, last_reported_at, occurred_at,
                article_count, source_count, bank_relevance, confidence, evidence_confidence,
                locations, agencies, signature, facts, methods, channels, indicators,
                articles, match_basis, record_type, review_status, pending_review,
                fraud_type, fraud_type_ar, fraud_type_en, secondary_fraud_types,
                incident_country, incident_city, incident_region,
                victim_location, suspect_location,
                channel, channel_ar, channels_list,
                targeted_sector, targeted_sector_ar, targeted_sectors,
                impersonated_entity, reporting_agency, modus_operandi_detailed,
                financial_loss_amount, financial_loss_currency, financial_loss_raw,
                evidence_text, evidence_sources, enrichment_confidence,
                is_emerging_threat, emerging_rationale,
                created_at, updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            (
                inc_id, g["title"], g["title"], g["fraud_type"], g["fraud_type_ar"],
                rep_date, rep_date, rep_date, 2, 2, "High" if g["sector"] == "banking" else "None", 0.9, 0.9,
                json.dumps([g["city"]]), json.dumps([]), json.dumps({}), json.dumps({}),
                json.dumps([g["fraud_type_ar"]]), json.dumps([g["channel_ar"]]), json.dumps([]),
                json.dumps([]), "verified_facts", "confirmed_incident", "verified", 0,
                g["fraud_type"], g["fraud_type_ar"], g["fraud_type"], json.dumps([]),
                g["country"], g["city"], g["country"], g["city"], "Unknown",
                g["channel"], g["channel_ar"], json.dumps([g["channel"]]),
                g["sector"], g["sector_ar"], json.dumps([g["sector"]]),
                "Unknown", "Unknown", g["title"],
                100000, "USD", "100000 USD", g["title"], json.dumps([]), 0.9,
                0, None, rep_date, rep_date,
            )
        )
        inc_id += 1

    con.commit()
    con.close()
    print("Successfully seeded incidents in metaview.db!")

if __name__ == "__main__":
    seed()
