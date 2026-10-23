# backend/app/fraud_intel.py
"""Fraud-focused threat intelligence engine (Arabic + English).

Design goals
------------
* A news item is *fraud intelligence* only when the text carries an explicit fraud
  signal (احتيال / تصيد / انتحال صفة / scam / phishing ...) **and** a financial or
  cyber context. Words such as "bank" or "money" alone never qualify an article.
* Everything that is *stated by the source* (``verified_facts``) is kept separate from
  everything the system *infers* (``analytical_assessment``, detection rule, ...).
* The engine is deterministic and explainable: every classification stores the terms
  that triggered it. It is a keyword/heuristic engine on purpose - no ML claims.

Public API
----------
``analyze(title, summary, body, source=None, require_saudi=False)``
    -> ``(analysis_dict | None, exclusion_reason | None)``
``has_fraud_signal(text)``      -> cheap pre-gate used before downloading article bodies
``norm(text)``                  -> Arabic/English normalisation used for matching
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640\u200c\u200d\u200e\u200f]")
_WS = re.compile(r"\s+")
_AR_LETTER = "\u0621-\u064A"


def norm(text: Optional[str]) -> str:
    """Normalise Arabic (alef/yaa/taa-marbuta/diacritics/digits) and lowercase."""
    t = (text or "").translate(_AR_DIGITS)
    t = _DIACRITICS.sub("", t)
    t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
          .replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي"))
    return _WS.sub(" ", t.lower()).strip()


Matcher = Union[str, Pattern[str]]


def _compile(term: str) -> Matcher:
    """Compile a human-written term into a matcher.

    * plain Arabic term      -> substring match on normalised text
    * ``=term`` (Arabic)     -> whole word (optional Arabic prefix like ال/و/ب/ل)
    * ``re:pattern``         -> raw regex applied to normalised text
    * ASCII term             -> word-boundary regex; trailing ``*`` means stem match
    """
    if term.startswith("re:"):
        return re.compile(term[3:])
    whole = term.startswith("=")
    t = norm(term.lstrip("="))
    if t.isascii():
        stem = t.endswith("*")
        t = t.rstrip("*")
        pat = r"(?<![a-z0-9])" + re.escape(t) + (r"[a-z]*" if stem else r"s?") + r"(?![a-z0-9])"
        return re.compile(pat)
    if whole:
        return re.compile(
            r"(?<![" + _AR_LETTER + r"])(?:ال|وال|بال|لل|و|ب|ل|ك|ف)?" + re.escape(t) + r"(?![" + _AR_LETTER + r"])"
        )
    return t


def _m(matcher: Matcher, text: str) -> bool:
    if isinstance(matcher, str):
        return matcher in text
    return matcher.search(text) is not None


def _hits(terms: List[Tuple[str, Matcher]], text: str) -> List[str]:
    return [raw for raw, mt in terms if _m(mt, text)]


def _prep(raw_terms: List[str]) -> List[Tuple[str, Matcher]]:
    out = []
    for t in raw_terms:
        label = t[3:].split("(")[0] if t.startswith("re:") else t.lstrip("=")
        out.append((label, _compile(t)))
    return out


# --------------------------------------------------------------------------- #
# Fraud signal gate
# --------------------------------------------------------------------------- #
STRONG_FRAUD = _prep([
    # Arabic
    "احتيال", "محتال", "نصابين", "نصابون", "عمليات نصب", "النصب والاحتيال", "نصب واحتيال", "النصب الإلكتروني",
    "تصيد إلكتروني", "التصيد الإلكتروني", "التصيد الاحتيالي", "تصيد احتيالي", "هجمات التصيد", "رسائل تصيد",
    "روابط تصيد", "انتحال صفة", "انتحلوا صفة", "انتحل صفة", "منتحل صفة", "منتحلي صفة", "ينتحلون صفة",
    "انتحال شخصية", "رسائل احتيالية", "رسائل مضللة", "روابط مشبوهة", "رابط مشبوه", "موقع وهمي", "مواقع وهمية",
    "متجر وهمي", "متاجر وهمية", "استثمار وهمي", "استثمارات وهمية", "منصة وهمية", "منصات وهمية", "إعلانات وهمية",
    "مكالمات احتيالية", "اتصالات احتيالية", "اتصالات مشبوهة", "الهندسة الاجتماعية", "سرقة الهوية", "سرقة هوية",
    "اختراق حساب", "اختراق حسابات", "الاستيلاء على حساب", "الاستيلاء على الحسابات", "المخططات الهرمية",
    "مخطط هرمي", "بونزي", "توظيف الأموال", "توظيف أموال", "شبكة احتيال", "عصابة احتيال", "تحايل", "خداع الضحايا",
    "الاحتيال المالي", "الاحتيال المصرفي", "الاحتيال الإلكتروني", "جرائم الاحتيال", "=وهمية", "=وهمي", "=وهميين", "=وهميون",
    # English
    "fraud*", "scam*", "phishing", "smishing", "vishing", "ponzi", "pyramid scheme", "impersonat*",
    "identity theft", "sim swap*", "social engineering", "money mule*", "swindl*", "fraudster*",
])

# Financial / cyber context - an article about "fraud" in sport/exams is not ours.
CONTEXT = _prep([
    "=مال", "اموال", "بنك", "بنوك", "مصرف", "=حساب", "حسابات", "بطاق", "ائتمان", "تحويل", "=ريال", "رابط", "روابط",
    "رسائل", "رسالة", "اتصال", "مكالم", "هاتف", "جوال", "واتساب", "الكتروني", "رقمي", "سيبراني", "استثمار",
    "اسهم", "تداول", "عملات", "متجر", "تسوق", "فواتير", "فاتورة", "سداد", "ابشر", "=نفاذ", "هوية", "بيانات",
    "ضحايا", "ضحية", "منصة", "تطبيق", "ضريب", "زكاة", "قرض", "تمويل", "راتب", "وظيفة",
    "money", "bank", "account", "card", "payment", "online", "link", "sms", "phone", "whatsapp", "invest*",
    "crypto*", "victim*", "financial", "cyber", "transfer*", "loan*", "password*", "wallet*",
])

NEGATIVE_CONTEXT = _prep([
    "كرة القدم", "مباراة", "لاعب", "الدوري", "امتحان", "الاختبارات النهائية", "الغش في الاختبار", "match", "football",
])

# Fraud terms that are too generic to qualify an article on their own (they appear in passing in
# awareness / training / consumer-protection pieces). They only count together with a strong term
# or when at least two of them appear in the headline+summary.
WEAK_FRAUD_LABELS = {"الهندسة الاجتماعية", "تحايل", "social engineering", "سرقة هوية", "سرقة الهوية"}

# Other countries: a Saudi *outlet* republishing/covering news about these is not Saudi fraud news
# unless the headline/summary itself names Saudi Arabia.
FOREIGN_TERMS = _prep([
    "=كويت", "=اردن", "اردني", "=مصر", "=بحرين", "=قطر", "=لبنان", "=سوريا", "=عراق", "=يمن", "=مغرب", "=تونس",
    "=جزاير", "=ليبيا", "=فلسطين", "=سودان", "=امارات", "=دبي", "=ابوظبي", "=امريكا", "الولايات المتحده",
    "=بريطانيا", "=تركيا", "=ايران", "=الهند", "=باكستان",
    "kuwait", "jordan", "egypt", "bahrain", "qatar", "uae", "emirates", "dubai", "america", "united states",
    "britain", "india", "pakistan",
])


def has_fraud_signal(text: str) -> bool:
    """Cheap pre-gate on title+summary: is any explicit fraud term present?"""
    return bool(_hits(STRONG_FRAUD, norm(text)))


# --------------------------------------------------------------------------- #
# Saudi relevance
# --------------------------------------------------------------------------- #
# NOTE: terms shared with other countries (المملكة المتحدة, الأمن العام اللبناني, النيابة العامة, kingdom ...) are deliberately NOT
# used as Saudi markers.
SAUDI_TERMS = _prep([
    "السعودية", "سعودي", "re:المملكه(?! (?:المتحده|الاردنيه|المغربيه|الهاشميه|البحرين))", "=رياض", "=جدة", "=مكة", "المدينة المنورة",
    "=دمام", "=ابها", "=تبوك", "=بريدة", "=نجران", "=جازان", "=حائل", "=ينبع", "=طائف", "=احساء", "=قصيم",
    "=عسير", "=جبيل", "رؤية 2030", "=ساما", "البنك المركزي السعودي", "هيئة السوق المالية",
    "الهيئة الوطنية للأمن السيبراني", "رئاسة أمن الدولة", "الزكاة والضريبة والجمارك",
    "هيئة الاتصالات والفضاء", "توكلنا", "بطاقة مدى",
    "saudi", "ksa", "riyadh", "jeddah", "makkah", "mecca", "dammam",
    "saudi central bank", "capital market authority", "national cybersecurity authority",
])

SAUDI_BANK_TERMS = _prep([
    "الراجحي", "الأهلي السعودي", "البنك الأهلي السعودي", "snb", "بنك الرياض", "البنك السعودي الفرنسي", "=ساب", "sabb",
    "البنك السعودي البريطاني", "بنك الجزيرة", "بنك البلاد", "بنك الإنماء", "الإنماء", "stc pay", "اس تي سي باي",
    "urpay", "يور باي", "البنك العربي الوطني", "البنك السعودي للاستثمار", "=ساما",
    "البنك المركزي السعودي", "saudi central bank", "al rajhi", "alinma", "riyad bank", "bank albilad",
    "بطاقة مدى", "mada card",
])

# Markers that only count as "Saudi" when the publisher is itself a Saudi outlet
# (they exist in other Arab countries too, so they are not Saudi evidence on their own).
DOMESTIC_MARKERS = _prep([
    "الأمن العام", "النيابة العامة", "وزارة الداخلية", "وزارة التجارة", "أبشر", "=نفاذ",
    "سداد", "المواطنين والمقيمين", "=مقيم", "=مقيمين", "=وافد", "=وافدين", "رئاسة أمن الدولة",
    "الجوازات", "الأحوال المدنية",
])

BANK_GENERIC = _prep([
    "بنك", "بنوك", "مصرف", "مصرفي", "مصرفية", "بنكي", "بنكية", "عملاء البنك", "عملاء البنوك", "bank", "banking",
])
BANK_RAILS = _prep([
    "حساب بنكي", "حسابات بنكية", "حساب مصرفي", "حسابات مصرفية", "ايبان", "iban", "بطاقة ائتمانية", "بطاقات ائتمانية",
    "بطاقة مصرفية", "بطاقة بنكية", "بطاقات مصرفية", "بطاقات بنكية", "الصراف الآلي", "صراف", "atm", "تحويل بنكي",
    "رمز التحقق", "otp", "الرقم السري", "رقم سري", "الحساب البنكي", "credit card", "debit card",
])
PAYMENT_RAILS = _prep([
    "تحويل", "تحويلات", "دفع", "سداد", "محفظة", "محافظ", "استثمار", "اموال", "=ريال", "payment*", "transfer*", "wallet*",
])

# --------------------------------------------------------------------------- #
# Fraud taxonomy (9 categories requested)
# --------------------------------------------------------------------------- #
CATEGORY_ORDER = [
    "financial_fraud", "bill_payment_fraud", "gov_impersonation", "link_phishing", "investment_fraud",
    "ecommerce_fraud", "social_engineering", "account_identity_theft", "cyber_financial",
]

CATEGORY_AR = {
    "financial_fraud": "احتيال مالي ومصرفي",
    "bill_payment_fraud": "احتيال سداد وفواتير",
    "gov_impersonation": "انتحال صفة جهات حكومية",
    "link_phishing": "احتيال الروابط والتصيد",
    "investment_fraud": "احتيال الاستثمار",
    "ecommerce_fraud": "احتيال التجارة الإلكترونية",
    "social_engineering": "الهندسة الاجتماعية",
    "account_identity_theft": "سرقة الحسابات والهوية",
    "cyber_financial": "الجرائم السيبرانية المرتبطة بالاحتيال المالي",
}

# (term, weight) - weight 3 = strongly indicative, 2 = supporting, 1 = weak
_CATEGORY_RAW: Dict[str, List[Tuple[str, int]]] = {
    "financial_fraud": [
        ("احتيال مالي", 3), ("احتيال مصرفي", 3), ("احتيال بنكي", 3), ("الاحتيال المالي", 3), ("الاحتيال المصرفي", 3),
        ("شبكة احتيال", 3), ("عصابة احتيال", 3), ("نصب مالي", 3), ("bank fraud", 3), ("card fraud", 3),
        ("financial fraud", 3), ("موظف بنك", 3), ("موظفي البنك", 3), ("موظفي البنوك", 3),
        ("بطاقات ائتمانية", 2), ("بطاقة ائتمانية", 2), ("بطاقات مصرفية", 2), ("بطاقة بنكية", 2), ("حسابات بنكية", 2),
        ("رمز التحقق", 2), ("كلمة المرور", 2), ("كلمة السر", 2), ("otp", 2), ("رقم سري", 2), ("الرقم السري", 2),
        ("تحويل بنكي", 2), ("تحويل مالي", 2), ("غسل الأموال", 2), ("ايبان", 2), ("iban", 2), ("أموال العملاء", 2),
        ("بنك", 1), ("مصرف", 1), ("بنوك", 1), ("=حساب", 1), ("=ريال", 1), ("bank*", 1), ("account*", 1),
    ],
    "bill_payment_fraud": [
        ("فاتورة وهمية", 3), ("فواتير وهمية", 3), ("فواتير مزيفة", 3), ("احتيال الفواتير", 3), ("invoice fraud", 3),
        ("bill fraud", 3), ("سداد وهمي", 3), ("مخالفات وهمية", 3), ("مخالفة مرورية وهمية", 3),
        ("سداد", 2), ("فواتير", 2), ("فاتورة", 2), ("مدفوعات", 2), ("دفع الفواتير", 2), ("رسوم مستحقة", 2),
        ("مخالفات مرورية", 2), ("غرامة", 2), ("غرامات", 2), ("بوابة الدفع", 2), ("payment link", 2), ("invoice*", 2),
        ("كهرباء", 1), ("مياه", 1),
    ],
    "gov_impersonation": [
        ("انتحال صفة", 3), ("انتحلوا صفة", 3), ("انتحل صفة", 3), ("منتحل صفة", 3), ("منتحلي صفة", 3),
        ("ينتحلون صفة", 3), ("انتحال شخصية", 3), ("impersonat*", 3), ("government impersonation", 3),
        ("جهات حكومية", 2), ("جهة حكومية", 2), ("موظف حكومي", 2), ("رجال الأمن", 2), ("رجل أمن", 2), ("أبشر", 2),
        ("=نفاذ", 2), ("توكلنا", 2), ("الزكاة والضريبة", 2), ("وزارة الداخلية", 2), ("وزارة التجارة", 2),
        ("وزارة الصحة", 2), ("هيئة الاتصالات", 2), ("government agency", 2),
        ("وزارة", 1), ("هيئة", 1),
    ],
    "link_phishing": [
        ("تصيد إلكتروني", 3), ("التصيد الإلكتروني", 3), ("التصيد الاحتيالي", 3), ("تصيد احتيالي", 3),
        ("هجمات التصيد", 3), ("رسائل تصيد", 3), ("روابط تصيد", 3), ("phishing", 3), ("smishing", 3),
        ("رابط مشبوه", 3), ("روابط مشبوهة", 3), ("رسائل نصية احتيالية", 3), ("رسائل احتيالية", 3),
        ("موقع مزيف", 3), ("مواقع مزيفة", 3), ("موقع وهمي", 3), ("مواقع وهمية", 3), ("صفحة مزيفة", 3),
        ("malicious link", 3), ("fake website", 3),
        ("رابط", 2), ("روابط", 2), ("الرابط", 2), ("رسائل نصية", 2), ("رسالة نصية", 2), ("sms", 2),
        ("qr", 2), ("domain", 2), ("spoof*", 2),
        ("رسائل", 1), ("رسالة", 1), ("موقع", 1),
    ],
    "investment_fraud": [
        ("استثمار وهمي", 3), ("استثمارات وهمية", 3), ("توظيف الأموال", 3), ("توظيف أموال", 3), ("مخطط هرمي", 3),
        ("المخططات الهرمية", 3), ("بونزي", 3), ("ponzi", 3), ("pyramid scheme", 3), ("investment scam", 3),
        ("investment fraud", 3), ("منصات تداول وهمية", 3), ("تداول وهمي", 3), ("عملات رقمية وهمية", 3),
        ("crypto scam", 3), ("forex scam", 3),
        ("هيئة السوق المالية", 2), ("سوق الأسهم", 2), ("تداول الأسهم", 2), ("توصيات الأسهم", 2), ("عملات رقمية", 2),
        ("كريبتو", 2), ("crypto*", 2), ("أرباح مضمونة", 2), ("عوائد مضمونة", 2), ("أرباح خيالية", 2),
        ("غير مرخص", 2), ("بدون ترخيص", 2), ("منصة تداول", 2), ("استثمار", 2), ("استثمارات", 2), ("invest*", 2),
        ("أسهم", 1), ("تداول", 1),
    ],
    "ecommerce_fraud": [
        ("متجر وهمي", 3), ("متاجر وهمية", 3), ("متاجر إلكترونية وهمية", 3), ("تسوق وهمي", 3),
        ("منصة بيع وهمية", 3), ("fake store", 3), ("online shopping scam", 3), ("marketplace scam", 3),
        ("e-commerce fraud", 3), ("ecommerce fraud", 3), ("إعلانات وهمية", 3), ("عروض وهمية", 3), ("سلع وهمية", 3),
        ("شراء وهمي", 3),
        ("تسوق إلكتروني", 2), ("التسوق الإلكتروني", 2), ("متجر إلكتروني", 2), ("متاجر إلكترونية", 2),
        ("التجارة الإلكترونية", 2), ("بائع", 2), ("المشتري", 2), ("منصات البيع", 2), ("حراج", 2), ("التوصيل", 2),
        ("الدفع عند الاستلام", 2), ("online shopping", 2), ("e-commerce", 2), ("ecommerce", 2), ("marketplace", 2),
        ("عروض", 1), ("تخفيضات", 1), ("متجر", 1),
    ],
    "social_engineering": [
        ("الهندسة الاجتماعية", 3), ("social engineering", 3), ("vishing", 3), ("احتيال هاتفي", 3),
        ("الاحتيال الهاتفي", 3), ("مكالمات احتيالية", 3), ("اتصالات احتيالية", 3), ("اتصالات مشبوهة", 3),
        ("مكالمات مشبوهة", 3), ("اتصال هاتفي احتيالي", 3), ("خداع الضحايا", 3), ("استدراج الضحايا", 3),
        ("استدراج", 3), ("إيهام", 3),
        ("اتصال هاتفي", 2), ("اتصالات هاتفية", 2), ("مكالمة", 2), ("مكالمات", 2), ("واتساب", 2), ("whatsapp", 2),
        ("تيليجرام", 2), ("telegram", 2), ("سناب", 2), ("منصات التواصل", 2), ("وسائل التواصل", 2),
        ("أرقام مجهولة", 2), ("ضحايا", 2), ("الضحية", 2), ("جوائز", 2), ("الفوز", 2), ("مساعدة مالية", 2),
        ("قروض", 2), ("phone call", 2), ("scammer*", 2), ("lottery", 2), ("prize*", 2),
        ("مجهول", 1), ("هاتف", 1),
    ],
    "account_identity_theft": [
        ("سرقة الهوية", 3), ("سرقة هوية", 3), ("انتحال هوية", 3), ("سرقة الحسابات", 3), ("اختراق حساب", 3),
        ("اختراق حسابات", 3), ("الاستيلاء على حساب", 3), ("الاستيلاء على الحسابات", 3), ("sim swap*", 3),
        ("تبديل الشريحة", 3), ("identity theft", 3), ("account takeover", 3), ("credential theft", 3),
        ("سرقة بيانات", 3), ("سرقة البيانات الشخصية", 3), ("سرقة بيانات البطاقة", 3),
        ("اختراق", 2), ("بيانات شخصية", 2), ("بيانات الدخول", 2), ("بيانات الحساب", 2), ("اسم المستخدم", 2),
        ("شريحة", 2), ("data breach", 2), ("credentials", 2), ("hacked account", 2), ("stolen account", 2),
        ("بيانات", 1), ("هوية", 1),
    ],
    "cyber_financial": [
        ("جرائم سيبرانية مالية", 3), ("برمجيات خبيثة مصرفية", 3), ("banking trojan", 3), ("برنامج الفدية", 3),
        ("هجوم الفدية", 3), ("ransomware", 3), ("malware", 3), ("برمجيات خبيثة", 3), ("برمجية خبيثة", 3),
        ("أحصنة طروادة", 3), ("هجوم سيبراني", 3), ("هجمات سيبرانية", 3), ("جرائم سيبرانية", 3),
        ("الجرائم المعلوماتية", 3), ("نظام مكافحة الجرائم المعلوماتية", 3), ("cybercrime", 3), ("cyber fraud", 3),
        ("cyber attack*", 3),
        ("سيبراني", 2), ("سيبرانية", 2), ("إلكتروني", 2), ("معلوماتي", 2), ("فيروس", 2), ("تطبيق مشبوه", 2),
        ("تطبيقات مشبوهة", 2), ("تطبيقات خبيثة", 2), ("apk", 2), ("remote access", 2), ("anydesk", 2),
        ("التحكم عن بعد", 2), ("تثبيت تطبيق", 2),
        ("رقمي", 1), ("تقنية", 1),
    ],
}
CATEGORY_TERMS: Dict[str, List[Tuple[str, int, Matcher]]] = {
    cat: [(t, w, _compile(t)) for t, w in terms] for cat, terms in _CATEGORY_RAW.items()
}

# --------------------------------------------------------------------------- #
# Extraction dictionaries (all matched on normalised text)
# --------------------------------------------------------------------------- #
CHANNELS: Dict[str, List[str]] = {
    "رسائل نصية (SMS)": ["رسائل نصية", "رسالة نصية", "sms", "smishing"],
    "واتساب": ["واتساب", "whatsapp"],
    "اتصال هاتفي": ["اتصال هاتفي", "اتصالات هاتفية", "مكالمة", "مكالمات", "vishing", "phone call", "اتصال"],
    "بريد إلكتروني": ["بريد إلكتروني", "البريد الإلكتروني", "email", "e-mail"],
    "منصات التواصل الاجتماعي": ["منصات التواصل", "وسائل التواصل", "سناب", "انستقرام", "إنستغرام", "تيك توك", "تويتر",
                                "فيسبوك", "منصة x", "تيليجرام", "telegram", "social media", "instagram", "facebook",
                                "snapchat", "tiktok"],
    "روابط / مواقع مزيفة": ["رابط", "روابط", "موقع وهمي", "مواقع وهمية", "موقع مزيف", "مواقع مزيفة", "fake website",
                            "phishing link", "malicious link"],
    "تطبيقات جوال": ["تطبيق", "تطبيقات", "apk"],
    "متاجر / أسواق إلكترونية": ["متجر", "متاجر", "تسوق", "حراج", "marketplace"],
    "بطاقات / صراف آلي": ["بطاقة", "بطاقات", "الصراف", "atm"],
    "رمز QR": ["qr", "رمز الاستجابة السريعة"],
}

TARGETS: Dict[str, List[str]] = {
    "كبار السن": ["كبار السن", "المسنين", "elderly"],
    "المستثمرون": ["مستثمر", "المستثمرين", "investors"],
    "عملاء البنوك": ["عملاء البنوك", "عملاء البنك", "bank customers", "bank clients"],
    "المتسوقون / المشترون": ["المتسوقين", "المتسوق", "المشترين", "shoppers", "buyers"],
    "المواطنون والمقيمون": ["المواطنين", "المقيمين", "مواطنون", "مقيمون", "residents", "citizens"],
    "الشركات والمنشآت": ["الشركات", "المنشآت", "منشآت", "companies", "businesses"],
    "الباحثون عن عمل": ["الباحثين عن عمل", "الوظائف", "job seekers"],
    "مستخدمو الخدمات الحكومية الرقمية": ["أبشر", "=نفاذ", "توكلنا"],
}

AGENCIES: Dict[str, List[str]] = {
    "البنك المركزي السعودي (ساما)": ["البنك المركزي السعودي", "=ساما", "saudi central bank", "=sama"],
    "هيئة السوق المالية": ["هيئة السوق المالية", "capital market authority"],
    "الأمن العام": ["الأمن العام", "public security"],
    "النيابة العامة": ["النيابة العامة", "public prosecution"],
    "الهيئة الوطنية للأمن السيبراني": ["الهيئة الوطنية للأمن السيبراني", "national cybersecurity authority"],
    "رئاسة أمن الدولة": ["رئاسة أمن الدولة"],
    "وزارة الداخلية": ["وزارة الداخلية", "ministry of interior"],
    "وزارة التجارة": ["وزارة التجارة", "ministry of commerce"],
    "هيئة الاتصالات والفضاء والمعلومات": ["هيئة الاتصالات", "citc", "cst"],
    "هيئة الزكاة والضريبة والجمارك": ["الزكاة والضريبة", "zatca"],
    "هيئة الرقابة ومكافحة الفساد": ["نزاهة", "هيئة الرقابة ومكافحة الفساد"],
    "شرطة المنطقة": ["شرطة منطقة", "شرطة محافظة", "police"],
}

EVENT_ENFORCEMENT = _prep(["القبض", "ضبط", "الإطاحة", "إطاحة", "إيقاف", "أوقفت", "ألقت القبض", "إحالة", "أحالت",
                           "توقيف", "arrest*", "busted", "detain*", "apprehend*"])
EVENT_COURT = _prep(["محكمة", "إدانة", "السجن", "بالسجن", "sentenc*", "convict*"])
EVENT_WARNING = _prep(["تحذر", "حذرت", "يحذر", "تحذير", "تنبيه", "تنبه", "احذر", "تحذيرات", "warn*", "advisory", "alert*"])
EVENT_AWARENESS = _prep(["توعية", "التوعوية", "نصائح", "إرشادات", "كيف تحمي", "طرق الوقاية", "awareness", "tips"])

METHOD_VERBS = _prep(["عبر", "من خلال", "باستخدام", "عن طريق", "بواسطة", "بانتحال", "بادعاء", "إيهام", "أوهم",
                      "استدراج", "استدرج", "الإيقاع", "إيقاع", "الحصول على", "سحب", "تحويل", "إرسال", "أرسلوا",
                      "تواصل", "اتصل", "using", "via", "through", "posing as", "pretend*", "claim*", "lured",
                      "tricked"])

ARABIC_UNITS = r"(?:متهم|متهمين|شخص|أشخاص|اشخاص|مقيم|مقيمين|وافد|وافدين|يمني|يمنيين|سوداني|سودانيين|مواطن|مواطنين|ضحية|ضحايا|حساب|حسابات|رسالة|رسائل|متجر|متاجر|موقع|مواقع|رابط|روابط|عصابة|شبكة|شبكات|بلاغ|بلاغات)"
RE_COUNT_AR = re.compile(r"(\d[\d,]*)\s+" + ARABIC_UNITS)
RE_COUNT_EN = re.compile(r"\b(\d[\d,]*)\s+(?:suspects?|people|persons?|individuals?|victims?|accounts?|messages?|websites?|"
                         r"links?|expatriates?|residents?|citizens?|arrests?|cases?)\b", re.I)
RE_AMOUNT_AR = re.compile(r"(\d[\d,\.]*)\s*(?:مليون|مليار|ألف|الف)?\s*(?:ريال|ر\.س|دولار|درهم)")
RE_AMOUNT_EN = re.compile(r"\b(?:SR|SAR|USD|\$)\s?(\d[\d,\.]*)\s*(?:million|billion|thousand)?|"
                          r"(\d[\d,\.]*)\s*(?:million|billion|thousand)?\s*(?:riyals?|SAR|dollars?)", re.I)
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?؟؛])\s+|\n+")

# --------------------------------------------------------------------------- #
# Templates: indicators, recommendations and detection rules (SYSTEM-GENERATED)
# --------------------------------------------------------------------------- #
INDICATORS_TEMPLATE: Dict[str, List[str]] = {
    "financial_fraud": ["اتصال/رسالة من جهة تنتحل صفة البنك تطلب رمز التحقق أو الرقم السري",
                        "تحويلات واردة متعددة من أطراف جديدة ثم تحويل سريع للمبلغ إلى الخارج (حسابات وسيطة)",
                        "عملية جديدة بمبلغ مختلف عن نمط العميل خلال دقائق من تغيير بيانات الاتصال"],
    "bill_payment_fraud": ["رابط دفع لا ينتمي لنطاق الجهة الرسمية أو مزود السداد المعتمد",
                           "فاتورة/مخالفة بمبلغ أو رقم مرجعي لا يطابق السجلات الرسمية",
                           "إلحاح زمني في الرسالة (سداد فوري لتجنب الغرامة)"],
    "gov_impersonation": ["رسالة/اتصال يذكر جهة حكومية ثم يطلب بيانات دخول أو رمز تحقق",
                          "اسم مرسل أو نطاق يحاكي اسم جهة حكومية بفارق حرف (typosquatting)",
                          "طلب تحويل مبلغ لتجنب إجراء نظامي مزعوم"],
    "link_phishing": ["نطاق حديث التسجيل أو يحاكي نطاق البنك",
                      "رابط مختصر داخل رسالة نصية يطلب تحديث البيانات",
                      "تسجيل دخول من جهاز/IP جديد بعد نقر رابط من رسالة نصية"],
    "investment_fraud": ["وعود بعوائد مضمونة أو مرتفعة بشكل غير واقعي",
                         "جهة غير مدرجة ضمن الجهات المرخصة من هيئة السوق المالية",
                         "تحويلات متعددة من أفراد مختلفين إلى حساب/محفظة واحدة بوصف استثمار أو توظيف أموال"],
    "ecommerce_fraud": ["متجر حديث الإنشاء بأسعار أقل بكثير من السوق",
                        "دفع مسبق عبر تحويل مباشر دون وسيط ضمان",
                        "ارتفاع نسبة الشكاوى/النزاعات للتاجر خلال فترة قصيرة"],
    "social_engineering": ["مكالمة طويلة يتبعها إضافة مستفيد جديد وتحويل مبلغ كبير",
                           "ضغط نفسي/إلحاح (حظر الحساب، ربح جائزة، مساعدة عاجلة)",
                           "طلب تثبيت تطبيق أو مشاركة الشاشة أثناء الاتصال"],
    "account_identity_theft": ["تفعيل التطبيق المصرفي على جهاز جديد مع تغيير رقم الجوال",
                               "تبديل شريحة SIM قبل عمليات حساسة",
                               "رفع حدود التحويل ثم تحويل فوري بعد تسجيل دخول غير معتاد"],
    "cyber_financial": ["تطبيق خارج المتجر الرسمي (APK) مثبت على جهاز العميل",
                        "مؤشرات تحكم عن بعد أو تراكب شاشة (overlay) أثناء جلسة التطبيق المصرفي",
                        "أول عملية تحويل كبيرة بعد تثبيت تطبيق جديد أو منح صلاحيات إمكانية الوصول"],
}

RECOMMENDATIONS: Dict[str, List[str]] = {
    "financial_fraud": ["لا تشارك رمز التحقق أو الرقم السري مع أي جهة حتى لو ادّعت أنها البنك",
                        "تحقق من الاتصال عبر الرقم الرسمي للبنك أو التطبيق الرسمي فقط",
                        "فعّل التنبيهات الفورية للعمليات وحدود التحويل"],
    "bill_payment_fraud": ["سدّد الفواتير عبر التطبيقات والمنصات الرسمية فقط (سداد، أبشر)",
                           "راجع المخالفات والفواتير من الجهة المعنية مباشرة وليس من الروابط الواردة",
                           "تجاهل الرسائل التي تطلب السداد الفوري"],
    "gov_impersonation": ["الجهات الحكومية لا تطلب بيانات الدخول أو رموز التحقق عبر الرسائل أو الاتصال",
                          "تحقق من هوية المتصل عبر القنوات الرسمية للجهة",
                          "بلّغ عن الرقم أو الرسالة للجهة المختصة"],
    "link_phishing": ["لا تنقر الروابط الواردة في رسائل غير متوقعة",
                      "افتح الموقع الرسمي بكتابة العنوان يدوياً",
                      "بلّغ عن الرسائل المشبوهة لمزود الخدمة وللجهات المختصة"],
    "investment_fraud": ["تحقق من ترخيص الجهة لدى هيئة السوق المالية قبل أي استثمار",
                         "احذر من العوائد المضمونة أو المرتفعة بشكل غير واقعي",
                         "لا تحوّل أموالاً لحسابات أفراد بدعوى الاستثمار"],
    "ecommerce_fraud": ["تعامل مع المتاجر الموثقة (سجل تجاري / معروف) وتحقق من تقييماتها",
                        "فضّل الدفع عند الاستلام أو وسائل دفع تتيح استرداد المبلغ",
                        "احذر من الأسعار المنخفضة بشكل مبالغ فيه"],
    "social_engineering": ["لا تُجرِ عمليات مالية أثناء مكالمة يطلبها المتصل",
                           "لا تثبت تطبيقات ولا تشارك شاشة جوالك بطلب من مجهول",
                           "أنهِ المكالمة وتواصل مع الجهة عبر قناتها الرسمية"],
    "account_identity_theft": ["فعّل التحقق بخطوتين وغيّر كلمات المرور دورياً",
                               "راقب أي فقدان مفاجئ لخدمة الشريحة (قد يدل على تبديل SIM)",
                               "لا تشارك صور الهوية أو بياناتها مع جهات غير موثوقة"],
    "cyber_financial": ["ثبّت التطبيقات من المتاجر الرسمية فقط",
                        "لا تمنح صلاحيات إمكانية الوصول أو التحكم عن بعد لتطبيقات غير موثوقة",
                        "حدّث نظام الجوال وفعّل الحماية"],
}

# Detection-rule templates. They are HYPOTHESES derived from the category (is_inference).
RULE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "financial_fraud": {
        "code": "FIN",
        "name_ar": "حساب وسيط (Mule) بنمط استقبال وتحويل سريع",
        "hypothesis_ar": "حساب يستقبل تحويلات من 3 أطراف جديدة أو أكثر ثم يحوّل ≥ 80% من المبلغ خلال 60 دقيقة هو حساب وسيط محتمل.",
        "logic": "COUNT(DISTINCT sender WHERE sender_first_seen > now-30d) >= 3 within 24h AND outbound_amount / inbound_amount >= 0.8 within 60m",
        "required_data": ["transactions(amount, ts, counterparty_id, channel)", "counterparty first-seen date",
                          "account age / KYC profile", "fraud-ops confirmed labels"],
    },
    "bill_payment_fraud": {
        "code": "BILL",
        "name_ar": "دفع فاتورة عبر قناة غير رسمية أو لمفوتر غير معتاد",
        "hypothesis_ar": "دفعة فاتورة أُطلقت من رابط خارجي (referrer ≠ تطبيق البنك/سداد) أو لمفوتر لم يتعامل معه العميل سابقاً ترتبط بنسبة احتيال أعلى.",
        "logic": "payment.type = 'bill' AND (referrer NOT IN official_set OR biller_id NOT IN customer_history) AND amount > P90(customer_bill_amount)",
        "required_data": ["bill payment events(biller_id, amount, referrer)", "SADAD biller registry",
                          "customer bill history 12m", "confirmed fraud labels"],
    },
    "gov_impersonation": {
        "code": "GOV",
        "name_ar": "طلب OTP بعد اتصال/رسالة تنتحل جهة حكومية",
        "hypothesis_ar": "طلب رمز تحقق أو تغيير بيانات اتصال خلال 15 دقيقة من رسالة/اتصال يذكر جهة حكومية يرفع احتمال الاحتيال.",
        "logic": "otp_request OR contact_info_change within 15m AFTER inbound_sms WHERE sender_alias LIKE ANY(gov_brand_aliases) AND sender NOT IN verified_sender_ids",
        "required_data": ["SMS sender-ID verification feed (من مزود الاتصالات)", "OTP request logs",
                          "contact-info change events", "gov brand alias list"],
    },
    "link_phishing": {
        "code": "PHISH",
        "name_ar": "دخول من جهاز جديد بعد نقر رابط مشبوه",
        "hypothesis_ar": "تسجيل دخول من جهاز/IP جديد خلال 30 دقيقة من نقر العميل رابطاً من نطاق حديث أو مشابه لنطاق البنك ثم تغيير بيانات الحساب = اختراق عبر تصيد.",
        "logic": "new_device_login within 30m AFTER click(domain_age < 30d OR levenshtein(domain, bank_domain) <= 2) AND (password_reset OR payee_added)",
        "required_data": ["login/device fingerprint logs", "SMS/link click telemetry أو بلاغات العملاء",
                          "domain age + similarity feed (WHOIS/threat feed)", "payee/credential change events"],
    },
    "investment_fraud": {
        "code": "INV",
        "name_ar": "تجميع تحويلات استثمارية في حساب فرد غير مرخص",
        "hypothesis_ar": "حساب فرد يتلقى ≥ 10 تحويلات من مرسلين مختلفين خلال 7 أيام بوصف 'استثمار/توظيف/أرباح' يرجّح مخطط استثمار احتيالي.",
        "logic": "COUNT(DISTINCT sender) >= 10 within 7d AND description MATCHES /(استثمار|توظيف|ارباح|crypto)/ AND beneficiary NOT IN cma_licensed_entities",
        "required_data": ["transfer descriptions", "beneficiary account type (individual/business)",
                          "CMA licensed-entities list", "confirmed fraud labels"],
    },
    "ecommerce_fraud": {
        "code": "ECOM",
        "name_ar": "تاجر جديد بنسبة نزاعات مرتفعة",
        "hypothesis_ar": "تاجر عمره < 30 يوماً مع مدفوعات متعددة من عملاء مختلفين ونسبة نزاعات/استرداد > 5% يرجّح متجراً وهمياً.",
        "logic": "merchant_age < 30d AND COUNT(DISTINCT cardholder) >= 20 within 7d AND dispute_rate >= 0.05",
        "required_data": ["merchant onboarding date + MCC", "card/payment transactions", "chargeback & dispute records",
                          "customer complaints"],
    },
    "social_engineering": {
        "code": "SOC",
        "name_ar": "مستفيد جديد وتحويل كبير بعد جلسة طويلة",
        "hypothesis_ar": "إضافة مستفيد جديد يتبعها تحويل ≥ 5× متوسط العميل خلال 30 دقيقة، مع جلسة تطبيق طويلة غير معتادة، يرجّح تعرّض العميل لهندسة اجتماعية.",
        "logic": "payee_added within 30m BEFORE transfer.amount >= 5 * customer_avg_transfer AND session_duration > P95(customer_session)",
        "required_data": ["payee-add events", "transfer amounts + customer baseline", "session telemetry",
                          "customer segment (age)", "confirmed fraud labels"],
    },
    "account_identity_theft": {
        "code": "ATO",
        "name_ar": "استيلاء على حساب: جهاز جديد + رفع الحدود + تحويل فوري",
        "hypothesis_ar": "تفعيل التطبيق على جهاز جديد مع تغيير رقم الجوال أو رفع الحدود ثم تحويل فوري خلال 24 ساعة يرجّح استيلاءً على الحساب.",
        "logic": "device_bind_new AND (phone_change OR limit_increase) within 24h BEFORE transfer.amount >= X",
        "required_data": ["device binding events", "SIM-change signal من مزود الاتصالات", "limit-change events",
                          "transfers"],
    },
    "cyber_financial": {
        "code": "CYB",
        "name_ar": "مؤشرات تحكم عن بعد أثناء جلسة مصرفية",
        "hypothesis_ar": "جلسة تطبيق مصرفي مع مؤشرات تحكم عن بعد/تراكب شاشة تتبعها عملية تحويل كبيرة ترجّح تطبيقاً خبيثاً.",
        "logic": "app_session.remote_access_flag OR overlay_detected AND transfer.amount >= X within session",
        "required_data": ["mobile app runtime telemetry (RASP)", "device risk signals", "installed-app risk list",
                          "transfers"],
    },
}

# --------------------------------------------------------------------------- #
# Source evidence tiers
# --------------------------------------------------------------------------- #
SAUDI_PUBLISHER_DOMAINS = (
    "okaz.com.sa", "sabq.org", "alwatan.com.sa", "al-madina.com", "alyaum.com", "argaam.com", "saudigazette.com.sa",
    "alriyadh.com", "aleqt.com", "spa.gov.sa", "arabnews.com", "alarabiya.net", "aawsat.com", "mubasher.info",
    "almowaten.net", "twasul.info", "akhbaar24.argaam.com", "sauress.com", "maaal.com", "alekhbariya.net",
)


def _evidence_base(domain: str, official: bool) -> float:
    d = (domain or "").lower()
    if official or d.endswith(".gov.sa"):
        return 0.92
    if any(d.endswith(x) for x in SAUDI_PUBLISHER_DOMAINS) or d.endswith(".sa"):
        return 0.80
    if d:
        return 0.68
    return 0.60


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "\u0600" <= c <= "\u06FF") / len(letters)


def _split_sentences(text: str) -> List[str]:
    out = []
    for s in _SENT_SPLIT.split(text or ""):
        s = s.strip(" \t-–—•·")
        if 25 <= len(s) <= 450:
            out.append(s)
    return out


def _uniq(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _group_hits(mapping: Dict[str, List[str]], text_norm: str) -> List[str]:
    found = []
    for label, terms in mapping.items():
        if any(_m(_compile(t), text_norm) for t in terms):
            found.append(label)
    return found


def _extract_numbers(raw_text: str) -> Dict[str, List[str]]:
    t = (raw_text or "").translate(_AR_DIGITS)
    counts = [f"{m.group(0).strip()}" for m in RE_COUNT_AR.finditer(t)] + \
             [m.group(0).strip() for m in RE_COUNT_EN.finditer(t)]
    amounts = [m.group(0).strip() for m in RE_AMOUNT_AR.finditer(t)] + \
              [m.group(0).strip() for m in RE_AMOUNT_EN.finditer(t)]
    return {"counts": _uniq(counts)[:6], "amounts": _uniq(amounts)[:6]}


def _score_categories(text_norm: str, title_norm: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    scores: Dict[str, int] = {}
    hits: Dict[str, List[str]] = {}
    for cat, terms in CATEGORY_TERMS.items():
        s, h = 0, []
        for raw, w, mt in terms:
            if _m(mt, text_norm):
                s += w
                h.append(raw)
                if w >= 3 and _m(mt, title_norm):
                    s += 2  # strong term in the headline
        scores[cat] = s
        hits[cat] = h
    return scores, hits


def _pick_category(scores: Dict[str, int], text_norm: str) -> str:
    best = max(CATEGORY_ORDER, key=lambda c: (scores[c], -CATEGORY_ORDER.index(c)))
    if scores[best] == 0:
        return "financial_fraud"
    return best


def _event_type(text_norm: str) -> str:
    if _hits(EVENT_ENFORCEMENT, text_norm) or _hits(EVENT_COURT, text_norm):
        return "confirmed_incident"
    if _hits(EVENT_WARNING, text_norm):
        return "warning"
    if _hits(EVENT_AWARENESS, text_norm):
        return "guidance"
    return "reported_incident"


def _bank_relevance(text_norm: str, category: str, is_saudi: bool) -> Dict[str, Any]:
    named = _hits(SAUDI_BANK_TERMS, text_norm)
    generic = _hits(BANK_GENERIC, text_norm)
    rails = _hits(BANK_RAILS, text_norm)
    pay = _hits(PAYMENT_RAILS, text_norm)
    if named or (rails and is_saudi):
        level, basis = "high", "stated_in_source"
        reason = ("ذُكر في المصدر: " + "، ".join(_uniq(named + rails)[:5]))
    elif generic and is_saudi:
        level, basis = "high", "stated_in_source"
        reason = "المصدر يذكر البنوك/القطاع المصرفي صراحة في سياق سعودي: " + "، ".join(_uniq(generic)[:3])
    elif pay or category in ("financial_fraud", "bill_payment_fraud", "investment_fraud", "account_identity_theft"):
        level, basis = "medium", "inferred"
        reason = ("استنتاج النظام: المخطط يعتمد على تحويلات/مدفوعات يُرجَّح مرورها عبر حسابات بنكية سعودية، "
                  "ولم يُذكر بنك أو قناة مصرفية صراحة في المصدر")
    else:
        level, basis = "low", "inferred"
        reason = "لا توجد إشارة في المصدر إلى قنوات مصرفية؛ الارتباط بالقطاع المصرفي ضعيف أو غير مثبت"
    return {"level": level, "basis": basis, "reason": reason,
            "named_entities": _uniq(named)[:6], "bank_terms": _uniq(generic + rails)[:6]}


# --------------------------------------------------------------------------- #
# Content type: WHAT KIND of publication is this?
#   confirmed_incident  - the source reports a concrete fraud event with enforcement / judicial action
#                         (arrest, seizure, charge, court)
#   reported_incident   - a concrete event is described (victim, attempt, occurrence) but there is no
#                         enforcement wording: details or confirmation are incomplete
#   fraud_warning       - a warning or trend piece about fraud, without a concrete incident
#   awareness_campaign  - a campaign / initiative / training whose subject is fraud
#   general_cybersecurity - cyber awareness/training content where fraud is incidental
#   unrelated           - no fraud signal at all (kept stored, never counted)
# --------------------------------------------------------------------------- #
CT_CONFIRMED = "confirmed_incident"
CT_REPORTED = "reported_incident"
CT_WARNING = "fraud_warning"
CT_AWARENESS = "awareness_campaign"
CT_GENERAL = "general_cybersecurity"
CT_UNRELATED = "unrelated"
CONTENT_TYPES = (CT_CONFIRMED, CT_REPORTED, CT_WARNING, CT_AWARENESS, CT_GENERAL, CT_UNRELATED)
CONTENT_TYPE_AR = {
    CT_CONFIRMED: "حادثة احتيال مؤكدة",
    CT_REPORTED: "حادثة احتيال مبلَّغ عنها",
    CT_WARNING: "تحذير احتيالي",
    CT_AWARENESS: "حملة توعية",
    CT_GENERAL: "أمن سيبراني عام",
    CT_UNRELATED: "غير ذي صلة",
}
# Types that describe a concrete event: only these can form an incident record.
INCIDENT_CONTENT_TYPES = (CT_CONFIRMED, CT_REPORTED)
# Content types shown to users as fraud material.
VISIBLE_CONTENT_TYPES = (CT_CONFIRMED, CT_REPORTED, CT_WARNING, CT_AWARENESS)

ENFORCEMENT_STEMS = _prep([
    "قبض", "اطاح", "ضبط", "توقيف", "احال", "محكمة", "ادان", "بالسجن", "حكم عليه",
    "اوقفت", "اوقفوا", "اوقفته", "اوقفتهم",
    # "توقف" alone means "stopped"; it only signals an arrest when a person is its object
    r"re:(?:توقف|يوقف|توقفت|توقفان) (?:\d+|\w*(?:مقيم|متهم|شخص|اشخاص|مواطن|وافد|محتال|نصاب|عصابه|مشتبه))",
    "arrest*", "busted", "detain*", "apprehend*", "sentenc*", "convict*", "charged",
])
# a concrete occurrence described by the source (headline/summary only)
INCIDENT_CUES = _prep([
    "واقعة", "الوقائع", "وقع ضحية", "وقعوا ضحية", "وقعت ضحية", "حاول شخص", "حاول مجهول", "استولى", "استولوا",
    "سرقوا", "تعرض لعملية", "تعرضت لعملية", "تعرضوا لعملية",
])
WARNING_CUES = _prep([
    "تحذر", "حذر", "تحذير", "تنبيه", "تنبه", "احذر", "مخاطر", "تتزايد", "يتزايد", "تزايد", "تزداد", "ارتفاع",
    "تنشط", "تستهدف", "يستهدف", "تنامي", "تفشي", "تصاعد", "warn*", "advisory", "alert*",
])
# NOTE: "re:" patterns run on normalised text, so they are written in normalised form.
CAMPAIGN_CUES = _prep([
    "مبادرة", "مبادرات", "توعي", "توعو", "=وعي", "تثقيف", "محاضرة", "ورشة", "ورش عمل", "دورة تدريبية",
    "برنامج تدريبي", "مؤتمر", "ملتقى", "ندوة", "نصائح", "ارشادات",
    r"re:(?:تطلق|اطلاق|اطلقت|تدشن|تدشين|يطلق|تطلقان)\w* (?:ال)?حمله",
    r"re:حمله (?:توع|تثقيف|وطني|اعلامي)",
    "campaign*", "awareness", "initiative*", "workshop*", "webinar*",
])
CYBER_CUES = _prep(["سيبراني", "سيبرانية", "معلوماتي", "cyber*"])


def relevant_text(title: str, summary: str, body: str) -> str:
    """Headline + summary + the body sentences that are themselves about the fraud.

    Page boilerplate (menus, related links, sidebars of unrelated news) never qualifies, so numbers
    and places mentioned there cannot leak into an article's facts.
    """
    head = f"{(title or '').strip()}. {(summary or '').strip()}".strip()
    head_norm = norm(head)
    allow_enf = bool(_hits(ENFORCEMENT_STEMS, head_norm))
    keep = [head]
    for s in _split_sentences(body or ""):
        sn = norm(s)
        if _hits(STRONG_FRAUD, sn) or (allow_enf and _hits(ENFORCEMENT_STEMS, sn)):
            keep.append(s)
    return "\n".join(_uniq(keep))


def classify_content_type(title: str, summary: str, rel_raw: str = "") -> Tuple[str, List[str]]:
    """Decide the content type of an article that passed the fraud-signal gate.

    Decision order (first match wins), evaluated on the headline/summary so that body boilerplate
    cannot change the result:
      1. enforcement or a described occurrence                         -> confirmed_incident
      2. campaign/training wording in the headline                     -> awareness_campaign if the headline
                                                                          itself names fraud, else general_cybersecurity
      3. warning/trend wording                                         -> fraud_warning
      4. campaign/training wording in the summary                      -> awareness_campaign / general_cybersecurity
      5. cyber wording without fraud in the headline                   -> general_cybersecurity
      6. otherwise (no concrete event reported)                        -> fraud_warning
    """
    title_norm = norm(title)
    head_norm = norm(f"{title} {summary}")
    title_strong = [s for s in _uniq(_hits(STRONG_FRAUD, title_norm)) if s not in WEAK_FRAUD_LABELS]
    enforcement = _uniq(_hits(ENFORCEMENT_STEMS, head_norm))
    incident_cues = _uniq(_hits(INCIDENT_CUES, head_norm))
    title_campaign = _uniq(_hits(CAMPAIGN_CUES, title_norm))
    head_campaign = _uniq(_hits(CAMPAIGN_CUES, head_norm))
    head_warning = _uniq(_hits(WARNING_CUES, head_norm))
    cyber_title = _uniq(_hits(CYBER_CUES, title_norm))

    if enforcement:
        return CT_CONFIRMED, ["enforcement:" + "|".join(enforcement[:3])]
    if incident_cues:
        return CT_REPORTED, ["described_occurrence_without_enforcement:" + "|".join(incident_cues[:3])]
    if title_campaign:
        if title_strong:
            return CT_AWARENESS, ["campaign_in_title:" + "|".join(title_campaign[:3]), "fraud_in_title"]
        return CT_GENERAL, ["campaign_in_title:" + "|".join(title_campaign[:3]), "no_fraud_term_in_title"]
    if head_warning:
        return CT_WARNING, ["warning_wording:" + "|".join(head_warning[:3])]
    if head_campaign:
        if title_strong:
            return CT_AWARENESS, ["campaign_in_summary:" + "|".join(head_campaign[:3]), "fraud_in_title"]
        return CT_GENERAL, ["campaign_in_summary:" + "|".join(head_campaign[:3]), "no_fraud_term_in_title"]
    if cyber_title and not title_strong:
        return CT_GENERAL, ["cyber_wording_without_fraud_in_title"]
    return CT_WARNING, ["no_concrete_event_reported"]


# --------------------------------------------------------------------------- #
# Stated fraud methods. A tag is produced ONLY when the source text itself contains the wording;
# the quote is the supporting sentence. Patterns are written in normalised form.
# --------------------------------------------------------------------------- #
METHOD_LEXICON: List[Tuple[str, str, str]] = [
    ("victim_data_misuse", "استخدام بيانات الضحايا", r"بيانات (?:ال)?(?:ضحاياهم|ضحاياه|ضحايا|ضحيتهم)"),
    ("online_purchase", "شراء سلع عبر المتاجر الإلكترونية",
     r"شراء[^.،\n]{0,70}(?:عبر|من خلال) (?:ال)?متاجر (?:ال)?الكترونيه"),
    ("resale_or_shipping", "بيع السلع أو شحنها", r"بيعها او شحنها|بيعها وشحنها|اعاده بيع"),
    ("illicit_transfer", "تحويل أموال بطرق غير مشروعة أو إلى الخارج",
     r"تحويل\w* [^.،\n]{0,50}(?:الي الخارج|غير (?:ال)?مشروع)"),
    ("impersonation", "انتحال صفة أو هوية", r"انتحال|ينتحل|انتحل|منتحل"),
    ("suspicious_link", "رابط أو موقع مشبوه",
     r"رابط\w* (?:الكتروني\w* )?مشبوه|روابط تصيد|رابط تصيد|رابط احتيالي|مواقع وهميه|موقع وهمي|موقع مزيف"),
    ("phone_call", "اتصال أو مكالمة احتيالية", r"(?:اتصال|اتصالات|مكالمه|مكالمات) (?:هاتفي\w*|احتيالي\w*|مشبوه\w*)"),
    ("otp_or_credentials", "طلب رمز التحقق أو بيانات الدخول", r"رمز التحقق|\botp\b|الرقم السري|كلمه المرور|كلمه السر"),
    ("fraudulent_ads", "إعلانات احتيالية أو وهمية", r"اعلان\w* (?:احتيالي\w*|وهمي\w*)|اعلانات احتياليه"),
    ("fake_store", "متجر إلكتروني وهمي", r"متاجر? (?:الكترونيه )?وهميه?"),
    ("fake_investment", "استثمار أو منصة تداول وهمية", r"استثمار\w* وهمي\w*|منصات? (?:تداول )?وهميه?|توظيف (?:ال)?اموال"),
    ("pyramid_scheme", "مخطط هرمي / بونزي", r"مخطط\w* هرمي\w*|بونزي|\bponzi\b"),
    ("social_engineering", "الهندسة الاجتماعية", r"الهندسه الاجتماعيه"),
    ("ai_assisted", "استخدام تقنيات الذكاء الاصطناعي", r"الذكاء الاصطناعي"),
]
_METHOD_RES = [(tag, label, re.compile(rx)) for tag, label, rx in METHOD_LEXICON]


def extract_method_tags(rel_raw: str) -> List[Dict[str, str]]:
    """Methods the source itself states, each with the sentence that states it (no inference)."""
    sentences = _split_sentences(rel_raw or "")
    out: List[Dict[str, str]] = []
    for tag, label, rx in _METHOD_RES:
        for s in sentences:
            if rx.search(norm(s)):
                out.append({"tag": tag, "label": label, "quote": s[:320]})
                break
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def analyze(
    title: str,
    summary: str = "",
    body: str = "",
    source: Optional[Dict[str, Any]] = None,
    require_saudi: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Analyse one article. Returns ``(analysis, None)`` or ``(None, exclusion_reason)``.

    Exclusion reasons: ``no_fraud_signal`` | ``non_financial_context`` | ``not_saudi``.
    """
    source = source or {}
    title = (title or "").strip()
    summary = (summary or "").strip()
    body = (body or "").strip()
    full_raw = " \n".join(x for x in (title, summary, body) if x)
    full_norm = norm(full_raw)
    title_norm = norm(title)

    # 1) explicit fraud signal -------------------------------------------------
    #    A STRONG term must appear in the headline/summary (the reported subject), not merely
    #    somewhere in the page body (menus, related links, passing mentions).
    head_norm = norm(f"{title} {summary}")
    signals = _uniq(_hits(STRONG_FRAUD, full_norm))
    head_signals = _uniq(_hits(STRONG_FRAUD, head_norm))
    head_strong = [s for s in head_signals if s not in WEAK_FRAUD_LABELS]
    head_weak = [s for s in head_signals if s in WEAK_FRAUD_LABELS]
    if not head_strong and len(head_weak) < 2:
        return None, "no_fraud_signal"

    # 2) financial / cyber context --------------------------------------------
    ctx = _uniq(_hits(CONTEXT, full_norm))
    negative = _hits(NEGATIVE_CONTEXT, full_norm)
    title_strong = [s for s in _hits(STRONG_FRAUD, title_norm) if s not in WEAK_FRAUD_LABELS]
    if not ctx or (negative and not title_strong) or (negative and len(signals) == 1 and len(ctx) < 2):
        return None, "non_financial_context"

    # 3) Saudi relevance -------------------------------------------------------
    domain = (source.get("domain") or "").lower()
    publisher_saudi = bool(source.get("saudi_publisher")) or domain.endswith(".sa")

    def _saudi_hits(text: str) -> List[str]:
        # bare "المملكة" is also Bahrain/Jordan/Morocco: only trusted from a Saudi publisher
        return [t for t in _uniq(_hits(SAUDI_TERMS, text)) if t != "المملكه" or publisher_saudi]

    head_saudi = _saudi_hits(head_norm)
    body_saudi = _saudi_hits(full_norm)
    foreign_head = _uniq(_hits(FOREIGN_TERMS, head_norm))
    domestic = _uniq(_hits(DOMESTIC_MARKERS, head_norm)) if publisher_saudi else []
    saudi_terms: List[str] = []
    if head_saudi:
        is_saudi, saudi_basis, saudi_terms = True, "explicit_text", head_saudi
    elif source.get("saudi_scope") and not foreign_head:
        is_saudi, saudi_basis = True, "saudi_domestic_feed"
    elif domestic and not foreign_head:
        is_saudi, saudi_basis, saudi_terms = True, "saudi_outlet_domestic_marker", domestic
    elif publisher_saudi and not foreign_head and len(body_saudi) >= 2:
        # Saudi outlet, nothing foreign in the headline, and the article body itself names Saudi
        # places/entities more than once
        is_saudi, saudi_basis, saudi_terms = True, "saudi_outlet_body_terms", body_saudi
    else:
        is_saudi, saudi_basis = False, "none"
    if require_saudi and not is_saudi:
        return None, "not_saudi"

    # 4) classification --------------------------------------------------------
    scores, cat_hits = _score_categories(full_norm, title_norm)
    category = _pick_category(scores, full_norm)
    secondary = [c for c in CATEGORY_ORDER if c != category and scores[c] >= 3]
    secondary.sort(key=lambda c: -scores[c])
    event_type = _event_type(full_norm)

    # 5) extraction (facts = what the source says) ----------------------------
    # Scraped pages often carry menus / related-link sidebars. Facts (numbers, channels, agencies,
    # methods) are therefore taken ONLY from the headline, the summary and body sentences that
    # themselves carry a fraud signal - never from the whole page.
    rel_raw = relevant_text(title, summary, body)
    rel_norm = norm(rel_raw)
    content_type, content_reasons = classify_content_type(title, summary, rel_raw)
    method_tags = extract_method_tags(rel_raw)
    channels = _group_hits(CHANNELS, rel_norm)
    targets = _group_hits(TARGETS, rel_norm)
    agencies = _group_hits(AGENCIES, rel_norm)
    numbers = _extract_numbers(rel_raw)

    fact_sentences: List[str] = []
    method_sentences: List[str] = []
    for s in _split_sentences(full_raw):
        sn = norm(s)
        has_signal = any(_m(mt, sn) for _, mt in STRONG_FRAUD)
        has_method = any(_m(mt, sn) for _, mt in METHOD_VERBS)
        if has_signal:
            fact_sentences.append(s)
        if has_method and (has_signal or any(_m(mt, sn) for _, mt in CONTEXT)):
            method_sentences.append(s)
    fact_sentences = _uniq(fact_sentences)[:3]
    method_sentences = _uniq([s for s in method_sentences])[:3]

    base_text = summary or (body.split("\n")[0] if body else "")
    incident_summary = ""
    if fact_sentences:
        incident_summary = " ".join(fact_sentences[:2])
    elif base_text:
        incident_summary = base_text
    else:
        incident_summary = title
    incident_summary = incident_summary[:420]

    bank = _bank_relevance(full_norm, category, is_saudi)

    # 6) confidence (formulas documented) -------------------------------------
    # evidence_confidence: reliability of the reported facts = source tier
    #                      (+0.03 per concrete fact: amounts/counts/agency, max +0.08).
    # analysis_confidence: classifier certainty = 0.35 + 0.06*category_score
    #                      (+0.05 if body text was analysed), capped at 0.92.
    ev = _evidence_base(domain, bool(source.get("official")))
    concrete = (1 if numbers["amounts"] else 0) + (1 if numbers["counts"] else 0) + (1 if agencies else 0)
    evidence_conf = round(min(0.95, ev + min(0.08, 0.03 * concrete)), 2)
    top = scores[category]
    analysis_conf = round(min(0.92, 0.35 + 0.06 * top + (0.05 if body else 0.0)), 2)
    confidence = round(min(evidence_conf, analysis_conf), 2)

    # 7) assemble ---------------------------------------------------------------
    tmpl = RULE_TEMPLATES[category]
    rule = {
        "rule_id": f"SFI-{tmpl['code']}-001",
        "status": "proposed_untested",
        "is_inference": True,
        "name": tmpl["name_ar"],
        "hypothesis": tmpl["hypothesis_ar"],
        "logic": tmpl["logic"],
        "required_data": tmpl["required_data"],
        "validation": ["Backtest على 90 يوماً من المعاملات مع تسميات احتيال مؤكدة من إدارة الاحتيال",
                       "قياس Precision عند حجم تنبيهات مقبول + Recall مقابل الحالات المؤكدة",
                       "تشغيل Shadow mode أسبوعين قبل التفعيل ومعايرة العتبات"],
        "derived_from": {"category": category, "signals_in_article": signals[:6], "channels": channels},
    }

    verified_facts = {
        "origin": "stated_in_source",
        "incident_summary": incident_summary,
        "fact_sentences": fact_sentences,
        "reported_method_sentences": method_sentences,
        "reported_channels": channels,
        "reported_targets": targets,
        "agencies_mentioned": agencies,
        "numbers": numbers,
        "fraud_terms_found": signals[:8],
        "method_tags": method_tags,
        "content_type": content_type,
        "content_type_reasons": content_reasons,
        "banks_mentioned": bank["named_entities"],
        "text_basis": "title+summary+article_body" if body else "title+summary_only",
    }

    method_inferred = (
        f"تصنيف النظام: «{CATEGORY_AR[category]}». " +
        ("القنوات المذكورة في المصدر: " + "، ".join(channels) + ". " if channels else "لم يحدد المصدر قناة التنفيذ. ") +
        ("طريقة التنفيذ كما وردت في المصدر مذكورة في reported_method_sentences." if method_sentences
         else "لم يصف المصدر أسلوب التنفيذ بوضوح.")
    )
    analytical_assessment = {
        "origin": "system_inference",
        "fraud_category": category,
        "fraud_category_ar": CATEGORY_AR[category],
        "secondary_categories": [{"key": c, "ar": CATEGORY_AR[c], "score": scores[c]} for c in secondary[:3]],
        "category_scores": {c: scores[c] for c in CATEGORY_ORDER if scores[c] > 0},
        "category_terms_matched": cat_hits[category][:10],
        "modus_operandi": method_inferred,
        "bank_relevance": bank,
        "saudi_relevance": {"is_saudi": is_saudi, "basis": saudi_basis, "terms": saudi_terms[:6]},
        "confidence_basis": {
            "evidence_confidence": "موثوقية المصدر (رسمي/صحيفة سعودية/ناشر آخر) + حقائق محددة مذكورة",
            "analysis_confidence": "درجة وضوح التصنيف حسب الكلمات المطابقة" + (" + تحليل نص المقال" if body else ""),
        },
        "limitations": ("التحليل مبني على العنوان والملخص فقط" if not body else "التحليل مبني على نص المقال المسحوب آليًا"),
    }
    indicators = {
        "mentioned_in_source": _uniq(signals + channels)[:10],
        "suggested_by_system": INDICATORS_TEMPLATE[category],
    }

    is_financial = bank["level"] in ("high", "medium") or category in (
        "financial_fraud", "bill_payment_fraud", "investment_fraud", "cyber_financial")

    return {
        "is_saudi": is_saudi,
        "saudi_basis": saudi_basis,
        "fraud_category": category,
        "fraud_category_ar": CATEGORY_AR[category],
        "event_type": event_type,
        "channels": channels,
        "target_group": targets,
        "bank_relevance": bank,
        "matched_terms": signals,
        "verified_facts": verified_facts,
        "analytical_assessment": analytical_assessment,
        "detection_indicators": indicators,
        "preventive_recommendations": {"origin": "system_template", "items": RECOMMENDATIONS[category]},
        "detection_rule": rule,
        "required_data": tmpl["required_data"],
        "evidence_confidence": evidence_conf,
        "analysis_confidence": analysis_conf,
        "confidence": confidence,
        "is_financial": is_financial,
        "content_type": content_type,
        "content_type_ar": CONTENT_TYPE_AR[content_type],
        "content_type_reasons": content_reasons,
        "method_tags": method_tags,
        "used_body": bool(body),
        "language": "ar" if arabic_ratio(full_raw) > 0.4 else "en",
    }, None


__all__ = ["analyze", "has_fraud_signal", "norm", "CATEGORY_AR", "CATEGORY_ORDER", "arabic_ratio",
           "CONTENT_TYPES", "CONTENT_TYPE_AR", "VISIBLE_CONTENT_TYPES", "classify_content_type", "relevant_text",
           "extract_method_tags", "METHOD_LEXICON"]
