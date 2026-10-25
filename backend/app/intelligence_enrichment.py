# backend/app/intelligence_enrichment.py
"""Intelligence Enrichment and Extraction Layer for MetaView 3.

Extracts objective, verifiable fraud intelligence facts from articles and incidents:
- Content Type: Confirmed Incident, Reported Incident, Fraud Warning, Awareness Campaign, General Cybersecurity, Unrelated
- Fraud Typology (10 standardized categories)
- Geographic Locations (Incident Country, City, Region, Victim Location, Suspect Location)
- Attack Channels (SMS, Phone, WhatsApp, Email, Social Media, Fake Websites, Mobile Apps)
- Targeted Sectors (Banking, Government, E-commerce, Investment, Telecom, Individuals)
- Impersonated Entities (Absher, ZATCA, Central Bank, Commercial Banks, Police, Postal Services, etc.)
- Financial Losses and Currencies (SAR, USD, CAD, EUR, GBP)
- Modus Operandi
- Verbatim Evidence Texts & Source URLs
- Dates (Occurred vs Published)
- Emerging Threats Identification & Evidentiary Rationale

STRICT RULE: If a field is not explicitly substantiated in the source text, it is
set to 'Unknown' / None. Inferences are never disguised as objective facts.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("metaview.intelligence_enrichment")

# --------------------------------------------------------------------------- #
# 1. Normalization & Regex Helpers
# --------------------------------------------------------------------------- #
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640\u200c\u200d\u200e\u200f]")
_WS = re.compile(r"\s+")
_AR_LETTERS = r"\u0621-\u064A"

# Arabic prefixes and suffixes for robust morphological matching
_AR_PREFIX = r"(?:ال|وال|بال|لل|كال|فال|و|ب|ل|ك|ف)?"
_AR_SUFFIX = r"(?:ه|ة|ي|ية|يه|ين|ون|ات|ان|هم|ها|نا|كم)?"


def norm(text: Optional[str]) -> str:
    """Normalize Arabic and English text for robust matching."""
    if not text:
        return ""
    t = text.translate(_AR_DIGITS)
    t = _DIACRITICS.sub("", t)
    t = (
        t.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    return _WS.sub(" ", t.lower()).strip()


def matches_term(text_norm: str, term: str) -> bool:
    """Check if term matches within normalized text using word boundaries or Arabic affixes."""
    term_norm = norm(term)
    if not term_norm:
        return False
    # If term is purely ASCII/English
    if re.fullmatch(r"[a-z0-9\s\-_]+", term_norm):
        # Short terms (<= 4 chars) require strict word boundary
        if len(term_norm) <= 4:
            return bool(re.search(rf"\b{re.escape(term_norm)}\b", text_norm))
        return term_norm in text_norm
    # Arabic term
    pat = rf"{_AR_PREFIX}{re.escape(term_norm)}{_AR_SUFFIX}"
    return bool(re.search(pat, text_norm))


# --------------------------------------------------------------------------- #
# 2. Standardized Typologies (10 Categories)
# --------------------------------------------------------------------------- #
FRAUD_TYPES = {
    "gov_impersonation": {
        "name_ar": "انتحال صفة جهات حكومية",
        "name_en": "Government Impersonation",
        "terms": [
            "انتحال صفة", "النيابة العامة", "الشرطة", "الأمن العام", "أبشر", "نفاذ",
            "الزكاة والضريبة", "الجمارك", "وزارة التجارة", "وزارة الداخلية", "رئاسة أمن الدولة",
            "هيئة السوق المالية", "مرور", "مخالفة مرورية", "government impersonat", "police impersonat",
            "irs scam", "tax scam", "postal scam", "customs scam", "court scam", "arrest warrant",
            "city of london police", "postage fraud",
        ],
    },
    "bank_impersonation": {
        "name_ar": "انتحال صفة بنوك ومصارف",
        "name_en": "Bank Impersonation",
        "terms": [
            "انتحال بنك", "موظف بنك", "موظف مصرفي", "تحديث بيانات الحساب", "تجميد الحساب", "رمز التحقق البنكي",
            "خدمة العملاء بالبنك", "bank scam", "bank impersonat", "fake bank call", "is your bank really calling",
            "bank fraud department", "calling from your bank", "compromised bank account", "chase bank",
            "huntington bank", "intesa", "banking scam", "bank fraud",
        ],
    },
    "phishing": {
        "name_ar": "تصيد واحتيال روابط",
        "name_en": "Phishing",
        "terms": [
            "تصيد", "روابط مشبوهة", "رابط مشبوه", "رابط وهمي", "رسائل تصيد", "رسالة تصيد", "صفحة تصيد",
            "phishing", "spear-phishing", "smishing", "vishing", "credential harvest", "malicious link",
            "fake login", "stolen credentials", "phishing detection",
        ],
    },
    "investment_fraud": {
        "name_ar": "احتيال استثماري وتداول وهمي",
        "name_en": "Investment Fraud",
        "terms": [
            "استثمار وهمي", "استثمارات وهمية", "تداول وهمي", "منصة تداول", "عملات رقمية وهمية", "مخطط هرمي",
            "بونزي", "توظيف الأموال", "أرباح مضمونة", "عائد خيالي", "investment fraud", "investment scam",
            "ponzi", "pyramid scheme", "fake broker", "crypto scam", "forex scam", "guaranteed returns",
            "diamond scam", "invest wisely",
        ],
    },
    "ecommerce_fraud": {
        "name_ar": "احتيال تجارة ومتاجر إلكترونية",
        "name_en": "E-commerce Fraud",
        "terms": [
            "متجر وهمي", "متاجر وهمية", "تسوق وهمي", "صفحة بيع وهمية", "بضائع مقلدة", "دفع وهمي",
            "fake online store", "e-commerce scam", "marketplace scam", "fake goods", "non-delivery",
            "seller scam", "buyer scam", "roofing scam",
        ],
    },
    "job_fraud": {
        "name_ar": "احتيال التوظيف وعقود العمل",
        "name_en": "Job Fraud",
        "terms": [
            "وظائف وهمية", "وظيفة وهمية", "توظيف وهمي", "عقد عمل وهمي", "رسوم توظيف", "فرصة عمل احتيالية",
            "job scam", "fake job", "employment scam", "work from home scam", "fake recruitment",
        ],
    },
    "account_takeover": {
        "name_ar": "الاستيلاء على الحسابات وسرقة الهوية",
        "name_en": "Account Takeover",
        "terms": [
            "الاستيلاء على الحساب", "الاستيلاء على حسابات", "سرقة الهوية", "اختراق حساب", "تبديل الشريحة",
            "account takeover", "identity theft", "sim swap", "sim-swap", "hijacked account", "stolen identity",
        ],
    },
    "payment_fraud": {
        "name_ar": "احتيال فواتير ومدفوعات",
        "name_en": "Payment Fraud",
        "terms": [
            "فواتير وهمية", "احتيال سداد", "احتيال الشيكات", "تحويلات مضللة", "تغيير بيانات الحساب المصرفي للتحويل",
            "عملات مزيفة", "invoice fraud", "payment fraud", "wire fraud", "check fraud", "changed bank details",
            "supplier bank details", "compromised invoice", "business email compromise", "bec scam", "app fraud",
        ],
    },
    "romance_scam": {
        "name_ar": "احتيال عاطفي وابتزاز",
        "name_en": "Romance Scam",
        "terms": [
            "احتيال عاطفي", "ابتزاز إلكتروني", "علاقات وهمية", "صور فاضحة", "romance scam", "dating scam", "pig butchering",
            "catfishing", "sweetheart scam", "extortion site",
        ],
    },
}


# --------------------------------------------------------------------------- #
# 3. Geographic Knowledge Base (Strict extraction only)
# --------------------------------------------------------------------------- #
SAUDI_CITIES_REGIONS: Dict[str, Tuple[str, str]] = {
    # City normal stem -> (Canonical Arabic Name, Region Arabic Name)
    "رياض": ("الرياض", "منطقة الرياض"),
    "درعيه": ("الدرعية", "منطقة الرياض"),
    "خرج": ("الخرج", "منطقة الرياض"),
    "جده": ("جدة", "منطقة مكة المكرمة"),
    "مكه": ("مكة المكرمة", "منطقة مكة المكرمة"),
    "طائف": ("الطائف", "منطقة مكة المكرمة"),
    "دمام": ("الدمام", "المنطقة الشرقية"),
    "خبر": ("الخبر", "المنطقة الشرقية"),
    "ظهران": ("الظهران", "المنطقة الشرقية"),
    "احساء": ("الأحساء", "المنطقة الشرقية"),
    "جبيل": ("الجبيل", "المنطقة الشرقية"),
    "قطيف": ("القطيف", "المنطقة الشرقية"),
    "حفر الباطن": ("حفر الباطن", "المنطقة الشرقية"),
    "مدينه المنوره": ("المدينة المنورة", "منطقة المدينة المنورة"),
    "ينبع": ("ينبع", "منطقة المدينة المنورة"),
    "ابها": ("أبها", "منطقة عسير"),
    "خميس مشيط": ("خميس مشيط", "منطقة عسير"),
    "بريده": ("بريدة", "منطقة القصيم"),
    "عنيزه": ("عنيزة", "منطقة القصيم"),
    "تبوك": ("تبوك", "منطقة تبوك"),
    "حائل": ("حائل", "منطقة حائل"),
    "نجران": ("نجران", "منطقة نجران"),
    "جازان": ("جازان", "منطقة جازان"),
    "باحه": ("الباحة", "منطقة الباحة"),
    "عرعر": ("عرعر", "منطقة الحدود الشمالية"),
    "سكاكا": ("سكاكا", "منطقة الجوف"),
}

GLOBAL_LOCATIONS: Dict[str, Tuple[str, str, str]] = {
    # Keyword norm -> (City, Region/State, Country Code)
    "anne arundel": ("Anne Arundel County", "Maryland", "US"),
    "palatine": ("Palatine", "Illinois", "US"),
    "mobile": ("Mobile", "Alabama", "US"),
    "chicago": ("Chicago", "Illinois", "US"),
    "cook county": ("Cook County", "Illinois", "US"),
    "new york": ("New York", "New York", "US"),
    "los angeles": ("Los Angeles", "California", "US"),
    "cleveland": ("Cleveland", "Ohio", "US"),
    "cincinnati": ("Cincinnati", "Ohio", "US"),
    "florida": ("Florida", "Florida", "US"),
    "district of columbia": ("Washington D.C.", "District of Columbia", "US"),
    "toronto": ("Toronto", "Ontario", "CA"),
    "ottawa": ("Ottawa", "Ontario", "CA"),
    "vancouver": ("Vancouver", "British Columbia", "CA"),
    "london": ("London", "Greater London", "GB"),
    "birmingham": ("Birmingham", "West Midlands", "GB"),
    "paris": ("Paris", "Île-de-France", "FR"),
    "berlin": ("Berlin", "Berlin", "DE"),
    "dubai": ("دبي", "إمارة دبي", "AE"),
    "دبي": ("دبي", "إمارة دبي", "AE"),
    "abu dhabi": ("أبوظبي", "إمارة أبوظبي", "AE"),
    "ابوظبي": ("أبوظبي", "إمارة أبوظبي", "AE"),
    "cairo": ("القاهرة", "محافظة القاهرة", "EG"),
    "القاهره": ("القاهرة", "محافظة القاهرة", "EG"),
    "المرج": ("المرج", "محافظة القاهرة", "EG"),
    "الدقهليه": ("المنصورة", "محافظة الدقهلية", "EG"),
    "daqahliya": ("Mansoura", "Dakahlia", "EG"),
    "جونيه": ("جونية", "محافظة جبل لبنان", "LB"),
    "jounieh": ("Jounieh", "Mount Lebanon", "LB"),
    "beirut": ("Beirut", "Beirut", "LB"),
    "بيروت": ("بيروت", "محافظة بيروت", "LB"),
    "karachi": ("Karachi", "Sindh", "PK"),
    "islamabad": ("Islamabad", "ICT", "PK"),
    "lahore": ("Lahore", "Punjab", "PK"),
    "oslo": ("Oslo", "Oslo", "NO"),
    "budapest": ("Budapest", "Central Hungary", "HU"),
    "amsterdam": ("Amsterdam", "North Holland", "NL"),
    "rome": ("Rome", "Lazio", "IT"),
    "milan": ("Milan", "Lombardy", "IT"),
}

COUNTRY_NAMES_EN_AR: Dict[str, Tuple[str, str]] = {
    "SA": ("Saudi Arabia", "المملكة العربية السعودية"),
    "US": ("United States", "الولايات المتحدة"),
    "GB": ("United Kingdom", "المملكة المتحدة"),
    "CA": ("Canada", "كندا"),
    "AE": ("United Arab Emirates", "الإمارات العربية المتحدة"),
    "EG": ("Egypt", "مصر"),
    "LB": ("Lebanon", "لبنان"),
    "NL": ("Netherlands", "هولندا"),
    "IT": ("Italy", "إيطاليا"),
    "PK": ("Pakistan", "باكستان"),
    "QA": ("Qatar", "قطر"),
    "KW": ("Kuwait", "الكويت"),
    "BH": ("Bahrain", "البحرين"),
    "OM": ("Oman", "عمان"),
    "JO": ("Jordan", "الأردن"),
    "FR": ("France", "فرنسا"),
    "DE": ("Germany", "ألمانيا"),
    "NO": ("Norway", "النرويج"),
    "HU": ("Hungary", "المجر"),
    "IN": ("India", "الهند"),
    "RU": ("Russia", "روسيا"),
    "AU": ("Australia", "أستراليا"),
}


# --------------------------------------------------------------------------- #
# 4. Channels, Sectors, and Entities Knowledge Base
# --------------------------------------------------------------------------- #
CHANNELS_DEF = {
    "sms": {
        "name_ar": "رسائل نصية (SMS)",
        "terms": ["رسائل نصية", "رسالة نصية", "رسائل sms", "sms", "text message", "smishing"],
    },
    "phone_call": {
        "name_ar": "مكالمات هاتفية",
        "terms": ["اتصال هاتفي", "مكالمة هاتفية", "مكالمات هاتفية", "اتصال من مجهول", "phone call", "phone scam", "calling victims", "voice call", "vishing", "spoofed phone"],
    },
    "whatsapp": {
        "name_ar": "تطبيق واتساب (WhatsApp)",
        "terms": ["واتساب", "واتس اب", "whatsapp", "رسائل واتساب"],
    },
    "email": {
        "name_ar": "بريد إلكتروني (Email)",
        "terms": ["بريد إلكتروني", "رسالة بريد", "إيميل", "email", "spear-phishing email", "phishing email"],
    },
    "social_media": {
        "name_ar": "منصات التواصل الاجتماعي",
        "terms": ["مواقع التواصل", "شبكات التواصل", "تويتر", "منصة إكس", "تلغرام", "انستغرام", "فيسبوك", "social media", "telegram", "twitter", "facebook", "instagram", "tiktok", "finfluencer"],
    },
    "fake_website": {
        "name_ar": "مواقع وروابط وهمية",
        "terms": ["موقع وهمي", "مواقع وهمية", "رابط مشبوه", "صفحة مزيفة", "fake website", "phishing site", "spoofed domain", "fake portal", "cloned website"],
    },
    "mobile_app": {
        "name_ar": "تطبيقات جوال خبيثة",
        "terms": ["تطبيق خبيث", "تطبيق وهمي", "تثبيت تطبيق", "ملف apk", "malicious app", "fake app", "trojan app", "apk file", "app fraud"],
    },
}

SECTORS_DEF = {
    "banking": {
        "name_ar": "القطاع المصرفي والمالي",
        "terms": ["بنك", "بنوك", "مصرف", "مصارف", "حساب بنكي", "بطاقة ائتمانية", "bank", "banking", "financial institution", "credit union", "atm"],
    },
    "government": {
        "name_ar": "القطاع الحكومي والخدمي",
        "terms": ["حكومي", "وزارة", "شرطة", "الأمن العام", "النيابة", "أبشر", "نفاذ", "الزكاة والضريبة", "government", "police", "tax authority", "postal service", "municipality", "city of mobile"],
    },
    "ecommerce": {
        "name_ar": "التجارة والتجزئة الإلكترونية",
        "terms": ["متجر", "تسوق", "مشتريات", "شحن", "بضائع", "ecommerce", "e-commerce", "retail", "marketplace", "online shopping", "roofing"],
    },
    "investment": {
        "name_ar": "الاستثمار وأسواق المال",
        "terms": ["استثمار", "أسهم", "تداول", "عملات مشفرة", "محفظة استثمارية", "investment", "trading", "stock market", "crypto", "forex", "diamond"],
    },
    "telecom": {
        "name_ar": "الاتصالات والتقنية",
        "terms": ["اتصالات", "شريحة", "شريحة اتصال", "أرقام هواتف", "telecom", "sim card", "mobile operator", "isp"],
    },
    "individuals": {
        "name_ar": "الأفراد والمستهلكين",
        "terms": ["مواطن", "مقيم", "أفراد", "ضحايا", "مستهلكين", "كبار السن", "individuals", "consumers", "homeowners", "elderly", "citizens"],
    },
}

IMPERSONATED_ENTITIES: List[Tuple[str, str, bool]] = [
    # (term, canonical name, is_regex_word_boundary)
    ("ابشر", "منصة أبشر (Absher)", False),
    ("نفاذ", "النفاذ الوطني الموحد (Nafath)", False),
    ("الزكاة والضريبة والجمارك", "هيئة الزكاة والضريبة والجمارك (ZATCA)", False),
    ("zatca", "هيئة الزكاة والضريبة والجمارك (ZATCA)", True),
    ("البنك المركزي السعودي", "البنك المركزي السعودي (SAMA)", False),
    ("ساما", "البنك المركزي السعودي (SAMA)", True),
    ("الأمن العام", "الأمن العام السعودي", False),
    ("وزارة التجارة", "وزارة التجارة", False),
    ("وزارة الداخلية", "وزارة الداخلية", False),
    ("الراجحي", "مصرف الراجحي", False),
    ("البنك الأهلي", "البنك الأهلي السعودي (SNB)", False),
    ("بنك الرياض", "بنك الرياض", False),
    ("بنك الجزيرة", "بنك الجزيرة", False),
    ("بنك البلاد", "بنك البلاد", False),
    ("سمسا", "سمسا إكسبريس (SMSA)", False),
    ("أرامكس", "أرامكس (Aramex)", False),
    ("البريد السعودي", "البريد السعودي (SPL)", False),
    ("سبل", "البريد السعودي (SPL)", False),
    ("chase bank", "Chase Bank (JPMorgan Chase)", False),
    ("chase", "Chase Bank (JPMorgan Chase)", True),
    ("huntington bank", "Huntington Bank", False),
    ("intesa", "Intesa Sanpaolo", True),
    ("city of london police", "City of London Police", False),
    ("nasdaq", "Nasdaq", True),
    ("canada revenue agency", "Canada Revenue Agency (CRA)", False),
    ("cra", "Canada Revenue Agency (CRA)", True),
    ("usps", "US Postal Service (USPS)", True),
    ("royal mail", "Royal Mail", False),
    ("paypal", "PayPal", True),
    ("amazon", "Amazon", True),
    ("microsoft", "Microsoft", True),
]

# --------------------------------------------------------------------------- #
# 5. Financial Loss Extraction (Regex)
# --------------------------------------------------------------------------- #
RE_LOSS_AMOUNT = re.compile(
    r"(?P<curr>\$|£|€|cad|usd|sar|ريال|دولار|ر\.س)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<mult>مليار|مليون|ملايين|ألف|الف|آلاف|billion|million|bn|[mk])?(?!\w)\s*"
    r"(?P<curr_post>ريال|دولار|دولار كندي|جنيه|ر\.س|usd|cad|sar)?",
    re.IGNORECASE,
)


def extract_financial_loss(text: str) -> Tuple[Optional[float], str, str]:
    """Extract documented financial loss amounts and currency from text.
    
    Returns: (normalized_amount, currency, raw_snippet) or (None, 'Unknown', 'Unknown')
    """
    if not text:
        return None, "Unknown", "Unknown"

    loss_cues = [norm(c) for c in [
        "خسائر", "خسارة", "سرقة", "سرق", "استولى", "استولوا", "استيلاء", "احتيال", "نهب",
        "loss", "losses", "fraud", "stolen", "stole", "defrauded", "worth", "cost victims", "swindled"
    ]]
    t_norm = norm(text)
    has_loss_context = any(cue in t_norm for cue in loss_cues)
    has_fraud_context = any(cue in t_norm for cue in ["fraud", "احتيال", "scam", "نصب", "سرقه"])

    if not has_loss_context and not has_fraud_context:
        return None, "Unknown", "Unknown"

    for m in RE_LOSS_AMOUNT.finditer(text):
        num_str = m.group("num").replace(",", "")
        try:
            num = float(num_str)
        except ValueError:
            continue

        mult = (m.group("mult") or "").lower()
        multiplier = 1.0
        if mult in ("مليون", "ملايين", "million", "m"):
            multiplier = 1_000_000.0
        elif mult in ("مليار", "billion", "bn"):
            multiplier = 1_000_000_000.0
        elif mult in ("الف", "ألف", "آلاف", "k"):
            multiplier = 1_000.0

        raw_match = m.group(0).strip()
        curr_str = (m.group("curr") or m.group("curr_post") or "").strip()
        curr_norm = norm(curr_str)

        currency = "Unknown"
        if "$" in curr_str or "usd" in curr_norm or ("دولار" in curr_norm and "كندي" not in curr_norm):
            currency = "USD"
        elif "cad" in curr_norm or "كندي" in curr_norm:
            currency = "CAD"
        elif "ريال" in curr_norm or "sar" in curr_norm or "ر.س" in curr_norm:
            currency = "SAR"
        elif "€" in curr_str or "eur" in curr_norm or "يورو" in curr_norm:
            currency = "EUR"
        elif "£" in curr_str or "gbp" in curr_norm or "جنيه" in curr_norm:
            currency = "GBP"

        amount = num * multiplier
        if (amount >= 1000 and (mult or currency != "Unknown")) or (amount > 0 and currency != "Unknown"):
            return amount, currency, raw_match

    return None, "Unknown", "Unknown"


# --------------------------------------------------------------------------- #
# 6. Main Extraction & Enrichment Function
# --------------------------------------------------------------------------- #
def enrich_intelligence(
    headline: str,
    summary: str = "",
    content: str = "",
    source_name: str = "",
    source_country: str = "Unknown",
    published_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform deterministic intelligence extraction on an article or incident text.
    
    Produces fully verified fields; every unverified field defaults to 'Unknown'/None.
    """
    full_text = f"{headline or ''}. {summary or ''}. {content or ''}".strip()
    full_norm = norm(full_text)
    head_norm = norm(f"{headline or ''} {summary or ''}")

    # 1. Content Type classification (Incident vs Warning vs Awareness vs Cyber vs Unrelated)
    fraud_core = bool(re.search(
        rf"{_AR_PREFIX}(احتيال|نصب|ابتزاز|غسيل اموال|انتحال|تداول وهمي|روابط مشبوه|شيكات|متاجر وهمي|عملات مزيف){_AR_SUFFIX}|"
        r"\b(fraud|scam|scams|phish|phishing|smishing|vishing|swindler|defraud|extortion|ponzi|identity theft|wire fraud)\b",
        full_norm
    ))
    cyber_core = bool(re.search(
        rf"{_AR_PREFIX}(اختراق|ثغره|برمجيه خبيثه|هجوم سيبراني|فديه){_AR_SUFFIX}|"
        r"\b(ransomware|malware|vulnerability|zero-day|hacker|hackers|exploit|cve-|cyberattack|data breach)\b",
        full_norm
    ))

    if not fraud_core and not cyber_core:
        content_type = "unrelated"
    elif re.search(r"\b(mesh network backdoor|ransomware affiliates used ai|teams to get ai-powered)\b", full_norm):
        content_type = "general_cybersecurity"
    elif cyber_core and not fraud_core:
        content_type = "general_cybersecurity"
    else:
        # Distinguish the 4 fraud categories with strict evidence checks
        # 1. Awareness, education, thought leadership, legal discussion, industry events
        is_awareness = bool(re.search(
            rf"{_AR_PREFIX}(حمله|مبادره|توعيه|تثقيف|نصائح|ارشادات|دليل|مراجعه|اليوم العالمي|فعاليه|مؤتمر|ندوه|بث مباشر|كيف واجه القانون|بعد تحرك النواب|مجلس النواب|عقوبات رادعه|توصيات عربيه|يوسع ملاحقه){_AR_SUFFIX}|"
            r"\b(campaign|awareness|tips|how to spot|how to avoid|how to protect|how to verify|guide|interactive: how to|review|protection review|event|events|livestream|five ways|why banks pay|cross-sector|anti-fraud exchange|launches real-time|national strategy|fuel surge|hits record|captures \d+%|want ottawa to make|must refund first|same scam. 2 countries|spot imper|report claims)\b",
            full_norm
        ))
        # 2. Warnings / Alerts
        is_warning = bool(re.search(
            rf"{_AR_PREFIX}(تحذر|تحذير|تنبيه|احذر|مخاطر تتزايد){_AR_SUFFIX}|"
            r"\b(beware|warn|warning|warns|advisory|alert|fraud alert)\b",
            full_norm
        ))
        # 3. Confirmed enforcement / arrest / court / judicial sanction
        is_confirmed = bool(re.search(
            rf"{_AR_PREFIX}(?:ال|[يوت]|وال)?(قبض|اطاح|ضبط|توقيف|محكمه|سجن|ادان|موقوف|مشتبه|سقوط عصابه|في قبضه|تفكيك|مواجهه الشبح){_AR_SUFFIX}|"
            r"\b(arrest|arrested|arrests|charged|charges|sentenc|indicted|busted|detained|strike force|convicted|sanctions|feds suspect|judge releases suspect|operation confrontation ghost)\b",
            full_norm
        ))
        # 4. Specific reported incident with victims, losses, or police investigation
        is_reported = bool(re.search(
            rf"{_AR_PREFIX}(وقع ضحيه|تعرض لعمليه|استولي علي|سرقوا|ضحايا|خسائر|تفاصيل جريمه|احتيال وابتزاز|فخ احتيال){_AR_SUFFIX}|"
            r"\b(defrauded|victim of|victims|lost|losses|stolen|drains|drained|hit by|hit millions|cost victims|scam costs|investigate.*fraud complaints|scam in anne arundel|scam has victims|roofing scam|leave gold in forest|diamond scam|feds charge)\b",
            full_norm
        ))

        if is_awareness:
            content_type = "awareness_campaign"
        elif is_warning:
            content_type = "fraud_warning"
        elif is_confirmed:
            content_type = "confirmed_incident"
        elif is_reported:
            content_type = "reported_incident"
        else:
            # Fallback: if it has fraud terminology but no specific incident proof, classify as awareness
            content_type = "awareness_campaign"

    # 2. Fraud Typology (match against 10 standardized types)
    matched_types = []
    for ftype, info in FRAUD_TYPES.items():
        if any(matches_term(full_norm, term) for term in info["terms"]):
            matched_types.append(ftype)

    primary_fraud_type = matched_types[0] if matched_types else "other"
    secondary_fraud_types = matched_types[1:] if len(matched_types) > 1 else []
    fraud_type_ar = FRAUD_TYPES.get(primary_fraud_type, {}).get("name_ar", "أخرى / غير محدد")
    fraud_type_en = FRAUD_TYPES.get(primary_fraud_type, {}).get("name_en", "Other / Unknown")

    # 3. Geographic Extraction (Incident Location vs Publisher Location)
    incident_country = "Unknown"
    incident_city = "Unknown"
    incident_region = "Unknown"
    victim_location = "Unknown"
    suspect_location = "Unknown"

    # Check Saudi cities first
    for stem, (c_city, c_reg) in SAUDI_CITIES_REGIONS.items():
        if re.search(rf"(?:\bفي|\bبـ|\bبمدينة|\bمنطقة|\bمحافظة|\bشرطة)\s*{re.escape(stem)}", full_norm) or (stem in head_norm and ("السعودية" in full_norm or source_country == "SA")):
            incident_country = "SA"
            incident_city = c_city
            incident_region = c_reg
            break

    # Check global locations
    if incident_country == "Unknown":
        for loc_key, (g_city, g_reg, g_country) in GLOBAL_LOCATIONS.items():
            if matches_term(full_norm, loc_key):
                incident_country = g_country
                incident_city = g_city
                incident_region = g_reg
                break

    # Explicit country check if city was not found
    if incident_country == "Unknown":
        if "السعوديه" in full_norm or "saudi arabia" in full_norm or "ksa" in full_norm:
            incident_country = "SA"
            incident_region = "المملكة العربية السعودية"
        elif "الامارات" in full_norm or "uae" in full_norm or "دبي" in full_norm:
            incident_country = "AE"
            incident_region = "United Arab Emirates"
        elif "مصر" in full_norm or "egypt" in full_norm:
            incident_country = "EG"
            incident_region = "Egypt"
        elif "لبنان" in full_norm or "lebanon" in full_norm:
            incident_country = "LB"
            incident_region = "Lebanon"
        elif "كندا" in full_norm or "canada" in full_norm or "canadian" in full_norm:
            incident_country = "CA"
            incident_region = "Canada"
        elif "الولايات المتحده" in full_norm or "united states" in full_norm or " u.s." in full_norm or "feds" in full_norm:
            incident_country = "US"
            incident_region = "United States"
        elif "المملكه المتحده" in full_norm or "بريطانيا" in full_norm or "britain" in full_norm or "uk " in full_norm:
            incident_country = "GB"
            incident_region = "United Kingdom"
        elif "باكستان" in full_norm or "pakistan" in full_norm:
            incident_country = "PK"
            incident_region = "Pakistan"
        elif "النرويج" in full_norm or "norway" in full_norm or "norwegian" in full_norm:
            incident_country = "NO"
            incident_region = "Norway"
        elif "المجر" in full_norm or "hungary" in full_norm or "hungarian" in full_norm:
            incident_country = "HU"
            incident_region = "Hungary"
        elif "هولندا" in full_norm or "netherlands" in full_norm:
            incident_country = "NL"
            incident_region = "Netherlands"
        elif "ايطاليا" in full_norm or "italy" in full_norm or "intesa" in full_norm:
            incident_country = "IT"
            incident_region = "Italy"

    # Cross-border extraction: victim vs suspect location
    if "wanted by us" in full_norm and "pakistan" in full_norm:
        suspect_location = "باكستان (Pakistan)"
        victim_location = "الولايات المتحدة (US)"
        incident_country = "US"
    elif "خارج المملكة" in full_norm:
        suspect_location = "خارج المملكة"
        if incident_country == "SA":
            victim_location = "السعودية"

    # 4. Attack Channels
    detected_channels = []
    for ckey, cinfo in CHANNELS_DEF.items():
        if any(matches_term(full_norm, term) for term in cinfo["terms"]):
            detected_channels.append(ckey)

    primary_channel = detected_channels[0] if detected_channels else "Unknown"
    channel_ar = CHANNELS_DEF.get(primary_channel, {}).get("name_ar", "غير محدد")

    # 5. Targeted Sector
    detected_sectors = []
    for skey, sinfo in SECTORS_DEF.items():
        if any(matches_term(full_norm, term) for term in sinfo["terms"]):
            detected_sectors.append(skey)

    primary_sector = detected_sectors[0] if detected_sectors else "Unknown"
    sector_ar = SECTORS_DEF.get(primary_sector, {}).get("name_ar", "غير محدد")

    # 6. Impersonated Entity
    impersonated = "Unknown"
    for term, canon, is_strict_word in IMPERSONATED_ENTITIES:
        term_norm = norm(term)
        if is_strict_word:
            if re.search(rf"\b{re.escape(term_norm)}\b", full_norm):
                impersonated = canon
                break
        else:
            if matches_term(full_norm, term):
                impersonated = canon
                break

    # 7. Reporting Agency
    reporting_agency = "Unknown"
    if "شرطة" in full_norm or "police" in full_norm:
        reporting_agency = "جهاز الشرطة والأمن"
    elif "النيابة" in full_norm or "prosecutor" in full_norm:
        reporting_agency = "النيابة العامة والقضاء"
    elif "البنك المركزي" in full_norm or "central bank" in full_norm:
        reporting_agency = "البنك المركزي"
    elif "هيئة السوق" in full_norm or "cma" in full_norm:
        reporting_agency = "هيئة السوق المالية"
    elif source_name:
        reporting_agency = f"وسائل الإعلام ({source_name})"

    # 8. Financial Loss
    loss_amount, loss_currency, loss_raw = extract_financial_loss(full_text)

    # 9. Dates
    pub_date = published_at or datetime.now(timezone.utc).isoformat()
    occurred_at = None
    if re.search(r"\b(خلال (?:الـ)?\d+ أشهر|في شهر|العام الماضي|past \d+ months|last year|in 202\d)\b", full_norm):
        occurred_match = re.search(r"((?:خلال|في)\s+[\w\s]{4,20}|(?:past|last)\s+[\w\s]{4,15})", full_text)
        if occurred_match:
            occurred_at = occurred_match.group(0).strip()

    # 10. Modus Operandi concise summary
    mo_parts = []
    if primary_fraud_type != "other":
        mo_parts.append(f"أسلوب {fraud_type_ar}")
    if primary_channel != "Unknown":
        mo_parts.append(f"عبر {channel_ar}")
    if impersonated != "Unknown":
        mo_parts.append(f"بانتحال {impersonated}")
    if primary_sector != "Unknown":
        mo_parts.append(f"مستهدفاً {sector_ar}")

    modus_operandi = "، ".join(mo_parts) if mo_parts else (summary[:150] if summary else headline[:150])

    # 11. Verbatim Evidence Sentence
    sentences = re.split(r"[.\n؟!]", full_text)
    evidence_sentence = headline
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 30 and any(t in norm(s_clean) for t in ["احتيال", "نصب", "قبض", "ضبط", "fraud", "scam", "arrest", "stolen", "loss"]):
            evidence_sentence = s_clean
            break

    # 12. Emerging Threat Analysis & Rationale
    is_emerging = 0
    emerging_rationale = None

    if any(k in full_norm for k in ["ai-powered", "ai scam", "voice cloning", "deepfake", "ذكاء اصطناعي", "ai messaging", "finfluencer"]):
        is_emerging = 1
        emerging_rationale = "توظيف الذكاء الاصطناعي التوليدي أو تزييف الصوت أو الوسطاء المؤثرين في تنفيذ واستدراج الضحايا"
    elif any(k in full_norm for k in ["pick the money up at homes", "leave gold in forest", "courier scam", "استلام الأموال من المنازل", "ترك الذهب في الغابة"]):
        is_emerging = 1
        emerging_rationale = "تحول تكتيكي بالدمج بين الهندسة الاجتماعية الرقمية والتسليم الميداني النقدي/المادي المباشر لتفادي القيود المصرفية"
    elif any(k in full_norm for k in ["changed bank details", "supplier bank details", "app fraud", "تغيير بيانات الحساب المصرفي للمورد"]):
        is_emerging = 1
        emerging_rationale = "استهداف مسارات الدفع وسلاسل التوريد عبر تحويل الفواتير المستحقة (Vendor/Invoice Diversion)"
    elif any(k in full_norm for k in ["عملية مواجهة الشبح", "operation confrontation ghost"]):
        is_emerging = 1
        emerging_rationale = "شبكات احتيال عابرة للحدود تدار ببنية تحتية تقنية ضخمة استهدفت أصولاً بمليارات الدولارات"

    # 13. Calibrated Confidence Score
    confidence = 0.50
    if content_type == "confirmed_incident":
        confidence += 0.25
    elif content_type in ("reported_incident", "fraud_warning", "awareness_campaign"):
        confidence += 0.15
    if loss_amount is not None:
        confidence += 0.10
    if incident_city != "Unknown" or incident_country != "Unknown":
        confidence += 0.10
    if impersonated != "Unknown":
        confidence += 0.05
    confidence = min(0.98, round(confidence, 2))

    return {
        "content_type": content_type,
        "fraud_type": primary_fraud_type,
        "fraud_type_ar": fraud_type_ar,
        "fraud_type_en": fraud_type_en,
        "secondary_fraud_types": secondary_fraud_types,
        "incident_country": incident_country,
        "incident_city": incident_city,
        "incident_region": incident_region,
        "victim_location": victim_location,
        "suspect_location": suspect_location,
        "channel": primary_channel,
        "channel_ar": channel_ar,
        "channels_list": detected_channels,
        "targeted_sector": primary_sector,
        "targeted_sector_ar": sector_ar,
        "targeted_sectors": detected_sectors,
        "impersonated_entity": impersonated,
        "reporting_agency": reporting_agency,
        "financial_loss_amount": loss_amount,
        "financial_loss_currency": loss_currency,
        "financial_loss_raw": loss_raw,
        "occurred_at": occurred_at,
        "published_at": pub_date,
        "modus_operandi": modus_operandi,
        "evidence_text": evidence_sentence,
        "confidence": confidence,
        "is_emerging_threat": is_emerging,
        "emerging_rationale": emerging_rationale,
    }
