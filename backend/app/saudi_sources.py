# backend/app/saudi_sources.py
"""Registry of Saudi news / official sources used by the Saudi Fraud Intelligence pipeline.

Every entry below was discovered and tested with a real HTTP request (see the ingestion
report). Nothing here is assumed: sources that turned out to have no machine-readable
feed are kept as ``probe`` entries so their availability and failure reason stay visible
in the UI instead of silently disappearing.

kinds
-----
rss         standard RSS/Atom feed
nextdata    official site that embeds its latest news as JSON in ``__NEXT_DATA__`` (SPA)
gnews       Google News RSS *search* (aggregator) - used to reach official/news sites that
            publish no feed of their own. Links are decoded to the original article URL.
probe       no usable feed/API; we only test reachability and record the reason
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import quote_plus


@dataclass(frozen=True)
class SaudiSource:
    code: str
    name: str
    url: str
    kind: str                       # rss | nextdata | gnews | probe
    lang: str = "ar"
    official: bool = False          # government / regulator
    saudi_scope: bool = False       # feed is Saudi-domestic news (article need not name KSA)
    domain: str = ""                # publisher domain (direct feeds)
    notes: str = ""
    enabled: bool = True
    extra: Dict[str, str] = field(default_factory=dict)


def _gnews(query: str, lang: str = "ar") -> str:
    q = quote_plus(f"{query} when:90d")
    if lang == "ar":
        return f"https://news.google.com/rss/search?q={q}&hl=ar&gl=SA&ceid=SA:ar"
    return f"https://news.google.com/rss/search?q={q}&hl=en&gl=SA&ceid=SA:en"


# ---- direct publisher feeds (verified live) ------------------------------------------
_DIRECT: List[SaudiSource] = [
    SaudiSource("saudigazette", "Saudi Gazette", "https://saudigazette.com.sa/rssFeed/74", "rss", "en",
                domain="saudigazette.com.sa", notes="Saudi Gazette main RSS (rssFeed/74)"),
    SaudiSource("sabq", "صحيفة سبق", "https://sabq.org/api/rss/articles", "rss", "ar",
                domain="sabq.org", notes="سبق - آخر الأخبار"),
    SaudiSource("okaz_1", "صحيفة عكاظ – قسم 1", "https://www.okaz.com.sa/rssFeed/1", "rss", "ar",
                saudi_scope=True, domain="okaz.com.sa"),
    SaudiSource("okaz_2", "صحيفة عكاظ – قسم 2", "https://www.okaz.com.sa/rssFeed/2", "rss", "ar", domain="okaz.com.sa"),
    SaudiSource("okaz_4", "صحيفة عكاظ – اقتصاد", "https://www.okaz.com.sa/rssFeed/4", "rss", "ar", domain="okaz.com.sa"),
    SaudiSource("almadina_local", "صحيفة المدينة – محليات", "https://www.al-madina.com/rssFeed/1", "rss", "ar",
                saudi_scope=True, domain="al-madina.com"),
    SaudiSource("almadina_companies", "صحيفة المدينة – أخبار الشركات", "https://www.al-madina.com/rssFeed/206", "rss", "ar",
                saudi_scope=True, domain="al-madina.com"),
    SaudiSource("almadina_all", "صحيفة المدينة – آخر الأخبار", "https://www.al-madina.com/rss/feed", "rss", "ar",
                domain="al-madina.com"),
    SaudiSource("alwatan_1", "صحيفة الوطن – قسم 1", "https://www.alwatan.com.sa/rssFeed/1", "rss", "ar", domain="alwatan.com.sa"),
    SaudiSource("alwatan_2", "صحيفة الوطن – قسم 2", "https://www.alwatan.com.sa/rssFeed/2", "rss", "ar", domain="alwatan.com.sa"),
    SaudiSource("alwatan_4", "صحيفة الوطن – قسم 4", "https://www.alwatan.com.sa/rssFeed/4", "rss", "ar", domain="alwatan.com.sa"),
    SaudiSource("argaam_news", "أرقام – أخبار السوق السعودي", "https://www.argaam.com/ar/rss/ho-main-news?sectionid=1523",
                "rss", "ar", saudi_scope=True, domain="argaam.com"),
]
_DIRECT += [
    SaudiSource(f"alyaum_{i}", f"صحيفة اليوم – قسم {i}", f"https://www.alyaum.com/rssFeed/{i}", "rss", "ar", domain="alyaum.com")
    for i in (1005, 1006, 1007, 1008, 1009, 1010)
]

# ---- official: SPA publishes its latest news as JSON inside the page ------------------
_OFFICIAL: List[SaudiSource] = [
    SaudiSource("spa", "وكالة الأنباء السعودية (واس)", "https://www.spa.gov.sa/ar", "nextdata", "ar",
                official=True, saudi_scope=True, domain="spa.gov.sa",
                notes="لا يوجد RSS؛ الأخبار الرئيسية مضمنة كـ JSON في الصفحة (__NEXT_DATA__) - تغطية محدودة (آخر 6 أخبار رئيسية)"),
]

# ---- sources with no machine-readable feed: availability is probed & recorded ---------
_PROBES: List[SaudiSource] = [
    SaudiSource("sama", "البنك المركزي السعودي (ساما)", "https://www.sama.gov.sa/ar-sa/News/Pages/default.aspx", "probe",
                official=True, domain="sama.gov.sa", notes="لا يوجد RSS رسمي؛ يتم الوصول لمحتواه عبر استعلامات site: في Google News"),
    SaudiSource("cma", "هيئة السوق المالية", "https://cma.gov.sa/ar/MediaCenter/News/Pages/default.aspx", "probe",
                official=True, domain="cma.gov.sa", notes="الصفحة تُبنى بجافاسكربت ولا يوجد RSS؛ يُغطّى عبر site: في Google News"),
    SaudiSource("nca", "الهيئة الوطنية للأمن السيبراني", "https://nca.gov.sa/ar/news/", "probe",
                official=True, domain="nca.gov.sa", notes="الصفحة تُبنى بجافاسكربت ولا يوجد RSS؛ يُغطّى عبر site: في Google News"),
    SaudiSource("moi", "وزارة الداخلية / الأمن العام", "https://moi.gov.sa/Pages/home?lang=ar&culture=ar-SA&ui-culture=ar-SA",
                "probe", official=True, domain="moi.gov.sa", notes="لا يوجد RSS؛ يُغطّى عبر site: في Google News"),
    SaudiSource("ppo", "النيابة العامة", "https://www.ppo.gov.sa/", "probe", official=True, domain="ppo.gov.sa",
                notes="لا يوجد RSS؛ يُغطّى عبر site: في Google News"),
    SaudiSource("alriyadh", "صحيفة الرياض", "https://www.alriyadh.com/rss", "probe", domain="alriyadh.com",
                notes="خدمة RSS معلنة لكن الموقع لا يتيح اتصال TLS صالحاً من الخادم؛ يُغطّى عبر site: في Google News"),
    SaudiSource("aleqtisadiah", "صحيفة الاقتصادية", "https://www.aleqt.com/rss", "probe", domain="aleqt.com",
                notes="لا يوجد RSS (404)؛ يُغطّى عبر site: في Google News"),
]

# ---- Google News search queries (aggregator) -------------------------------------------
_GNEWS_QUERIES_AR = [
    ("fraud_financial", "احتيال مالي السعودية"),
    ("fraud_electronic", "احتيال إلكتروني المملكة"),
    ("phishing", "تصيد إلكتروني السعودية"),
    ("bank_impersonation", "انتحال صفة بنك السعودية"),
    ("fraud_sms", "رسائل احتيالية روابط مشبوهة السعودية"),
    ("security_arrests", "الأمن العام القبض احتيال"),
    ("fake_store", "متجر وهمي احتيال السعودية"),
    ("fake_investment", "استثمار وهمي هيئة السوق المالية تحذر"),
    ("sama_warning", "البنك المركزي السعودي ساما تحذر احتيال"),
    ("nca_warning", "الهيئة الوطنية للأمن السيبراني تحذر تصيد"),
    ("ppo_fraud", "النيابة العامة السعودية احتيال"),
    ("phone_fraud", "مكالمات احتيالية الهندسة الاجتماعية السعودية"),
    ("account_theft", "اختراق حساب سرقة هوية السعودية"),
    ("bill_fraud", "احتيال فواتير سداد مخالفات وهمية السعودية"),
]
_GNEWS_SITE_QUERIES = [
    ("site_spa", "site:spa.gov.sa احتيال"),
    ("site_sama", "site:sama.gov.sa احتيال"),
    ("site_cma", "site:cma.gov.sa احتيال"),
    ("site_nca", "site:nca.gov.sa تصيد"),
    ("site_moi", "site:moi.gov.sa احتيال"),
    ("site_ppo", "site:ppo.gov.sa احتيال"),
    ("site_alriyadh", "site:alriyadh.com احتيال"),
    ("site_aleqt", "site:aleqt.com احتيال"),
    ("site_okaz", "site:okaz.com.sa احتيال"),
    ("site_sabq", "site:sabq.org احتيال"),
]
_GNEWS_QUERIES_EN = [
    ("en_fraud", "Saudi Arabia fraud scam"),
    ("en_phishing", "Saudi phishing warning"),
    ("en_sama", "Saudi Central Bank fraud warning"),
]

_GNEWS: List[SaudiSource] = (
    [SaudiSource(f"gnews_{c}", f"Google News ← {q}", _gnews(q), "gnews", "ar", extra={"query": q})
     for c, q in _GNEWS_QUERIES_AR + _GNEWS_SITE_QUERIES] +
    [SaudiSource(f"gnews_{c}", f"Google News ← {q}", _gnews(q, "en"), "gnews", "en", extra={"query": q})
     for c, q in _GNEWS_QUERIES_EN]
)

SAUDI_SOURCES: List[SaudiSource] = _OFFICIAL + _DIRECT + _GNEWS + _PROBES
SOURCES_BY_CODE: Dict[str, SaudiSource] = {s.code: s for s in SAUDI_SOURCES}


def get_source(code: str) -> Optional[SaudiSource]:
    return SOURCES_BY_CODE.get(code)
