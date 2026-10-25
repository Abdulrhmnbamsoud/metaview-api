# backend/app/sources.py
from typing import Dict, List

# Dedicated Global Fraud & Cybercrime Feeds
GLOBAL_FRAUD_FEEDS = [
    # US Financial Fraud, Bank Scams & Phishing
    "https://news.google.com/rss/search?q=banking+fraud+OR+phishing+scam+when:30d&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=investment+fraud+OR+identity+theft+when:30d&hl=en-US&gl=US&ceid=US:en",
    # UK / Action Fraud & British Bank Scams
    "https://news.google.com/rss/search?q=financial+fraud+arrest+OR+scam+when:30d&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=bank+scam+OR+phishing+when:30d&hl=en-GB&gl=GB&ceid=GB:en",
    # Canada Fraud
    "https://news.google.com/rss/search?q=bank+fraud+OR+scam+when:30d&hl=en-CA&gl=CA&ceid=CA:en",
    # Australia Fraud
    "https://news.google.com/rss/search?q=banking+scam+OR+phishing+when:30d&hl=en-AU&gl=AU&ceid=AU:en",
    # UAE & Regional Arab Fraud
    "https://news.google.com/rss/search?q=احتيال+مالي+OR+نصب+إلكتروني+when:30d&hl=ar&gl=AE&ceid=AE:ar",
    # Egypt Fraud
    "https://news.google.com/rss/search?q=نصب+واحتيال+OR+جرائم+إلكترونية+when:30d&hl=ar&gl=EG&ceid=EG:ar",
    # Leading Cybersecurity & Threat Intelligence Feeds
    "https://www.bleepingcomputer.com/feed/",
    "https://krebsonsecurity.com/feed/",
    "https://feeds.feedburner.com/TheHackersNews",
    "https://securelist.com/feed/",
]

# General news feeds preserved for general media monitoring
GENERAL_WORLD_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://rss.cnn.com/rss/cnn_world.rss",
    "https://www.theguardian.com/world/rss",
    "https://www.france24.com/en/rss",
    "https://rss.dw.com/rdf/rss-en-world",
    "https://www.lemonde.fr/rss/une.xml",
    "https://www.axios.com/feeds/feed.rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "http://feeds.nbcnews.com/feeds/worldnews",
    "https://www.cbsnews.com/latest/rss/world",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://thehill.com/feed/",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://feeds.reuters.com/reuters/worldNews",
]

DEFAULT_SOURCES = GLOBAL_FRAUD_FEEDS + GENERAL_WORLD_FEEDS

# Source origin country mapping (Publisher origin / Region)
SOURCE_COUNTRY_MAP: Dict[str, str] = {
    # Global Fraud Google News Feeds
    "gnews_us_fraud": "US",
    "gnews_us_invest": "US",
    "gnews_uk_fraud": "GB",
    "gnews_uk_scam": "GB",
    "gnews_ca_fraud": "CA",
    "gnews_au_fraud": "AU",
    "gnews_ae_fraud": "AE",
    "gnews_eg_fraud": "EG",
    # Specialized Threat Publishers
    "bleeping_computer": "US",
    "krebs_security": "US",
    "the_hackers_news": "US",
    "securelist": "RU",
    # General Publishers
    "bbc": "GB",
    "cnn": "US",
    "the_guardian": "GB",
    "france24": "FR",
    "dw": "DE",
    "le_monde": "FR",
    "axios": "US",
    "wsj": "US",
    "nyt": "US",
    "nbc": "US",
    "cbs": "US",
    "aljazeera": "QA",
    "the_hill": "US",
    "cnbc": "US",
    "reuters": "GB",
    "bloomberg": "US",
    "financial_times": "GB",
    "spa": "SA",
}

# Source display labels
SOURCE_DISPLAY_NAMES: Dict[str, str] = {
    "gnews_us_fraud": "Google News (US Fraud & Phishing)",
    "gnews_us_invest": "Google News (US Investment Scams)",
    "gnews_uk_fraud": "Google News (UK Financial Fraud)",
    "gnews_uk_scam": "Google News (UK Bank Scams)",
    "gnews_ca_fraud": "Google News (Canada Fraud)",
    "gnews_au_fraud": "Google News (Australia Bank Scams)",
    "gnews_ae_fraud": "أخبار الاحتيال الإلكتروني (الإمارات)",
    "gnews_eg_fraud": "أخبار الجرائم الإلكترونية (مصر)",
    "bleeping_computer": "BleepingComputer",
    "krebs_security": "Krebs on Security",
    "the_hackers_news": "The Hacker News",
    "securelist": "Kaspersky Securelist",
    "bbc": "BBC News",
    "cnn": "CNN International",
    "the_guardian": "The Guardian",
    "france24": "France 24",
    "dw": "Deutsche Welle",
    "le_monde": "Le Monde",
    "axios": "Axios",
    "wsj": "The Wall Street Journal",
    "nyt": "The New York Times",
    "nbc": "NBC News",
    "cbs": "CBS News",
    "aljazeera": "Al Jazeera",
    "the_hill": "The Hill",
    "cnbc": "CNBC",
    "reuters": "Reuters",
    "spa": "SPA (Saudi Press Agency)",
}
