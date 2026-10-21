# backend/app/sources.py
from typing import Dict, List

DEFAULT_SOURCES = [
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

# Source origin country mapping (Publisher origin, NOT event location)
SOURCE_COUNTRY_MAP: Dict[str, str] = {
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
}

# Source display labels
SOURCE_DISPLAY_NAMES: Dict[str, str] = {
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
}
