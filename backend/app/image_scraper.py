# backend/app/image_scraper.py
"""
MetaView 3 — Standalone News Image Scraper Module
Extracts high-fidelity news images from multiple cascading layers:
1. RSS media:content
2. RSS media:thumbnail
3. RSS enclosure (image/*)
4. HTML content <img> tags in RSS summary/content
5. Open Graph <meta property="og:image"> via HTTP fallback
6. Twitter Card <meta name="twitter:image">
7. Article body <img> elements

Equipped with SSRF protection, size & keyword filtering, and relative URL resolution.
"""

import re
import ipaddress
import socket
from urllib.parse import urlparse, urljoin
from typing import Optional, Dict, Any, List
import httpx
import logging
from bs4 import BeautifulSoup

# Blacklisted keywords indicating logos, tracking pixels, badges, avatars
BAD_IMAGE_KEYWORDS = [
    "icon", "logo", "avatar", "spacer", "pixel", "tracking", "badge",
    "advertisement", "ad-", "banner-ad", "button", "spinner", "placeholder",
    "favicon", "sprite", "1x1", "transparent"
]

# User‑Agent header for external HTTP requests (e.g., X profile scraping)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 MetaViewBot/2.0"
}

# Supported image extensions
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")

# Private and internal IP ranges to block (SSRF protection)
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / Cloud metadata
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


from .security_network import validate_url_for_ssrf


def is_safe_url(url: str) -> bool:
    """Verifies that the URL is public, uses HTTP/HTTPS, and avoids SSRF targets."""
    safe, _ = validate_url_for_ssrf(url)
    return safe


def is_candidate_image(url: str) -> bool:
    """Filters out icons, tracking pixels, SVGs, and obvious non-article assets."""
    if not url or not isinstance(url, str):
        return False
    
    clean_url = url.lower().split("?")[0]
    
    # Exclude SVG (usually icons/logos)
    if clean_url.endswith(".svg"):
        return False
        
    for bad in BAD_IMAGE_KEYWORDS:
        if bad in clean_url:
            return False
            
    return True


def extract_image_from_rss_entry(entry: Any, base_url: str = "") -> Optional[str]:
    """
    Extracts an image from feedparser entry via RSS tags in order of priority:
    1. media_content
    2. media_thumbnail
    3. enclosures
    4. <img> tag in summary / content
    """
    if not entry:
        return None

    # 1. media:content
    media_content = getattr(entry, "media_content", None)
    if media_content and isinstance(media_content, list):
        for media in media_content:
            m_url = media.get("url")
            m_type = media.get("type", "")
            m_medium = media.get("medium", "")
            if m_url and (m_medium == "image" or "image" in m_type or any(m_url.lower().endswith(ext) for ext in VALID_EXTENSIONS)):
                if is_candidate_image(m_url):
                    return m_url

    # 2. media:thumbnail
    media_thumbnail = getattr(entry, "media_thumbnail", None)
    if media_thumbnail and isinstance(media_thumbnail, list):
        for thumb in media_thumbnail:
            t_url = thumb.get("url")
            if t_url and is_candidate_image(t_url):
                return t_url

    # 3. enclosures
    enclosures = getattr(entry, "enclosures", None)
    if enclosures and isinstance(enclosures, list):
        for enc in enclosures:
            href = enc.get("href") or enc.get("url")
            enc_type = enc.get("type", "")
            if href and ("image" in enc_type or any(href.lower().endswith(ext) for ext in VALID_EXTENSIONS)):
                if is_candidate_image(href):
                    return href

    # 4. HTML <img> tags in summary or content
    content_candidates = []
    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
    if summary:
        content_candidates.append(summary)
    contents = getattr(entry, "content", None)
    if contents and isinstance(contents, list):
        for c in contents:
            if isinstance(c, dict) and "value" in c:
                content_candidates.append(c["value"])

    for html_text in content_candidates:
        if "<img" in html_text.lower():
            try:
                soup = BeautifulSoup(html_text, "html.parser")
                img = soup.find("img")
                if img and img.get("src"):
                    src = img["src"].strip()
                    resolved = urljoin(base_url, src) if base_url else src
                    if is_candidate_image(resolved):
                        return resolved
            except Exception:
                pass

    return None


async def scrape_article_webpage_image(article_url: str, client: Optional[httpx.AsyncClient] = None) -> Optional[str]:
    """
    Secondary fallback: Fetches article HTML to extract Open Graph or Twitter card image.
    Uses strict timeout (3.5s) and SSRF verification.
    """
    if not is_safe_url(article_url):
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 MetaViewBot/2.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    from .security_network import safe_fetch_url
    try:
        resp = await safe_fetch_url(article_url, timeout_sec=3.5, max_bytes=200000, headers=headers)
        if resp.status_code != 200:
            return None

        # Parse head tags
        soup = BeautifulSoup(resp.text[:200000], "html.parser")  # only read first 200KB for speed

        # 5. Open Graph: <meta property="og:image" content="...">
        og_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
        if og_img and og_img.get("content"):
            url = og_img["content"].strip()
            resolved = urljoin(article_url, url)
            if is_candidate_image(resolved) and is_safe_url(resolved):
                return resolved

        # 6. Twitter Card: <meta name="twitter:image" content="...">
        tw_img = soup.find("meta", attrs={"name": "twitter:image"}) or soup.find("meta", property="twitter:image")
        if tw_img and tw_img.get("content"):
            url = tw_img["content"].strip()
            resolved = urljoin(article_url, url)
            if is_candidate_image(resolved) and is_safe_url(resolved):
                return resolved

        # 7. Article HTML body main image
        article_tag = soup.find("article") or soup.find("main")
        if article_tag:
            img = article_tag.find("img")
            if img and img.get("src"):
                src = img["src"].strip()
                resolved = urljoin(article_url, src)
                if is_candidate_image(resolved) and is_safe_url(resolved):
                    return resolved

    except Exception:
        return None
    finally:
        if own_client:
            await client.aclose()

    return None


async def resolve_article_image(entry: Any, article_url: str, client: Optional[httpx.AsyncClient] = None, allow_web_scrape: bool = False) -> Optional[str]:
    """
    Main entry point for resolving an article image.
    Prioritizes fast RSS metadata extraction, falling back optionally to HTML metadata scrape.
    """
    # Try fast RSS extraction first
    rss_image = extract_image_from_rss_entry(entry, base_url=article_url)
    if rss_image and is_candidate_image(rss_image):
        return rss_image

    # If not found and web scrape allowed, fetch HTML tags
    if allow_web_scrape and article_url:
        return await scrape_article_webpage_image(article_url, client=client)

    return None

# ------------------------------------------------------------
# X (formerly Twitter) profile scraper – fetch recent tweets
# ------------------------------------------------------------

def fetch_x_profile(url: str, max_pages: int = 1) -> List[Dict]:
    """Fetch recent tweets from a public X profile page.

    Parameters
    ----------
    url: str
        Base X profile URL (e.g. ``https://x.com/thehackersnews?s=11``).
    max_pages: int
        Number of pagination steps to follow (default ``1``).

    Returns
    -------
    List[Dict]
        List of tweet dictionaries containing ``id``, ``text``, ``time`` and ``url``.
    """
    logger = logging.getLogger(__name__)
    tweets: List[Dict] = []
    next_url = url

    for page in range(max_pages):
        logger.debug("Fetching X page %s: %s", page + 1, next_url)
        if not is_safe_url(next_url):
            logger.error("Blocked unsafe X URL for SSRF: %s", next_url)
            break
        try:
            resp = httpx.get(next_url, headers=HEADERS, timeout=10.0)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", next_url, e)
            break
        if resp.status_code != 200:
            logger.error("Unexpected status %s for %s", resp.status_code, next_url)
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        for article in soup.select('article[data-testid="tweet"]'):
            tweet_id = article.get("data-tweet-id")
            text_elem = article.select_one('div[lang]')
            tweet_text = text_elem.get_text(separator=" ", strip=True) if text_elem else ""
            time_elem = article.select_one('time')
            tweet_time = time_elem["datetime"] if time_elem and time_elem.has_attr("datetime") else ""
            link_elem = article.select_one('a[href*="/status/"]')
            tweet_url = f"https://x.com{link_elem['href']}" if link_elem else ""
            if tweet_id and tweet_text:
                tweets.append({
                    "id": tweet_id,
                    "text": tweet_text,
                    "time": tweet_time,
                    "url": tweet_url,
                })
        # Find next pagination link (X uses ?s= query param)
        next_link = soup.select_one('a[role="link"][href*="?s="][rel="next"]')
        if not next_link:
            break
        next_url = f"https://x.com{next_link['href']}"
    return tweets

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_url = "https://x.com/thehackersnews?s=11"
    recent = fetch_x_profile(example_url, max_pages=2)
    import json
    print(json.dumps(recent, indent=2, ensure_ascii=False))
