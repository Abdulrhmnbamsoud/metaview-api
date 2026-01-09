# app/utils.py
import re
from typing import Optional, Tuple

CATEGORY_KEYWORDS = {
    "Politics": ["election", "parliament", "president", "minister", "government", "vote", "sanction", "diplomat"],
    "Economy": ["inflation", "gdp", "interest rate", "stocks", "market", "recession", "trade", "tariff", "bank"],
    "Security": ["attack", "terror", "military", "missile", "war", "explosion", "hostage", "security"],
    "Energy": ["oil", "gas", "opec", "barrel", "crude", "pipeline", "lng", "energy"],
    "Tech": ["ai", "artificial intelligence", "chip", "semiconductor", "cyber", "software", "cloud", "startup"],
}

# Entities: يلتقط أسماء إنجليزية (كلمات Capitalized) + كيانات شائعة
COMMON_ENTITIES = ["United States", "China", "Russia", "Ukraine", "Israel", "Gaza", "Iran", "Saudi Arabia", "EU", "NATO"]

def classify_category(text: str) -> Optional[str]:
    t = (text or "").lower()
    best = None
    best_score = 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best_score = score
            best = cat
    return best

def extract_entity(text: str) -> Optional[str]:
    if not text:
        return None

    # 1) common entities
    for ent in COMMON_ENTITIES:
        if ent.lower() in text.lower():
            return ent

    # 2) capitalized phrases (e.g., "Donald Trump", "Apple Inc")
    # يلتقط جملتين أو ثلاث كلمات كبيرة
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    if matches:
        # اختار الأكثر تكرارًا
        cand = max(set(matches), key=matches.count)
        return cand

    return None

def enrich(headline: str, summary: str, content: str) -> Tuple[Optional[str], Optional[str]]:
    blob = " ".join([headline or "", summary or "", content or ""]).strip()
    category = classify_category(blob)
    entity = extract_entity(blob)
    return category, entity
