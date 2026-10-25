# backend/app/incidents.py
"""Incident layer: Article -> Incident -> Fraud Pattern.

* **Article**  - the publisher's original news item (``articles`` + its per-article analysis in
  ``threat_intel``). Never merged, never deleted.
* **Incident** - one real fraud event, possibly covered by many articles (``incidents``).
  Only articles whose content type is ``confirmed_incident`` can form an incident.
* **Fraud pattern** - a fraud *method* that the sources state (``fraud_intel.METHOD_LEXICON``),
  aggregated over one or more incidents. Computed on demand from incidents (no extra state).

Deduplication never relies on headline similarity. Two articles are the same incident only when
they agree on *facts extracted from the text* (number of suspects, nationalities, amounts, seized
items, place, reporting agency) and were published within a few days of each other. Contradicting
facts (different place / different suspect count / different nationalities) veto a merge.
Everything here is deterministic and re-runnable (``rebuild`` is idempotent); incident rows are
derived data, so rebuilding them never touches an article.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import fraud_intel as fi
from .db import get_db

log = logging.getLogger("metaview.incidents")

MAX_GAP_DAYS = 5          # an article joins an incident only if published within this many days of it
MAX_FACT_SENTENCES = 5
MAX_METHOD_QUOTES = 4

_AR = "\u0621-\u064A"
_PREFIX = r"(?:ال|وال|بال|لل|كال|فال|و|ب|ل|ك|ف)?"


def _j(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, ensure_ascii=False)


def _loads(v: Any, default: Any = None) -> Any:
    if v is None or v == "":
        return default
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fact signature (what an article states about the event)
# --------------------------------------------------------------------------- #
# nationality key -> (normalised stem, Arabic label)
_NATIONALITIES: Dict[str, Tuple[str, str]] = {
    "yemeni": ("يمني", "يمنية"), "sudanese": ("سوداني", "سودانية"), "egyptian": ("مصري", "مصرية"),
    "syrian": ("سوري", "سورية"), "pakistani": ("باكستاني", "باكستانية"), "indian": ("هندي", "هندية"),
    "bangladeshi": ("بنغالي", "بنغالية"), "ethiopian": ("اثيوبي", "إثيوبية"), "nigerian": ("نيجيري", "نيجيرية"),
    "filipino": ("فلبيني", "فلبينية"), "jordanian": ("اردني", "أردنية"), "lebanese": ("لبناني", "لبنانية"),
    "iraqi": ("عراقي", "عراقية"), "palestinian": ("فلسطيني", "فلسطينية"), "moroccan": ("مغربي", "مغربية"),
    "tunisian": ("تونسي", "تونسية"), "somali": ("صومالي", "صومالية"), "eritrean": ("اريتري", "إريترية"),
    "indonesian": ("اندونيسي", "إندونيسية"), "nepalese": ("نيبالي", "نيبالية"), "afghan": ("افغاني", "أفغانية"),
}
_NAT_RES = {k: re.compile(r"(?<![" + _AR + r"])" + _PREFIX + stem + r"(?:ه|ين|ون|ات|ا)?(?![" + _AR + r"])")
            for k, (stem, _label) in _NATIONALITIES.items()}

# Saudi places: normalised stem -> display label
_PLACES: Dict[str, str] = {
    "رياض": "الرياض", "جده": "جدة", "مكه": "مكة المكرمة", "المدينه المنوره": "المدينة المنورة", "دمام": "الدمام",
    "ظهران": "الظهران", "ابها": "أبها", "تبوك": "تبوك", "بريده": "بريدة", "نجران": "نجران", "جازان": "جازان",
    "حائل": "حائل", "ينبع": "ينبع", "طائف": "الطائف", "احساء": "الأحساء", "قصيم": "القصيم", "عسير": "عسير",
    "جبيل": "الجبيل", "باحه": "الباحة", "عرعر": "عرعر", "قطيف": "القطيف", "خميس مشيط": "خميس مشيط",
    "حفر الباطن": "حفر الباطن", "الخرج": "الخرج",
}
_PLACE_RES = {k: re.compile(r"(?<![" + _AR + r"])" + (r"" if k.startswith("ال") else _PREFIX) + re.escape(k) + r"(?![" + _AR + r"])")
              for k in _PLACES}

_PERSON_STEMS = ["مقيم", "متهم", "اشخاص", "شخص", "وافد", "مواطن", "محتال", "نصاب", "عنصر", "عناصر", "موقوف",
                 "مشتبه"] + [stem for stem, _ in _NATIONALITIES.values()]
_PERSON_RE = re.compile(r"(?<![\d.,])(\d{1,3})\)?\s+(?:من\s+)?(?:" + "|".join(_PERSON_STEMS) + r")[" + _AR + r"]*")
_AMOUNT_RE = re.compile(r"(\d[\d,\.]*)\)?\s*(مليون|مليار|الف)?\s*(?:ريال|ر\.س|دولار)")
_POLICE_RE = re.compile(r"شرطه (?:منطقه |محافظه )?(?:ال)?(" + "|".join(re.escape(p) for p in _PLACES) + r")")

_ITEMS: Dict[str, List[str]] = {
    "gold": ["سبائك", "ذهب"], "phones": ["هاتف", "هواتف", "جوال", "جوالات"], "ipad": ["ايباد", "ipad"],
    "bank_cards": ["بطاقه صراف", "بطاقات صراف", "بطاقه بنكيه", "بطاقات بنكيه", "بطاقات مصرفيه", "بطاقه مصرفيه",
                   "بطاقه ائتمانيه", "بطاقات ائتمانيه"],
    "vehicles": ["مركبه", "مركبات", "سياره", "سيارات"], "cash_seized": ["مبلغ نقدي", "مبالغ نقديه"],
}
_ITEM_LABELS = {"gold": "ذهب/سبائك", "phones": "هواتف", "ipad": "آيباد", "bank_cards": "بطاقات بنكية/صراف",
                "vehicles": "مركبات", "cash_seized": "مبالغ نقدية"}


_STOP_TERMS = {fi.norm(w) for w in (
    "احتيال", "احتيالي", "احتيالية", "الاحتيال", "شبكة", "شرطة", "مقيمين", "المالي", "مالي", "المالية", "مالية",
    "بتهمة", "القبض", "الرياض", "السعودية", "المملكة", "الجهات", "الأمنية", "وزارة", "الداخلية", "أعلنت",
    "أعلن", "وكالة", "الأنباء", "التحقيق", "المواطنين", "المقيمين", "عبر", "خلال", "حيث", "التي", "الذي",
    "هذه", "هذا", "كما", "أيضاً", "وذلك", "بعد", "عن", "على", "إلى", "ضمن", "بين", "جرائم", "جريمة")}
_TERM_RE = re.compile(r"[\u0621-\u064A]{4,}")


@dataclass
class Signature:
    counts: Set[int] = field(default_factory=set)       # number of suspects/persons
    nats: Set[str] = field(default_factory=set)
    amounts: Set[int] = field(default_factory=set)
    locs: Set[str] = field(default_factory=set)
    agencies: Set[str] = field(default_factory=set)
    items: Set[str] = field(default_factory=set)
    terms: Set[str] = field(default_factory=set)        # content words: SUPPORTING signal only (never alone)

    def union(self, other: "Signature") -> "Signature":
        return Signature(self.counts | other.counts, self.nats | other.nats, self.amounts | other.amounts,
                         self.locs | other.locs, self.agencies | other.agencies, self.items | other.items,
                         self.terms | other.terms)

    def to_json(self) -> Dict[str, Any]:
        return {"persons_counts": sorted(self.counts), "nationalities": sorted(self.nats),
                "amounts": sorted(self.amounts), "locations": sorted(self.locs),
                "agencies": sorted(self.agencies), "items": sorted(self.items)}


def _amount_value(num: str, unit: Optional[str]) -> Optional[int]:
    try:
        v = float(num.replace(",", "").rstrip("."))
    except ValueError:
        return None
    v *= {"الف": 1_000, "مليون": 1_000_000, "مليار": 1_000_000_000}.get(unit or "", 1)
    return int(round(v)) if v >= 100 else None


def extract_signature(rel_raw: str) -> Signature:
    """Facts an article states about the event, taken from its relevant text only."""
    t = fi.norm(rel_raw)
    sig = Signature()
    for m in _PERSON_RE.finditer(t):
        n = int(m.group(1))
        if 1 <= n <= 500:
            sig.counts.add(n)
    for key, rx in _NAT_RES.items():
        if rx.search(t):
            sig.nats.add(key)
    for m in _AMOUNT_RE.finditer(t):
        v = _amount_value(m.group(1), m.group(2))
        if v:
            sig.amounts.add(v)
    for key, rx in _PLACE_RES.items():
        if rx.search(t):
            sig.locs.add(_PLACES[key])
    sig.agencies.update(fi._group_hits(fi.AGENCIES, t))
    for m in _POLICE_RE.finditer(t):
        sig.agencies.add("شرطة " + _PLACES[m.group(1)])
    for key, words in _ITEMS.items():
        if any(w in t for w in words):
            sig.items.add(key)
    sig.terms = {w for w in _TERM_RE.findall(t) if w not in _STOP_TERMS}
    return sig


SIM_THRESHOLD = 0.5   # overlap coefficient of content words
SIM_MIN_SHARED = 8    # ...and at least this many shared words


def text_similarity(a: Signature, b: Signature) -> Tuple[float, int]:
    """Overlap of the *content words of the relevant text* (never the headline alone)."""
    if len(a.terms) < 6 or len(b.terms) < 6:
        return 0.0, 0
    shared = len(a.terms & b.terms)
    return shared / min(len(a.terms), len(b.terms)), shared


def compare(a: Signature, b: Signature) -> Tuple[bool, int, Dict[str, Any]]:
    """Do two fact signatures describe the same event? (date gate is applied by the caller)

    Contradicting facts veto a merge. Otherwise they merge automatically when they share two *strong*
    facts (suspect count, amount, >=2 seized items) or one strong fact plus two context facts
    (nationalities, place, agency, text similarity). Weaker agreement (no conflict, score >= 2) is
    reported as ``possible``: it is NOT merged, it is queued for human review.
    """
    conflicts = []
    if a.locs and b.locs and not (a.locs & b.locs):
        conflicts.append("location")
    if a.nats and b.nats and not (a.nats & b.nats):
        conflicts.append("nationality")
    if a.counts and b.counts and not (a.counts & b.counts):
        conflicts.append("suspect_count")
    sim, shared = text_similarity(a, b)
    strong = {"suspect_count": bool(a.counts & b.counts), "amount": bool(a.amounts & b.amounts),
              "seized_items": len(a.items & b.items) >= 2}
    ctx = {"nationality": bool(a.nats & b.nats), "location": bool(a.locs & b.locs),
           "agency": bool(a.agencies & b.agencies),
           "text_similarity": sim >= SIM_THRESHOLD and shared >= SIM_MIN_SHARED}
    n_strong, n_ctx = sum(strong.values()), sum(ctx.values())
    score = 2 * n_strong + n_ctx
    ok = not conflicts and (n_strong >= 2 or (n_strong >= 1 and n_ctx >= 2))
    detail = {"strong": [k for k, v in strong.items() if v], "context": [k for k, v in ctx.items() if v],
              "conflicts": conflicts, "similarity": round(sim, 2),
              "possible": (not conflicts) and (not ok) and score >= 2}
    return ok, score, detail


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
@dataclass
class Member:
    row: sqlite3.Row
    published: Optional[datetime]
    rel: str
    sig: Signature
    tags: List[Dict[str, str]]


@dataclass
class Cluster:
    members: List[Member] = field(default_factory=list)
    sig: Signature = field(default_factory=Signature)
    dmin: Optional[datetime] = None
    dmax: Optional[datetime] = None
    reasons: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # article_id -> why it is in this incident
    human_confirmed: bool = False

    def add(self, m: Member, reason: Optional[Dict[str, Any]] = None) -> None:
        self.members.append(m)
        self.sig = self.sig.union(m.sig)
        self.reasons[m.row["article_id"]] = reason or {"type": "seed"}
        if reason and reason.get("type") == "human_approved":
            self.human_confirmed = True
        if m.published:
            self.dmin = m.published if not self.dmin or m.published < self.dmin else self.dmin
            self.dmax = m.published if not self.dmax or m.published > self.dmax else self.dmax

    def article_ids(self) -> Set[int]:
        return {m.row["article_id"] for m in self.members}

    def gap_days(self, when: Optional[datetime]) -> float:
        if when is None or self.dmin is None or self.dmax is None:
            return 0.0  # unknown dates never block, the fact checks still have to pass
        if when < self.dmin:
            return (self.dmin - when).total_seconds() / 86400
        if when > self.dmax:
            return (when - self.dmax).total_seconds() / 86400
        return 0.0


def _pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _range_gap_days(a: Cluster, b: Cluster) -> float:
    if not (a.dmin and a.dmax and b.dmin and b.dmax):
        return 0.0
    return max(0.0, (max(a.dmin, b.dmin) - min(a.dmax, b.dmax)).total_seconds() / 86400)


MAX_SUGGESTIONS_PER_CLUSTER = 3


def cluster_members(members: List[Member], forced: Optional[Set[Tuple[int, int]]] = None,
                    vetoed: Optional[Set[Tuple[int, int]]] = None,
                    ) -> Tuple[List[Cluster], List[Dict[str, Any]]]:
    """Group articles into incidents.

    ``forced`` / ``vetoed`` are article-id pairs decided by a human reviewer (approved / rejected merge
    suggestions); they always win over the automatic rules. Returns ``(clusters, suggestions)`` where
    ``suggestions`` are uncertain merges that were deliberately NOT applied.
    """
    forced, vetoed = set(forced or ()), set(vetoed or ())
    clusters: List[Cluster] = []
    order = sorted(members, key=lambda m: (m.published or datetime.max.replace(tzinfo=timezone.utc), m.row["article_id"]))
    for m in order:
        mid = m.row["article_id"]
        best: Optional[Tuple[int, int, Cluster, Dict[str, Any]]] = None
        for cl in clusters:
            ids = cl.article_ids()
            if any(_pair(mid, x) in vetoed for x in ids):
                continue
            if any(_pair(mid, x) in forced for x in ids):
                cand = (2, 0, cl, {"type": "human_approved"})
            else:
                if cl.gap_days(m.published) > MAX_GAP_DAYS:
                    continue
                ok, score, detail = compare(m.sig, cl.sig)
                if not ok:
                    continue
                cand = (1, score, cl, {"type": "rule", "score": score, **detail})
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best:
            best[2].add(m, best[3])
        else:
            cl = Cluster()
            cl.add(m, {"type": "seed"})
            clusters.append(cl)

    # a human-approved merge may join two clusters that the rules kept apart
    for a, b in sorted(forced):
        ca = next((c for c in clusters if a in c.article_ids()), None)
        cb = next((c for c in clusters if b in c.article_ids()), None)
        if ca and cb and ca is not cb:
            for m in list(cb.members):
                ca.add(m, {"type": "human_approved"})
            clusters.remove(cb)

    # uncertain merges -> human review queue (never applied automatically)
    cands = []
    for i, ci in enumerate(clusters):
        for j in range(i + 1, len(clusters)):
            cj = clusters[j]
            if _range_gap_days(ci, cj) > MAX_GAP_DAYS:
                continue
            ok, score, detail = compare(ci.sig, cj.sig)
            if not detail["possible"]:
                continue
            ids_i, ids_j = ci.article_ids(), cj.article_ids()
            if any(_pair(x, y) in vetoed for x in ids_i for y in ids_j):
                continue
            a, b = _pair(min(ids_i), min(ids_j))
            cands.append((score, i, j, a, b, detail))
    cands.sort(key=lambda c: (-c[0], c[3], c[4]))
    taken: Counter = Counter()
    suggestions: List[Dict[str, Any]] = []
    for score, i, j, a, b, detail in cands:
        if taken[i] >= MAX_SUGGESTIONS_PER_CLUSTER or taken[j] >= MAX_SUGGESTIONS_PER_CLUSTER:
            continue
        taken[i] += 1
        taken[j] += 1
        suggestions.append({"a": a, "b": b, "score": score, "detail": detail,
                            "articles_a": sorted(clusters[i].article_ids()),
                            "articles_b": sorted(clusters[j].article_ids())})
    return clusters, suggestions


# --------------------------------------------------------------------------- #
# Building the incident record from its articles
# --------------------------------------------------------------------------- #
_LEVEL = {"low": 0, "medium": 1, "high": 2}
_TITLE_SUFFIX = re.compile(r"\s+\|\s+.*$")


def _clean_title(t: str) -> str:
    return _TITLE_SUFFIX.sub("", (t or "").strip())


def _tokens(s: str) -> Set[str]:
    return {w for w in re.split(r"\W+", fi.norm(s)) if len(w) > 2}


def _near_duplicate(a: str, b: str) -> bool:
    na, nb = fi.norm(a), fi.norm(b)
    if na in nb or nb in na:
        return True
    ta, tb = _tokens(a), _tokens(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.6


def _evidence_conf(row: sqlite3.Row) -> float:
    return float(row["evidence_confidence"] or 0.0)


def build_incident(cl: Cluster) -> Dict[str, Any]:
    members = sorted(cl.members, key=lambda m: (-_evidence_conf(m.row), m.published or datetime.max.replace(tzinfo=timezone.utc)))
    best = members[0]
    arts = sorted(cl.members, key=lambda m: m.published or datetime.max.replace(tzinfo=timezone.utc))
    articles = [{
        "article_id": m.row["article_id"], "threat_id": m.row["threat_id"], "title": _clean_title(m.row["headline"]),
        "source": m.row["source"], "domain": m.row["domain"], "source_code": m.row["source_code"],
        "url": m.row["url"], "published_at": m.row["published_at"],
        "evidence_confidence": m.row["evidence_confidence"], "content_type": m.row["content_type"],
        "match_basis": cl.reasons.get(m.row["article_id"], {"type": "seed"}),
    } for m in arts]
    domains = {(m.row["domain"] or m.row["source"] or "").lower() for m in cl.members}

    # ---- facts: each item says how many articles state it ------------------------------------
    def tally(getter) -> Counter:
        c: Counter = Counter()
        for m in cl.members:
            for v in getter(m.sig):
                c[v] += 1
        return c

    persons = tally(lambda s: s.counts)
    nats = tally(lambda s: s.nats)
    amounts = tally(lambda s: s.amounts)
    locs = tally(lambda s: s.locs)
    agencies = tally(lambda s: s.agencies)
    items = tally(lambda s: s.items)

    sentences: List[Dict[str, str]] = []
    for m in members:
        for s in fi._split_sentences(m.rel):
            sn = fi.norm(s)
            if not (fi._hits(fi.STRONG_FRAUD, sn) or fi._hits(fi.ENFORCEMENT_STEMS, sn)):
                continue
            if any(_near_duplicate(s, x["text"]) for x in sentences):
                continue
            sentences.append({"text": s, "source": m.row["source"], "url": m.row["url"]})
            if len(sentences) >= MAX_FACT_SENTENCES:
                break
        if len(sentences) >= MAX_FACT_SENTENCES:
            break
    title = _clean_title(best.row["headline"])
    summary = next((x["text"] for x in sentences if fi.norm(x["text"]) != fi.norm(title)
                    and not _near_duplicate(x["text"], title)), "") or (sentences[0]["text"] if sentences else title)

    facts = {
        "origin": "stated_in_source",
        "persons_count": persons.most_common(1)[0][0] if persons else None,
        "persons_counts_seen": [{"value": v, "articles": n} for v, n in persons.most_common()],
        "nationalities": [{"key": k, "label": _NATIONALITIES[k][1], "articles": n} for k, n in nats.most_common()],
        "amounts": [{"value": v, "currency": "SAR", "articles": n} for v, n in sorted(amounts.items())],
        "locations": [{"label": k, "articles": n} for k, n in locs.most_common()],
        "agencies": [{"label": k, "articles": n} for k, n in agencies.most_common()],
        "seized_items": [{"key": k, "label": _ITEM_LABELS[k], "articles": n} for k, n in items.most_common()],
        "fact_sentences": sentences,
        "corroborated_by_articles": len(cl.members),
        "corroborated_by_sources": len(domains),
    }

    # ---- methods stated by the sources (+ channels found in those very sentences) -------------
    methods: Dict[str, Dict[str, Any]] = {}
    for m in members:
        for t in m.tags:
            e = methods.setdefault(t["tag"], {"tag": t["tag"], "label": t["label"], "quotes": [], "articles": 0})
            e["articles"] += 1
            if len(e["quotes"]) < MAX_METHOD_QUOTES and not any(_near_duplicate(t["quote"], q["quote"]) for q in e["quotes"]):
                e["quotes"].append({"quote": t["quote"], "source": m.row["source"], "url": m.row["url"]})
    method_list = sorted(methods.values(), key=lambda e: -e["articles"])
    quote_norm = " ".join(fi.norm(q["quote"]) for e in method_list for q in e["quotes"])
    channels = fi._group_hits(fi.CHANNELS, quote_norm) if quote_norm else []

    indicators = {
        "mentioned_in_source": [{"label": e["label"], "quote": e["quotes"][0]["quote"], "url": e["quotes"][0]["url"],
                                 "articles": e["articles"], "origin": "stated_in_source"} for e in method_list],
        "suggested_by_system": {"origin": "system_template", "items": fi.INDICATORS_TEMPLATE.get(best.row["fraud_category"], [])},
    }

    bank_level = max((m.row["bank_relevance"] or "low" for m in cl.members), key=lambda x: _LEVEL.get(x, 0))
    bonus = min(0.06, 0.02 * (max(1, min(len(domains), 4)) - 1))
    ev = round(min(0.97, _evidence_conf(best.row) + bonus), 2)
    an = max(float(m.row["analysis_confidence"] or 0.0) for m in cl.members)
    dates = [m.published for m in cl.members if m.published]
    return {
        "title": title, "summary": summary[:420],
        "fraud_category": best.row["fraud_category"], "fraud_category_ar": best.row["fraud_category_ar"],
        "first_reported_at": min(dates).isoformat() if dates else None,
        "last_reported_at": max(dates).isoformat() if dates else None,
        "article_count": len(cl.members), "source_count": len(domains),
        "bank_relevance": bank_level, "confidence": round(min(ev, an or ev), 2), "evidence_confidence": ev,
        "locations": [x["label"] for x in facts["locations"]], "agencies": [x["label"] for x in facts["agencies"]],
        "signature": cl.sig.to_json(), "facts": facts, "methods": method_list, "channels": channels,
        "indicators": indicators, "articles": articles,
        "match_basis": "facts: suspect_count/amount/seized_items (strong) + nationality/location/agency/text-overlap "
                        f"(context), published within {MAX_GAP_DAYS} days; conflicting facts veto a merge; "
                        "headline similarity is never used; uncertain pairs go to human review",
        "record_type": fi.CT_CONFIRMED if any(m.row["content_type"] == fi.CT_CONFIRMED for m in cl.members) else fi.CT_REPORTED,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
INCIDENT_TYPES_SQL = "', '".join(fi.INCIDENT_CONTENT_TYPES)
_LOAD_SQL = f"""
    SELECT t.id AS threat_id, t.article_id, t.incident_id, t.content_type, t.fraud_category, t.fraud_category_ar,
           t.bank_relevance, t.evidence_confidence, t.analysis_confidence,
           a.headline, a.article_summary, a.content, a.url, a.source, a.domain, a.published_at, a.source_code
    FROM threat_intel t JOIN articles a ON a.id = t.article_id
    WHERE t.content_type IN ('{INCIDENT_TYPES_SQL}') AND t.is_saudi = 1 AND t.fraud_category IS NOT NULL
"""

REVIEW_AUTO = "auto_merged"          # merged by rule (>= 2 articles)
REVIEW_SINGLE = "single_source"      # one article only: nothing to cross-check
REVIEW_NEEDS = "needs_review"        # an uncertain merge with another incident/article awaits a human
REVIEW_HUMAN = "human_confirmed"     # includes a merge approved by a reviewer


def incident_ref(inc_id: int, first_reported_at: Optional[str], created_at: Optional[str] = None) -> str:
    year = (first_reported_at or created_at or "")[:4]
    year = year if year.isdigit() else str(datetime.now(timezone.utc).year)
    return f"INC-{year}-{int(inc_id):04d}"


def _load_decisions(con: sqlite3.Connection) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    forced: Set[Tuple[int, int]] = set()
    vetoed: Set[Tuple[int, int]] = set()
    for r in con.execute("SELECT article_a, article_b, status FROM merge_suggestions WHERE status IN ('approved','rejected')"):
        (forced if r["status"] == "approved" else vetoed).add(_pair(r["article_a"], r["article_b"]))
    return forced, vetoed


def rebuild(con: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    """Recompute incidents from the stored articles. Idempotent; never touches ``articles``.

    Reviewer decisions stored in ``merge_suggestions`` are honoured on every rebuild.
    """
    own = con is None
    con = con or get_db()
    try:
        con.execute("UPDATE threat_intel SET incident_id = NULL "
                    f"WHERE incident_id IS NOT NULL AND (content_type IS NULL OR content_type NOT IN ('{INCIDENT_TYPES_SQL}'))")
        rows = con.execute(_LOAD_SQL).fetchall()
        members = []
        for r in rows:
            rel = fi.relevant_text(r["headline"] or "", r["article_summary"] or "", r["content"] or "")
            members.append(Member(r, _parse_dt(r["published_at"]), rel, extract_signature(rel), fi.extract_method_tags(rel)))
        forced, vetoed = _load_decisions(con)
        clusters, suggestions = cluster_members(members, forced, vetoed)
        now = datetime.now(timezone.utc).isoformat()

        pending_by_article: Counter = Counter()
        for s in suggestions:
            for aid in s["articles_a"] + s["articles_b"]:
                pending_by_article[aid] += 1

        con.execute("DELETE FROM incident_articles")   # derived link table: rebuilt from scratch every time
        used: Set[int] = set()
        for cl in sorted(clusters, key=lambda c: min(m.row["article_id"] for m in c.members)):
            inc = build_incident(cl)
            pending = len({aid for aid in cl.article_ids() if pending_by_article[aid]})
            if pending:
                status = REVIEW_NEEDS
            elif cl.human_confirmed:
                status = REVIEW_HUMAN
            else:
                status = REVIEW_AUTO if len(cl.members) > 1 else REVIEW_SINGLE
            existing = Counter(m.row["incident_id"] for m in cl.members if m.row["incident_id"] and m.row["incident_id"] not in used)
            inc_id = min(existing, key=lambda k: (-existing[k], k)) if existing else None
            vals = (inc["title"], inc["summary"], inc["fraud_category"], inc["fraud_category_ar"], inc["first_reported_at"],
                    inc["last_reported_at"], inc["article_count"], inc["source_count"], inc["bank_relevance"],
                    inc["confidence"], inc["evidence_confidence"], _j(inc["locations"]), _j(inc["agencies"]),
                    _j(inc["signature"]), _j(inc["facts"]), _j(inc["methods"]), _j(inc["channels"]),
                    _j(inc["indicators"]), _j(inc["articles"]), inc["match_basis"], inc["record_type"], status,
                    1 if pending else 0, now)
            cols = ("title, summary, fraud_category, fraud_category_ar, first_reported_at, last_reported_at, "
                    "article_count, source_count, bank_relevance, confidence, evidence_confidence, locations, "
                    "agencies, signature, facts, methods, channels, indicators, articles, match_basis, "
                    "record_type, review_status, pending_review, updated_at")
            if inc_id and con.execute("SELECT 1 FROM incidents WHERE id = ?", (inc_id,)).fetchone():
                con.execute("UPDATE incidents SET " + ", ".join(f"{c.strip()} = ?" for c in cols.split(",")) + " WHERE id = ?",
                            vals + (inc_id,))
            else:
                cur = con.execute(f"INSERT INTO incidents ({cols}, created_at) VALUES ({','.join('?' * (len(vals) + 1))})",
                                  vals + (now,))
                inc_id = cur.lastrowid
            used.add(inc_id)
            con.executemany("UPDATE threat_intel SET incident_id = ? WHERE id = ?",
                            [(inc_id, m.row["threat_id"]) for m in cl.members])
            primary = inc["articles"][0]["article_id"] if inc["articles"] else None
            con.executemany(
                "INSERT INTO incident_articles (incident_id, article_id, threat_id, role, content_type, match_basis, "
                "published_at, linked_at) VALUES (?,?,?,?,?,?,?,?)",
                [(inc_id, a["article_id"], a["threat_id"], "primary" if a["article_id"] == primary else "coverage",
                  a["content_type"], _j(a["match_basis"]), a["published_at"], now) for a in inc["articles"]])
        if used:
            con.execute(f"DELETE FROM incidents WHERE id NOT IN ({','.join('?' * len(used))})", tuple(used))
        else:
            con.execute("DELETE FROM incidents")

        # merge suggestions: keep human decisions forever, refresh the still-open ones
        keep = {(s["a"], s["b"]) for s in suggestions}
        for s in suggestions:
            con.execute(
                "INSERT INTO merge_suggestions (article_a, article_b, score, shared, conflicts, status, created_at) "
                "VALUES (?,?,?,?,?,'pending',?) ON CONFLICT(article_a, article_b) DO UPDATE SET "
                "score = excluded.score, shared = excluded.shared, conflicts = excluded.conflicts "
                "WHERE merge_suggestions.status = 'pending'",
                (s["a"], s["b"], s["score"], _j(s["detail"]), _j(s["detail"]["conflicts"]), now))
        for r in con.execute("SELECT id, article_a, article_b FROM merge_suggestions WHERE status = 'pending'").fetchall():
            if (r["article_a"], r["article_b"]) not in keep:
                con.execute("DELETE FROM merge_suggestions WHERE id = ?", (r["id"],))
        con.commit()
        return {"incidents": len(clusters), "incident_articles": len(members),
                "articles_merged_into_existing": len(members) - len(clusters),
                "merge_suggestions_pending": len(suggestions)}
    except Exception:
        con.rollback()
        raise
    finally:
        if own:
            con.close()


# --------------------------------------------------------------------------- #
# Human review of uncertain merges
# --------------------------------------------------------------------------- #
def list_suggestions(con: sqlite3.Connection, status: Optional[str] = "pending") -> List[Dict[str, Any]]:
    sql, params = "SELECT * FROM merge_suggestions", ()
    if status:
        sql, params = sql + " WHERE status = ?", (status,)
    out = []
    for r in con.execute(sql + " ORDER BY score DESC, id", params).fetchall():
        def art(aid: int) -> Dict[str, Any]:
            a = con.execute(
                "SELECT a.id, a.headline, a.source, a.url, a.published_at, t.incident_id, t.content_type "
                "FROM articles a LEFT JOIN threat_intel t ON t.article_id = a.id WHERE a.id = ?", (aid,)).fetchone()
            return dict(a) if a else {"id": aid}
        out.append({"id": r["id"], "status": r["status"], "score": r["score"], "article_a": art(r["article_a"]),
                    "article_b": art(r["article_b"]), "why_not_merged": _loads(r["shared"], {}),
                    "reviewer": r["reviewer"], "reason": r["reason"], "decided_at": r["decided_at"],
                    "created_at": r["created_at"]})
    return out


def decide_suggestion(suggestion_id: int, decision: str, reviewer: str = "", reason: str = "") -> Dict[str, Any]:
    """Record a human decision and rebuild. ``approved`` forces the merge, ``rejected`` forbids it."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    con = get_db()
    try:
        row = con.execute("SELECT id FROM merge_suggestions WHERE id = ?", (suggestion_id,)).fetchone()
        if not row:
            raise LookupError("merge suggestion not found")
        con.execute("UPDATE merge_suggestions SET status = ?, reviewer = ?, reason = ?, decided_at = ? WHERE id = ?",
                    (decision, (reviewer or "")[:80], (reason or "")[:500], datetime.now(timezone.utc).isoformat(),
                     suggestion_id))
        con.commit()
        report = rebuild(con)
    finally:
        con.close()
    return {"id": suggestion_id, "decision": decision, "rebuild": report}


# --------------------------------------------------------------------------- #
# Read side
# --------------------------------------------------------------------------- #
_RECORD_TYPE_AR = {fi.CT_CONFIRMED: fi.CONTENT_TYPE_AR[fi.CT_CONFIRMED], fi.CT_REPORTED: fi.CONTENT_TYPE_AR[fi.CT_REPORTED]}
_REVIEW_AR = {REVIEW_AUTO: "دُمجت تلقائياً بحقائق متطابقة", REVIEW_SINGLE: "مصدر واحد - لا يوجد ما يقارَن به",
              REVIEW_NEEDS: "تحتاج مراجعة بشرية (دمج غير مؤكد)", REVIEW_HUMAN: "دمج مؤكد من مراجع"}


def incident_to_dict(r: sqlite3.Row, detail: bool = False) -> Dict[str, Any]:
    articles = _loads(r["articles"], [])
    b_rel = r["bank_relevance"]
    normalized_b_rel = b_rel.lower() if b_rel and b_rel.lower() in ("high", "medium", "low") else None
    raw_sum = (r["summary"] or "").strip()
    if "<" in raw_sum or "http" in raw_sum:
        import html, re
        raw_sum = html.unescape(raw_sum)
        raw_sum = re.sub(r"<[^>]+>", " ", raw_sum)
        raw_sum = re.sub(r"https?://\S+", "", raw_sum)
        raw_sum = re.sub(r"\s+", " ", raw_sum).strip()
    if not raw_sum or len(raw_sum) < 20:
        f_type = r["fraud_type_ar"] if (hasattr(r, "keys") and "fraud_type_ar" in r.keys() and r["fraud_type_ar"]) else "الاحتيال المالي"
        raw_sum = f"واقعة احتيال مالي موثقة بنمط [{f_type}]، باشرت فيها الجهات المعنية إجراءات المتابعة والرصد لحماية التعاملات."

    d: Dict[str, Any] = {
        "id": r["id"], "ref": incident_ref(r["id"], r["first_reported_at"], r["created_at"]),
        "title": r["title"], "summary": raw_sum, "fraud_category": r["fraud_category"],
        "fraud_category_ar": r["fraud_category_ar"], "record_type": r["record_type"],
        "record_type_ar": _RECORD_TYPE_AR.get(r["record_type"], r["record_type"]),
        "review_status": r["review_status"], "review_status_ar": _REVIEW_AR.get(r["review_status"], r["review_status"]),
        "needs_review": bool(r["pending_review"]),
        "first_reported_at": r["first_reported_at"], "last_reported_at": r["last_reported_at"],
        "event_date": None,  # sources rarely state it; never guessed (publication dates are per coverage)
        "article_count": r["article_count"], "source_count": r["source_count"], "bank_relevance": normalized_b_rel,
        "confidence": r["confidence"], "evidence_confidence": r["evidence_confidence"],
        "locations": _loads(r["locations"], []), "agencies": _loads(r["agencies"], []),
        "channels": _loads(r["channels"], []), "articles": articles,
    }
    keys = set(r.keys()) if hasattr(r, "keys") else set()
    if "fraud_type" in keys:
        d.update({
            "fraud_type": r["fraud_type"],
            "fraud_type_ar": r["fraud_type_ar"],
            "incident_country": r["incident_country"],
            "incident_city": r["incident_city"],
            "incident_region": r["incident_region"],
            "victim_location": r["victim_location"],
            "suspect_location": r["suspect_location"],
            "channel": r["channel"],
            "channel_ar": r["channel_ar"],
            "targeted_sector": r["targeted_sector"],
            "targeted_sector_ar": r["targeted_sector_ar"],
            "impersonated_entity": r["impersonated_entity"],
            "reporting_agency": r["reporting_agency"],
            "financial_loss_amount": r["financial_loss_amount"],
            "financial_loss_currency": r["financial_loss_currency"],
            "occurred_at": r["occurred_at"],
            "evidence_text": r["evidence_text"],
            "is_emerging_threat": bool(r["is_emerging_threat"]),
            "emerging_rationale": r["emerging_rationale"],
        })
    if detail:
        d.update({
            "facts": _loads(r["facts"], {}), "methods": _loads(r["methods"], []),
            "indicators": _loads(r["indicators"], {}), "signature": _loads(r["signature"], {}),
            "match_basis": r["match_basis"], "rule": rule_summary_for(r["fraud_category"]),
        })
    return d


def rule_summary_for(category: str) -> Optional[Dict[str, Any]]:
    tmpl = fi.RULE_TEMPLATES.get(category)
    if not tmpl:
        return None
    con = get_db()
    try:
        rid = f"SFI-{tmpl['code']}-001"
        tested = con.execute("SELECT COUNT(*) n FROM rule_tests WHERE rule_id = ?", (rid,)).fetchone()["n"]
    finally:
        con.close()
    return {"rule_id": rid, "name": tmpl["name_ar"], "status": "tested" if tested else "proposed_untested"}


def counts(con: sqlite3.Connection) -> Dict[str, Any]:
    """Headline numbers. The success metric is *independent incidents*, not articles."""
    ct = {r["content_type"]: r["n"] for r in con.execute(
        "SELECT COALESCE(content_type, 'unclassified') AS content_type, COUNT(*) n FROM threat_intel "
        "WHERE is_saudi = 1 AND fraud_category IS NOT NULL GROUP BY 1")}
    inc = con.execute("SELECT COUNT(*) n, COALESCE(SUM(article_count), 0) arts FROM incidents").fetchone()
    last7 = con.execute("SELECT COUNT(*) n FROM incidents WHERE first_reported_at >= datetime('now','-7 days')").fetchone()["n"]
    high = con.execute("SELECT COUNT(*) n FROM incidents WHERE bank_relevance = 'high'").fetchone()["n"]
    cats = {r["fraud_category"]: r["n"] for r in con.execute(
        "SELECT fraud_category, COUNT(*) n FROM incidents GROUP BY fraud_category")}
    rt = {r["record_type"]: r["n"] for r in con.execute("SELECT record_type, COUNT(*) n FROM incidents GROUP BY record_type")}
    rs = {r["review_status"]: r["n"] for r in con.execute("SELECT review_status, COUNT(*) n FROM incidents GROUP BY review_status")}
    pending = con.execute("SELECT COUNT(*) n FROM merge_suggestions WHERE status = 'pending'").fetchone()["n"]
    return {
        "incidents_total": inc["n"], "incident_articles_total": inc["arts"], "incidents_last_7_days": last7,
        "incidents_high_bank_relevance": high, "incident_categories": cats,
        "incidents_confirmed": rt.get(fi.CT_CONFIRMED, 0), "incidents_reported": rt.get(fi.CT_REPORTED, 0),
        "incidents_needs_review": rs.get(REVIEW_NEEDS, 0), "incidents_by_review_status": rs,
        "merge_suggestions_pending": pending,
        "warnings_total": ct.get(fi.CT_WARNING, 0), "awareness_total": ct.get(fi.CT_AWARENESS, 0),
        "excluded_general_cybersecurity": ct.get(fi.CT_GENERAL, 0), "excluded_unrelated": ct.get(fi.CT_UNRELATED, 0),
        "unclassified": ct.get("unclassified", 0), "by_content_type": ct,
    }


def build_patterns(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Fraud patterns = stated methods aggregated over incidents (+ warning/awareness articles).
    
    Implements a strict 6-tier evidence-based classification hierarchy:
    - Confirmed Fraud Pattern: >= 2 confirmed incidents across independent sources.
    - Observed Repeated Pattern: >= 2 incidents (or 1 incident + multiple warnings).
    - Reported Fraud Method: 1 incident.
    - Warning-Based Intelligence: 0 incidents + >= 1 warning/awareness articles.
    - Emerging Pattern: Recent signals with preliminary indicators.
    - Insufficient Evidence: 0 incidents + 0 warnings.
    """
    pats: Dict[str, Dict[str, Any]] = {}

    def entry(tag: str, label: str) -> Dict[str, Any]:
        return pats.setdefault(tag, {
            "key": tag, "label_ar": label, "origin": "stated_in_source",
            "incidents": [], "warning_articles": [], "categories": Counter(),
            "channels": set(), "evidence": [], "sources": set(),
            "first_seen": None, "last_seen": None
        })

    def seen(p: Dict[str, Any], when: Optional[str]) -> None:
        if when:
            p["first_seen"] = when if not p["first_seen"] or when < p["first_seen"] else p["first_seen"]
            p["last_seen"] = when if not p["last_seen"] or when > p["last_seen"] else p["last_seen"]

    # 1. Incidents (confirmed or reported criminal cases)
    for r in con.execute("SELECT * FROM incidents ORDER BY last_reported_at DESC"):
        rec_type = r["record_type"] or "confirmed_incident"
        src_name = "مصادر الرصد"
        if r["agencies"]:
            ag_list = _loads(r["agencies"], [])
            if ag_list:
                src_name = ag_list[0]
        elif r["articles"]:
            art_list = _loads(r["articles"], [])
            if art_list and isinstance(art_list[0], dict) and art_list[0].get("source"):
                src_name = art_list[0]["source"]

        for m in _loads(r["methods"], []):
            if isinstance(m, str):
                tag = r["fraud_category"] or "pattern"
                label = m
                quote = r["title"]
                url = ""
                source = src_name
            elif isinstance(m, dict):
                tag = m.get("tag") or r["fraud_category"] or "pattern"
                label = m.get("label") or tag
                q = (m.get("quotes") or [{}])[0] if isinstance(m.get("quotes"), list) else {}
                quote = q.get("quote") or m.get("quote") or r["title"]
                url = q.get("url") or m.get("url") or ""
                source = q.get("source") or m.get("source") or src_name
            else:
                continue

            p = entry(tag, label)
            p["incidents"].append({
                "id": r["id"],
                "title": r["title"],
                "first_reported_at": r["first_reported_at"],
                "article_count": r["article_count"],
                "record_type": rec_type,
            })
            p["categories"][r["fraud_category_ar"]] += 1
            p["channels"].update(_loads(r["channels"], []))
            if source:
                p["sources"].add(source)
            seen(p, r["first_reported_at"])
            seen(p, r["last_reported_at"])
            p["evidence"].append({
                "type": "incident",
                "incident_id": r["id"],
                "title": r["title"],
                "quote": quote,
                "url": url,
                "source": source
            })

    # 2. Warnings & Awareness intelligence
    warn_rows = con.execute(
        "SELECT t.id, t.title, t.content_type, t.fraud_category, t.fraud_category_ar, t.verified_facts, "
        "t.analytical_assessment, t.detection_indicators, t.preventive_recommendations, "
        "a.headline, a.content, a.article_summary, a.url, a.source, a.published_at "
        "FROM threat_intel t JOIN articles a ON a.id = t.article_id "
        "WHERE t.is_saudi = 1 AND t.content_type IN (?, ?) AND t.fraud_category IS NOT NULL",
        (fi.CT_WARNING, fi.CT_AWARENESS)
    ).fetchall()

    for r in warn_rows:
        vf = r["verified_facts"]
        m_tags = []
        if vf and isinstance(vf, str) and vf.strip().startswith("{"):
            try:
                m_tags = json.loads(vf).get("method_tags", [])
            except Exception:
                pass
        elif isinstance(vf, dict):
            m_tags = vf.get("method_tags", [])

        if not m_tags:
            full_text = " ".join(x for x in (r["title"], r["headline"], r["article_summary"], r["content"]) if x)
            m_tags = fi.extract_method_tags(full_text)

        for m in m_tags:
            if isinstance(m, str):
                tag = m
                label = m
                quote = r["title"]
            elif isinstance(m, dict):
                tag = m.get("tag") or "warning_tag"
                label = m.get("label") or tag
                quote = m.get("quote") or r["title"]
            else:
                continue

            p = entry(tag, label)
            p["warning_articles"].append({
                "threat_id": r["id"],
                "title": r["title"],
                "url": r["url"],
                "source": r["source"],
                "published_at": r["published_at"],
            })
            p["categories"][r["fraud_category_ar"]] += 0
            if r["source"]:
                p["sources"].add(r["source"])
            seen(p, r["published_at"])
            p["evidence"].append({
                "type": "warning" if r["content_type"] == fi.CT_WARNING else "awareness",
                "threat_id": r["id"],
                "title": r["title"],
                "quote": quote,
                "url": r["url"],
                "source": r["source"],
            })

    DOMAIN_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
        "fraudulent_ads": {
            "description": "أسلوب احتيالي يعتمد على نشر إعلانات ممولة أو منشورات ترويجية عبر المنصات الرقمية وشبكات التواصل لاستدراج المستخدمين نحو صفحات تصيد أو تطبيقات مراسلة خاصة.",
            "steps": [
                "نشر إعلانات ترويجية احتيالية أو وهمية على منصات التواصل (إكس، فيسبوك، إنستغرام) تنتحل صفة علامات تجارية أو فرص استثمارية.",
                "تضمين روابط خارجية تنقل الضحية إلى صفحات هبوط وهمية أو مجموعات محادثة خاصة (واتساب/تيليجرام).",
                "استدراج الضحايا لإجراء عمليات دفع إلكتروني أو تحويل مالي مقابل سلع غير موجودة أو استثمارات وهمية."
            ],
            "channels": ["منصات التواصل الاجتماعي", "الإعلانات الممولة", "تطبيقات المراسلة"],
            "impersonated_entities": ["شركات تجارية", "منصات استثمارية", "جهات إعلانية"],
            "targeted_victims": ["مستخدمو شبكات التواصل الاجتماعي", "الباحثون عن عروض تسوق أو استثمار"],
            "potential_bank_risks": [
                "مخاطر استدراج العملاء لإجراء تحويلات بنكية طوعية لحسابات احتيالية (Authorized Push Payment).",
                "مدفوعات بطاقات عبر بوابات دفع غير موثوقة ينتج عنها نزاعات وعمليات احتيال عكسية (Chargebacks).",
                "مخاطر السمعة المصرفية واستغلال أسماء البنوك في الإعلانات الترويجية الاحتيالية."
            ],
            "fraud_indicators": [
                "عمليات دفع إلكتروني لروابط أو بوابات غير مصنفة تلي النقر على إعلانات خارجية.",
                "حركات تحويل مالي فورية بمبالغ نمطية متكررة لحسابات أفراد حديثة النشأة.",
                "تكرار الشكاوى حول مدفوعات لجهات تجارية غير مسجلة في سجلات التجارة الإلكترونية."
            ],
            "detection_hypotheses": [
                "SFI-ADV-001: رصد حركة مدفوعات متكررة لروابط وبوابات دفع مشبوهة مقترنة بإعلانات مضللة.",
                "SFI-FIN-001: رصد تحويلات عملاء لحسابات وسيطة (Mules) عقب التفاعل مع عروض ترويجية غير موثوقة."
            ],
            "required_data": [
                "سجلات العمليات المصرفية (المبلغ، وقت العملية، القناة المستهدفة).",
                "بيانات المستفيد ورمز التاجر (Merchant Category Code - MCC).",
                "سجلات المصادقة الثنائية (3DS) وعناوين IP المرتبطة بالعملية."
            ],
            "uncertainties_and_limitations": [
                "التحليل مستند إلى مواد الرصد والتوعية الإعلامية المفتوحة؛ لا تتوفر سجلات داخلية لتحديد الحجم المالي الفعلي للعمليات.",
                "الخطوات المعروضة تمثل المسار النموذجي الموثق في التحذير، ولا تعني بالضرورة ثبوت ارتكابها في قضية جنائية واحدة بعينها ما لم يصدر بها بيان ضبط رسمي."
            ]
        },
        "bank_impersonation": {
            "description": "انتحال صفة موظفي البنوك أو ممثلي خدمة العملاء لخداع الضحايا والحصول على بياناتهم السرية ورموز التحقق.",
            "steps": [
                "التواصل مع الضحية عبر مكالمات هاتفية أو رسائل نصية زاعمين وجود مشكلة أمنية أو تحديث للحساب البنكي.",
                "طلب رمز التحقق لمرة واحدة (OTP) أو بيانات بطاقة الصراف أو اسم المستخدم وكلمة المرور.",
                "تنفيذ عمليات سحب مالي فوري أو تحويلات عبر القنوات الإلكترونية للبنك."
            ],
            "channels": ["مكالمات هاتفية", "رسائل نصية قصيرة (SMS)", "تطبيقات مراسلة"],
            "impersonated_entities": ["بنوك سعودية", "خدمة عملاء المصارف", "إدارات مكافحة الاحتيال"],
            "targeted_victims": ["عملاء البنوك والمصارف"],
            "potential_bank_risks": [
                "اختراق مباشر لحسابات العملاء وسحب الأرصدة (Account Takeover - ATO).",
                "مطالبات التعويض ونزاعات العمليات غير المصرح بها.",
                "مخاطر الثقة والأمان في الخدمات المصرفية الرقمية."
            ],
            "fraud_indicators": [
                "تغيير جهاز الدخول أو كلمة المرور متبوعاً بتحويل مالي فوري لكامل الرصيد خلال دقائق.",
                "إضافة مستفيد جديد والتحويل له فور استلام رمز OTP خارج أوقات الاستخدام المعتادة للعميل.",
                "عمليات دخول متعددة من عناوين IP أو مواقع جغرافية غير معتادة."
            ],
            "detection_hypotheses": [
                "SFI-BNK-001: رصد تحويل مالي بمبلغ كبير خلال 10 دقائق من تسجيل جهاز دخول جديد أو طلب OTP."
            ],
            "required_data": [
                "سجلات الدخول وتغيير الأجهزة (Device Fingerprinting / Telemetry).",
                "سجلات طلبات OTP وقنوات إرسالها.",
                "سجلات التحويلات المالية الفورية (سريع)."
            ],
            "uncertainties_and_limitations": [
                "التحليل مبني على قضايا الضبط الجنائي المنشورة؛ تقييم المعاملات يتطلب مطابقة البيانات السلوكية للعملاء داخلياً."
            ]
        },
        "gov_impersonation": {
            "description": "انتحال صفة مسؤولين أو منصات حكومية (مثل أبشر، الشرطة، النيابة، أو المحاكم) لممارسة الضغط والترهيب على الضحايا.",
            "steps": [
                "إرسال رسائل أو الاتصال مدعين وجود مخالفة قانونية، تعليق خدمات، أو إيقاف هوية.",
                "طلب تسوية مالية فورية أو تسجيل الدخول عبر روابط وهمية مطابقة للمنصات الحكومية.",
                "الاستيلاء على مبالغ السداد أو بيانات النفاذ الوطني واستغلالها في الدخول للحسابات البنكية."
            ],
            "channels": ["مكالمات هاتفية", "رسائل نصية", "روابط تصيد إلكتروني"],
            "impersonated_entities": ["وزارة الداخلية", "منصة أبشر", "النيابة العامة", "هيئة الزكاة والضريبة والجمارك"],
            "targeted_victims": ["المواطنون والمقيمون الخاضعون لخدمات منصات حكومية"],
            "potential_bank_risks": [
                "استغلال بيانات النفاذ الوطني للوصول المباشر إلى التطبيقات المصرفية المرتبطة.",
                "تحويلات سريعة من حسابات الضحايا لحسابات وسيطة (Mules)."
            ],
            "fraud_indicators": [
                "تسجيل دخول مصرفي يتبع زيارة روابط نفاذ وطني غير رسمية.",
                "عمليات تحويل مالية فورية تحت مسميات سداد رسوم أو غرامات وهمية."
            ],
            "detection_hypotheses": [
                "SFI-GOV-001: رصد محاولات تحويل مالي مقترنة بنشاط احتيالي منتحل للصفة الرسمية."
            ],
            "required_data": [
                "سجلات المصادقة الثنائية وتغيير كلمات المرور.",
                "سجلات العمليات والمدفوعات الحكومية المعكوسة."
            ],
            "uncertainties_and_limitations": [
                "تعتمد الأدلة على بيانات الضبط الأمني؛ تتبع القنوات يتطلب تنسيقاً مصرفياً وأمنياً مشتركاً."
            ]
        },
        "social_engineering": {
            "description": "التلاعب النفسي بالضحايا لاستدراجهم للإفصاح عن معلومات سرية أو القيام بتحويلات مالية تحت تأثير الإلحاح أو الخوف.",
            "steps": [
                "بناء علاقة ثقة أو اصطناع حالة طارئة تدفع الضحية لاتخاذ قرار سريع دون تفكير.",
                "طلب المساعدة المالية أو دفع رسوم مستعجلة عبر قنوات تحويل بنكي.",
                "انقطاع التواصل بمجرد إتمام التحويل المالي."
            ],
            "channels": ["منصات التواصل الاجتماعي", "تطبيقات المراسلة", "مكالمات هاتفية"],
            "impersonated_entities": ["معارف وأقارب", "شخصيات عامة", "جهات دعم فني"],
            "targeted_victims": ["مستخدمو المنصات الرقمية وكبار السن"],
            "potential_bank_risks": [
                "عمليات احتيال الدفع المصرح بها (APP Fraud) التي يصعب استردادها عبر القنوات التقليدية."
            ],
            "fraud_indicators": [
                "تحويلات مالية استثنائية لمستفيدين جدد دون سوابق تعامل.",
                "مكالمة هاتفية جارية بالتزامن مع إجراء عملية التحويل المصرفي."
            ],
            "detection_hypotheses": [
                "SFI-SOC-001: رصد تحويلات مالية متسرعة مصاحبة لمكالمات هاتفية نشطة على تطبيق الجوال."
            ],
            "required_data": ["حالة المكالمة أثناء جلسة التطبيق البنكي", "سجل تاريخ المعاملات مع المستفيد"],
            "uncertainties_and_limitations": [
                "تتفاوت أساليب الهندسة الاجتماعية بشكل كبير ويصعب حصرها في نمط تقني واحد."
            ]
        },
        "ai_assisted": {
            "description": "توظيف تقنيات الذكاء الاصطناعي التوليدي مثل تزييف الأصوات (Voice Cloning) والصور لتعزيز مصداقية عمليات الاحتيال.",
            "steps": [
                "جمع عينات صوتية أو مرئية للضحية أو أقربائه من منصات التواصل الاجتماعي.",
                "توليد تسجيلات صوتية تحاكي صوت الشخص في حالة طوارئ لطلب تحويل مالي عاجل.",
                "استلام المبالغ وتحويلها عبر سلسلة من الحسابات الوسيطة لإخفاء أثرها."
            ],
            "channels": ["رسائل صوتية", "مكالمات هاتفية متقدمة", "فيديوهات مزيفة (Deepfake)"],
            "impersonated_entities": ["أفراد العائلة", "مسؤولون تنفيذيون بالشركات", "شخصيات عامة"],
            "targeted_victims": ["أهالي وأقارب المستهدفين", "إدارات المالية والتحويل بالشركات"],
            "potential_bank_risks": [
                "احتيال الرؤساء التنفيذيين (CEO Fraud) وتحويلات تجارية غير مصرح بها.",
                "صعوبة الكشف المبكر نظراً لمصداقية التزييف العالية."
            ],
            "fraud_indicators": [
                "طلبات تحويل مالي عاجلة مع تشديد على السرية وعدم الرجوع للقنوات المعتادة.",
                "أوامر دفع مستعجلة خارج الدورة المستندية المعتادة للمؤسسة."
            ],
            "detection_hypotheses": [
                "SFI-AI-001: تعزيز المصادقة متعددة القنوات في التحويلات العاجلة ذات الحجم الكبير."
            ],
            "required_data": ["سجلات التحويلات الاستثنائية", "قنوات تأكيد الأوامر المالية"],
            "uncertainties_and_limitations": [
                "أسلوب ناشئ في مراحله الأولى محلياً؛ يعتمد التحليل على تحذيرات تقنية واستشرافية."
            ]
        }
    }

    out = []
    direct_bank_keywords = ["حساب بنكي", "حسابات مصرفية", "بطاقة", "بطاقات", "تحويل مالي", "تحويلات غير مشروعة",
                            "رمز التحقق", "otp", "تطبيق بنكي", "سداد", "فيزا", "ماستركارد", "مدى", "مصرف", "بنوك", "الحسابات البنكية"]
    indirect_bank_keywords = ["متجر", "استثمار", "تداول", "شراء إلكتروني", "إعلانات", "دفع إلكتروني", "محفظة", "عملات", "منصات", "وظائف"]

    for p in pats.values():
        conf_incs = [i for i in p["incidents"] if i.get("record_type") == "confirmed_incident"]
        rep_incs = [i for i in p["incidents"] if i.get("record_type") != "confirmed_incident"]
        n_conf = len(conf_incs)
        n_rep = len(rep_incs)
        n_inc = n_conf + n_rep
        n_warn = len(p["warning_articles"])
        n_sources = max(len(p["sources"]), 1 if (n_inc + n_warn) > 0 else 0)

        # 1. Classification Tier
        if n_conf >= 2 and n_sources >= 2:
            tier = "confirmed_fraud_pattern"
            tier_ar = "نمط احتيالي مؤكد وموثق"
            tier_rationale = f"تم تصنيف النمط كـ«نمط احتيالي مؤكد وموثق» نظراً لتوثيق {n_conf} حوادث جنائية رسمية مؤكدة مدعومة بمصادر مستقلة متعددة ({n_sources} مصادر)."
            summary = f"نمط احتيالي مؤكد وموثق مستخرج من {n_conf} حوادث جنائية رسمية مدعومة بـ {n_sources} مصادر مستقلة."
        elif n_inc >= 2 or (n_inc >= 1 and n_warn >= 2):
            tier = "observed_repeated_pattern"
            tier_ar = "أسلوب احتيالي متكرر الرصد"
            tier_rationale = f"تم تصنيف النمط كـ«أسلوب احتيالي متكرر الرصد» لتكرار ظهوره في {n_inc} حوادث و{n_warn} تحذيرات، مع عدم اكتمال شروط النمط المؤكد بالكامل."
            summary = f"أسلوب احتيالي متكرر الرصد عبر {n_inc} حوادث و{n_warn} تحذيرات من مصادر متعددة."
        elif n_inc == 1:
            tier = "reported_fraud_method"
            tier_ar = "أسلوب احتيالي مبلّغ عنه"
            tier_rationale = f"أسلوب احتيالي مستخرج من واقعة واحدة مبلّغ عنها، بانتظار استكمال التحقق القضائي أو تسجيل وقائع متطابقة إضافية لتأكيد صفة التكرار."
            summary = f"أسلوب احتيالي مبلّغ عنه في واقعة واحدة موثقة عبر {n_sources} مصادر إخبارية ورسمية."
        elif n_inc == 0 and n_warn >= 1:
            tier = "warning_based_intelligence"
            tier_ar = "تحذير من أسلوب احتيالي محتمل"
            tier_rationale = f"تم تصنيف المحتوى كـ«تحذير من أسلوب احتيالي محتمل» استناداً إلى صدور {n_warn} تحذيرات أو حملات توعوية موثقة دون تسجيل حوادث جنائية مؤكدة مرتبطة به حتى تاريخه. تنبيه: تكرار المقالات أو التغطيات عن نفس التحذير لا يُعد دليلاً على تكرار الاحتيال الفعلي."
            summary = f"تحذير ميتا فيو صادر عبر {n_warn} مواد تحذيرية وتوعوية حول أسلوب احتيال محتمل، دون تسجيل حوادث جنائية مؤكدة مرتبطة به حتى الآن."
        elif (n_inc + n_warn) > 0:
            tier = "emerging_pattern"
            tier_ar = "أسلوب احتيالي ناشئ قيد الرصد"
            tier_rationale = "أسلوب احتيالي ناشئ ذو مؤشرات أولية قيد الرصد ومتابعة ميتا فيو."
            summary = f"أسلوب احتيالي ناشئ قيد الرصد الأولي عبر {n_sources} مصادر."
        else:
            tier = "insufficient_evidence"
            tier_ar = "بيانات توثيقية غير كافية"
            tier_rationale = "لا تتوافر أدلة موثقة كافية في السجلات الحالية لدعم تصنيف هذا الأسلوب."
            summary = "بيانات توثيقية غير كافية لدعم اعتماد النمط."

        # 2. Bank Relevance Analysis
        quotes_text = " ".join(e["quote"] for e in p["evidence"]) + " " + p["label_ar"]
        has_direct = any(kw in quotes_text for kw in direct_bank_keywords) or p["key"] == "bank_impersonation"
        has_indirect = any(kw in quotes_text for kw in indirect_bank_keywords) or p["key"] in ("fraudulent_ads", "ecommerce_fraud", "investment_fraud")

        if has_direct:
            bank_rel = "high"
            bank_rel_ar = "عالية (مباشرة)"
            bank_rel_rat = "صلة مصرفية مباشرة: يتضمن الأسلوب استهدافاً مباشراً للحسابات المصرفية، أو طلب بيانات بطاقات الدفع ورموز التحقق (OTP)، أو إجراء تحويلات مالية غير مشروعة."
        elif has_indirect:
            bank_rel = "medium"
            bank_rel_ar = "متوسطة (تقدير تحليلي)"
            bank_rel_rat = "صلة مصرفية محتملة (تقدير تحليلي): يستغل المحتالون منصات رقمية أو إعلانات لاستدراج الضحايا نحو قنوات دفع أو تحويل أموال لاحقة، دون استهداف مباشر للبنية التحتية للبنك."
        else:
            bank_rel = "low"
            bank_rel_ar = "منخفضة / غير مباشرة"
            bank_rel_rat = "صلة مصرفية منخفضة: لا تشير الأدلة المرصودة حالياً إلى وجود ارتباط مباشر بالمعاملات أو القنوات البنكية."

        # 3. Confidence Metrics
        if n_conf >= 2:
            conf_score = 0.88
            conf_level = "high"
            conf_level_ar = "مرتفع"
            conf_rat = "درجة الثقة مستندة إلى بيانات الضبط الرسمية الصادرة عن الجهات المختصة وتعدد المصادر المستقلة."
        elif n_inc >= 1 or n_warn >= 2:
            conf_score = 0.72
            conf_level = "medium"
            conf_level_ar = "متوسط"
            conf_rat = "درجة الثقة مبنية على تحذيرات وتغطيات موثقة ومطابقة في النصوص الصريحة للمصادر."
        elif n_warn == 1:
            conf_score = 0.65
            conf_level = "medium"
            conf_level_ar = "متوسط"
            conf_rat = "درجة الثقة مستندة إلى مصدر موثق ينقل تحذيراً رسمياً أو بياناً صريحاً، مع غياب حوادث جنائية مسجلة حتى الآن."
        else:
            conf_score = 0.40
            conf_level = "low"
            conf_level_ar = "منخفض"
            conf_rat = "درجة الثقة منخفضة نظراً لمحدودية الأدلة والوقائع المستقلة."

        # 4. Modus Operandi & Banking Analysis Blueprint
        bp = DOMAIN_BLUEPRINTS.get(p["key"])
        if bp:
            how_it_works = {
                "description": bp["description"],
                "steps": bp["steps"],
                "channels": bp["channels"],
                "impersonated_entities": bp["impersonated_entities"],
                "targeted_victims": bp["targeted_victims"],
            }
            banking_analysis = {
                "potential_bank_risks": bp["potential_bank_risks"],
                "fraud_indicators": bp["fraud_indicators"],
                "detection_hypotheses": bp["detection_hypotheses"],
                "required_data": bp["required_data"],
                "uncertainties_and_limitations": bp["uncertainties_and_limitations"],
            }
        else:
            channels_list = sorted(p["channels"]) if p["channels"] else ["قنوات رقمية / غير محددة بدقة في المصدر"]
            how_it_works = {
                "description": f"أسلوب احتيالي متمثل في «{p['label_ar']}» كما ورد صراحة في نصوص المصادر المرصودة.",
                "steps": ["خطوات التنفيذ التفصيلية لم تُوثّق في الأدلة المتاحة؛ يُعرض التوصيف العام المعتمد فقط تجنباً للاستنتاج غير المدعوم."],
                "channels": channels_list,
                "impersonated_entities": ["لم تذكر المصادر جهات منتحلة محددة لهذا النمط"],
                "targeted_victims": ["عموم المتصفحين والمستخدمين"],
            }
            banking_analysis = {
                "potential_bank_risks": [
                    "مخاطر استغلال القنوات المصرفية في تسوية معاملات احتيالية.",
                    "مخاطر النزاعات المصرفية في حال مرور العمليات عبر وسائل الدفع التابعة للبنك."
                ],
                "fraud_indicators": [
                    "حركات تحويل مالي غير معتادة مطابقة لتوقيت البلاغات.",
                    "عمليات دفع إلكتروني عبر منصات مشبوهة."
                ],
                "detection_hypotheses": [
                    f"SFI-GEN-001: رصد الأنماط المالية المقترنة بـ «{p['label_ar']}» عبر قواعد البيانات السلوكية."
                ],
                "required_data": [
                    "سجلات العمليات المصرفية ومطابقة التوقيت.",
                    "بيانات أطراف التحويل."
                ],
                "uncertainties_and_limitations": [
                    "التحليل مبني على نصوص المصادر المتاحة دون تعميم غير مبرر.",
                    "يتطلب تأكيد المخاطر المصرفية مراجعة العمليات الفعلية داخل البنك."
                ]
            }

        out.append({
            "key": p["key"],
            "label_ar": p["label_ar"],
            "origin": p["origin"],
            "incident_count": n_inc,
            "confirmed_incident_count": n_conf,
            "reported_incident_count": n_rep,
            "warning_article_count": n_warn,
            "source_count": n_sources,
            "pattern_tier": tier,
            "pattern_tier_ar": tier_ar,
            "classification_rationale": tier_rationale,
            "summary": summary,
            "bank_relevance": bank_rel,
            "bank_relevance_ar": bank_rel_ar,
            "bank_relevance_rationale": bank_rel_rat,
            "confidence_score": conf_score,
            "confidence_level": conf_level,
            "confidence_level_ar": conf_level_ar,
            "confidence_rationale": conf_rat,
            "support": "multiple_incidents" if n_inc >= 2 else ("single_incident" if n_inc == 1 else "warnings_only"),
            "categories": [{"label_ar": k, "incidents": v} for k, v in p["categories"].most_common()],
            "channels": sorted(p["channels"]),
            "first_seen": p["first_seen"],
            "last_seen": p["last_seen"],
            "how_it_works": how_it_works,
            "banking_analysis": banking_analysis,
            "incidents": p["incidents"],
            "warning_articles": p["warning_articles"],
            "evidence": p["evidence"][:12],
        })

    tier_weight = {
        "confirmed_fraud_pattern": 5,
        "observed_repeated_pattern": 4,
        "reported_fraud_method": 3,
        "warning_based_intelligence": 2,
        "emerging_pattern": 1,
        "insufficient_evidence": 0,
    }
    out.sort(key=lambda x: (-tier_weight.get(x["pattern_tier"], 0), -x["incident_count"], -x["warning_article_count"], x["key"]))
    return out


def build_rules(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """One detection-rule hypothesis per fraud category, tied to the incidents/patterns behind it.

    A rule is ``tested`` only if a real backtest/shadow result exists in ``rule_tests``; otherwise it
    stays ``proposed_untested``. Evidence shows the *threat exists*, it does not validate the rule.
    """
    tests: Dict[str, List[Dict[str, Any]]] = {}
    for r in con.execute("SELECT * FROM rule_tests ORDER BY tested_at DESC"):
        tests.setdefault(r["rule_id"], []).append({k: r[k] for k in r.keys()})
    incs: Dict[str, List[sqlite3.Row]] = {}
    for r in con.execute("SELECT * FROM incidents ORDER BY last_reported_at DESC"):
        incs.setdefault(r["fraud_category"], []).append(r)
    warns: Dict[str, List[sqlite3.Row]] = {}
    for r in con.execute(
            "SELECT t.id, t.title, t.fraud_category, a.url, a.source FROM threat_intel t "
            "JOIN articles a ON a.id = t.article_id WHERE t.is_saudi = 1 AND t.content_type = ? "
            "AND t.fraud_category IS NOT NULL", (fi.CT_WARNING,)):
        warns.setdefault(r["fraud_category"], []).append(r)
    out = []
    for cat in fi.CATEGORY_ORDER:
        rows, wrows = incs.get(cat, []), warns.get(cat, [])
        if not rows and not wrows:
            continue
        tmpl = fi.RULE_TEMPLATES[cat]
        rid = f"SFI-{tmpl['code']}-001"
        tag_counts: Counter = Counter()
        tag_labels: Dict[str, str] = {}
        for r in rows:
            for m in _loads(r["methods"], []):
                if isinstance(m, str):
                    t_key = r["fraud_category"] or "pattern"
                    tag_counts[t_key] += 1
                    tag_labels[t_key] = m
                elif isinstance(m, dict):
                    t_key = m.get("tag") or r["fraud_category"] or "pattern"
                    tag_counts[t_key] += 1
                    tag_labels[t_key] = m.get("label") or t_key
        t = tests.get(rid, [])
        out.append({
            "rule_id": rid, "category": cat, "category_ar": fi.CATEGORY_AR[cat], "name": tmpl["name_ar"],
            "hypothesis": tmpl["hypothesis_ar"], "logic": tmpl["logic"], "required_data": tmpl["required_data"],
            "status": "tested" if t else "proposed_untested", "is_inference": True,
            "test_results": t[:5],
            "validation_plan": ["Backtest على 90 يوماً من المعاملات مع تسميات احتيال مؤكدة من إدارة الاحتيال",
                                "قياس Precision عند حجم تنبيهات مقبول + Recall مقابل الحالات المؤكدة",
                                "تشغيل Shadow mode أسبوعين قبل التفعيل ومعايرة العتبات"],
            "origin": "system_template_hypothesis",
            "evidence": {
                "basis": "confirmed_incidents" if rows else "warnings_only",
                "note": "الأدلة تثبت وجود التهديد في المصادر، ولا تثبت صحة القاعدة؛ القاعدة فرضية لم تُختبر على بيانات المعاملات."
                        if not t else "للقاعدة نتائج اختبار مسجلة.",
                "incident_count": len(rows),
                "incident_ids": [r["id"] for r in rows],
                "incidents": [{"id": r["id"], "title": r["title"], "article_count": r["article_count"],
                               "first_reported_at": r["first_reported_at"],
                               "article_urls": [a.get("url") for a in _loads(r["articles"], []) if isinstance(a, dict) and a.get("url")]} for r in rows[:5]],
                "warning_article_count": len(wrows),
                "warning_articles": [{"threat_id": w["id"], "title": w["title"], "url": w["url"]} for w in wrows[:5]],
                "patterns": [{"key": k, "label_ar": tag_labels.get(k, k), "incident_count": n} for k, n in tag_counts.most_common()],
            },
        })
    out.sort(key=lambda x: (-x["evidence"]["incident_count"], -x["evidence"]["warning_article_count"]))
    return out


def record_rule_test(rule_id: str, dataset: str, precision: Optional[float], recall: Optional[float],
                     alerts: Optional[int], notes: str = "", tester: str = "") -> None:
    """Store the outcome of a REAL backtest / shadow run (the only way a rule becomes ``tested``)."""
    con = get_db()
    try:
        con.execute("INSERT INTO rule_tests (rule_id, tested_at, dataset, precision_, recall, alerts, notes, tester) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (rule_id, datetime.now(timezone.utc).isoformat(), dataset, precision, recall, alerts, notes, tester))
        con.commit()
    finally:
        con.close()


def build_report(con: sqlite3.Connection, examples: int = 5) -> Dict[str, Any]:
    """Before/after report: how many articles vs independent incidents, with real links."""
    c = counts(con)
    saudi_articles = con.execute(
        "SELECT COUNT(*) n FROM threat_intel WHERE is_saudi = 1 AND fraud_category IS NOT NULL").fetchone()["n"]
    stored_articles = con.execute("SELECT COUNT(*) n FROM articles").fetchone()["n"]
    inc_rows = con.execute("SELECT * FROM incidents ORDER BY COALESCE(last_reported_at, created_at) DESC, id DESC").fetchall()
    incident_list = []
    for r in inc_rows:
        d = incident_to_dict(r)
        incident_list.append({
            "id": d["id"], "ref": d["ref"], "title": d["title"], "record_type": d["record_type"],
            "review_status": d["review_status"], "category": d["fraud_category_ar"], "locations": d["locations"],
            "article_count": d["article_count"], "source_count": d["source_count"],
            "first_reported_at": d["first_reported_at"], "last_reported_at": d["last_reported_at"],
            "coverage": [{"source": (a.get("source") or a.get("reporting_agency") or "مصدر رسمي") if isinstance(a, dict) else "مصدر رسمي",
                          "title": (a.get("title") or a.get("headline") or d["title"]) if isinstance(a, dict) else str(a),
                          "url": a.get("url") or "" if isinstance(a, dict) else "",
                          "published_at": (a.get("published_at") or d["first_reported_at"]) if isinstance(a, dict) else d["first_reported_at"],
                          "why_linked": (a.get("match_basis") or {}).get("type") if isinstance(a, dict) and isinstance(a.get("match_basis"), dict) else "seed",
                          "shared_facts": ((a.get("match_basis") or {}).get("strong", [])
                                           + (a.get("match_basis") or {}).get("context", [])) if isinstance(a, dict) and isinstance(a.get("match_basis"), dict) else []}
                         for a in d["articles"]],
        })

    def sample(ctype: str) -> List[Dict[str, Any]]:
        return [{"title": r["headline"], "source": r["source"], "url": r["url"], "published_at": r["published_at"],
                 "reasons": _loads(r["verified_facts"], {}).get("content_type_reasons", [])}
                for r in con.execute(
                    "SELECT a.headline, a.source, a.url, a.published_at, t.verified_facts FROM threat_intel t "
                    "JOIN articles a ON a.id = t.article_id WHERE t.is_saudi = 1 AND t.content_type = ? "
                    "ORDER BY a.published_at DESC LIMIT ?", (ctype, examples))]

    pending = list_suggestions(con, "pending")
    multi = sum(1 for i in incident_list if i["article_count"] > 1)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "articles": {"stored_in_db": stored_articles, "saudi_fraud_articles": saudi_articles,
                     "by_content_type": c["by_content_type"]},
        "incidents": {
            "independent_total": c["incidents_total"], "confirmed": c["incidents_confirmed"],
            "reported": c["incidents_reported"], "needs_review": c["incidents_needs_review"],
            "multi_article": multi, "single_source": c["incidents_total"] - multi,
            "articles_behind_incidents": c["incident_articles_total"],
            "articles_per_incident": round(c["incident_articles_total"] / c["incidents_total"], 2) if c["incidents_total"] else None,
        },
        "separate_from_incidents": {"fraud_warnings": c["warnings_total"], "awareness_campaigns": c["awareness_total"]},
        "excluded_from_indicators": {"general_cybersecurity": c["excluded_general_cybersecurity"],
                                     "unrelated": c["excluded_unrelated"]},
        "unclassified": c["unclassified"],
        "pending_merge_suggestions": len(pending),
        "incident_list": incident_list,
        "pending_review": pending,
        "examples": {t: sample(t) for t in (fi.CT_WARNING, fi.CT_AWARENESS, fi.CT_GENERAL, fi.CT_UNRELATED)},
    }


def report_markdown(rep: Dict[str, Any]) -> str:
    a, i = rep["articles"], rep["incidents"]
    lines = [
        "# Incident Intelligence Report", "", f"_generated {rep['generated_at']}_", "",
        "| Metric | Value |", "|---|---|",
        f"| Articles stored in DB | {a['stored_in_db']} |",
        f"| Saudi fraud-engine articles | {a['saudi_fraud_articles']} |",
        f"| **Independent incidents** | **{i['independent_total']}** |",
        f"| - confirmed (enforcement/court) | {i['confirmed']} |",
        f"| - reported (details unconfirmed) | {i['reported']} |",
        f"| - needing human review | {i['needs_review']} |",
        f"| - covered by >1 article / single source | {i['multi_article']} / {i['single_source']} |",
        f"| Articles behind those incidents | {i['articles_behind_incidents']} |",
        f"| Fraud warnings (not incidents) | {rep['separate_from_incidents']['fraud_warnings']} |",
        f"| Awareness campaigns (not incidents) | {rep['separate_from_incidents']['awareness_campaigns']} |",
        f"| Excluded: general cybersecurity | {rep['excluded_from_indicators']['general_cybersecurity']} |",
        f"| Excluded: unrelated | {rep['excluded_from_indicators']['unrelated']} |",
        f"| Unclassified | {rep['unclassified']} |",
        f"| Pending merge suggestions | {rep['pending_merge_suggestions']} |", "",
        "## Incidents", "",
    ]
    for inc in rep["incident_list"]:
        lines += [f"### {inc['ref']} - {inc['title']}",
                  f"- type: {inc['record_type']} | review: {inc['review_status']} | category: {inc['category']} | "
                  f"locations: {', '.join(inc['locations']) or 'not stated'}",
                  f"- {inc['article_count']} article(s) from {inc['source_count']} source(s), "
                  f"{inc['first_reported_at']} -> {inc['last_reported_at']}"]
        for cv in inc["coverage"]:
            lines.append(f"  - [{cv['source']}]({cv['url']}) - {cv['title']} _(linked: {cv['why_linked']}"
                         f"{'; ' + ', '.join(cv['shared_facts']) if cv['shared_facts'] else ''})_")
        lines.append("")
    for t, items in rep["examples"].items():
        lines += [f"## Examples - {t}", ""]
        lines += [f"- [{x['source']}]({x['url']}) - {x['title']}" for x in items] or ["- none"]
        lines.append("")
    if rep["pending_review"]:
        lines += ["## Merge suggestions awaiting review", ""]
        for s in rep["pending_review"]:
            lines.append(f"- #{s['id']} score {s['score']}: {s['article_a'].get('headline')} <-> "
                         f"{s['article_b'].get('headline')} ({s['why_not_merged'].get('context')})")
    return "\n".join(lines)


__all__ = ["rebuild", "extract_signature", "compare", "cluster_members", "build_incident", "incident_to_dict",
           "counts", "build_patterns", "build_rules", "record_rule_test", "Signature", "MAX_GAP_DAYS",
           "list_suggestions", "decide_suggestion", "build_report", "report_markdown", "incident_ref"]
