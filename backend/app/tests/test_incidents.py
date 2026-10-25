# backend/app/tests/test_incidents.py
"""Regression tests for the Article -> Incident -> Fraud-Pattern layer.

The fixture ``fixtures/saudi_prod_items.json`` holds real articles that production stored on
2026-10-02..10-06 (bodies trimmed to 1500 chars): the Riyadh fraud-network arrest (8 outlets),
the Asir cybersecurity lecture, the «شيّك» campaign, a school-fees warning, etc.

Success metric under test: the number of INDEPENDENT incidents, not the number of articles.
"""
import json
import os
import re
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from backend.app import db as db_mod
from backend.app import fraud_intel as fi
from backend.app import incidents
from backend.app.intel import analyze_and_store

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "saudi_prod_items.json")
RIYADH_IDS = set(range(117, 125))      # 8 outlets covering ONE arrest
OKAZ_ID = 129                          # impersonation attempt reported by Okaz
SHIK_IDS = {126, 127}                  # «شيّك قبل تتفاعل» awareness campaign (Al Arabiya + another outlet)
LECTURE_ID = 113                       # Asir municipality cybersecurity lecture
TREND_ID = 115                         # "rising fraud crimes" trend piece (must NOT merge into the arrest)


def _words(s):
    """Normalised word sequence, so quotes compare equal regardless of punctuation."""
    return " ".join(re.findall(r"\w+", fi.norm(s)))


def load_items():
    with open(FIXTURE, encoding="utf-8") as fh:
        return {it["id"]: it for it in json.load(fh)}


def store_item(it, url=None, source=None, published_at=None, title=None):
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "source": source or it["source"], "domain": "", "country": "", "headline": title or it["title"],
        "content": it.get("excerpt") or "", "article_summary": it.get("summary") or "",
        "url": url or it["url"], "published_at": published_at or it["published_at"], "created_at": now,
        "category": None, "entity": None, "image_url": "", "content_hash": None, "fetched_at": now,
        "source_code": it.get("source_code") or "x", "language": "ar",
    }
    aid, _created = db_mod.insert_article_if_new(row)
    art = {**row, "article_id": aid, "saudi_scope": True, "saudi_publisher": True, "official": False}
    tid, why = analyze_and_store(art, require_saudi=True)
    assert tid, f"fixture {it['id']} not stored: {why}"
    return aid, tid


class IsolatedDB(unittest.TestCase):
    """Fresh SQLite file per test, loaded with the real production fixture."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        p = mock.patch.object(db_mod, "DB_PATH", os.path.join(self.tmp, "t.db"))
        p.start()
        self.addCleanup(p.stop)
        from backend.app import config
        p_cfg = mock.patch.object(config, "DB_PATH", os.path.join(self.tmp, "t.db"))
        p_cfg.start()
        self.addCleanup(p_cfg.stop)
        db_mod.init_db()
        self.items = load_items()
        self.aid = {}
        self.tid = {}
        for iid, it in self.items.items():
            self.aid[iid], self.tid[iid] = store_item(it)
        incidents.rebuild()

    # helpers
    def q(self, sql, *params):
        con = db_mod.get_db()
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def ctype(self, iid):
        return self.q("SELECT content_type FROM threat_intel WHERE article_id = ?", self.aid[iid])[0][0]

    def incident_rows(self):
        return self.q("SELECT * FROM incidents ORDER BY id")


# --------------------------------------------------------------------------- #
class RiyadhNetworkIsOneIncident(IsolatedDB):
    def test_eight_articles_collapse_into_one_incident(self):
        rows = self.incident_rows()
        riyadh = [r for r in rows if r["article_count"] == len(RIYADH_IDS)]
        self.assertEqual(len(riyadh), 1, "the Riyadh arrest must be exactly one incident")
        inc = incidents.incident_to_dict(riyadh[0], detail=True)
        self.assertEqual(inc["source_count"], len(RIYADH_IDS))

    def test_incident_links_every_covering_article(self):
        inc = [r for r in self.incident_rows() if r["article_count"] == len(RIYADH_IDS)][0]
        d = incidents.incident_to_dict(inc)
        got = {a["url"] for a in d["articles"]}
        want = {self.items[i]["url"] for i in RIYADH_IDS}
        self.assertEqual(got, want)
        linked = {r[0] for r in self.q("SELECT article_id FROM threat_intel WHERE incident_id = ?", inc["id"])}
        self.assertEqual(linked, {self.aid[i] for i in RIYADH_IDS})

    def test_facts_are_the_ones_stated_by_the_sources(self):
        inc = incidents.incident_to_dict(
            [r for r in self.incident_rows() if r["article_count"] == len(RIYADH_IDS)][0], detail=True)
        f = inc["facts"]
        self.assertEqual(f["persons_count"], 9)
        self.assertIn("الرياض", [x["label"] for x in f["locations"]])
        self.assertTrue({"yemeni", "sudanese"} <= {x["key"] for x in f["nationalities"]})
        self.assertIn(25965, [x["value"] for x in f["amounts"]])
        self.assertTrue({"gold", "phones", "bank_cards"} <= {x["key"] for x in f["seized_items"]})
        self.assertEqual(f["origin"], "stated_in_source")

    def test_nothing_in_the_incident_is_invented(self):
        """Every fact sentence, every method quote must literally occur in a source article."""
        inc = incidents.incident_to_dict(
            [r for r in self.incident_rows() if r["article_count"] == len(RIYADH_IDS)][0], detail=True)
        corpus = _words(" ".join(" ".join([self.items[i]["title"], self.items[i]["summary"] or "",
                                           self.items[i]["excerpt"] or ""]) for i in RIYADH_IDS))
        for s in inc["facts"]["fact_sentences"]:
            self.assertIn(_words(s["text"]), corpus)
        self.assertTrue(inc["methods"], "stated methods expected")
        for m in inc["methods"]:
            self.assertTrue(m["quotes"])
            for q in m["quotes"]:
                self.assertIn(_words(q["quote"]), corpus)
        for ind in inc["indicators"]["mentioned_in_source"]:
            self.assertEqual(ind["origin"], "stated_in_source")
            self.assertIn(_words(ind["quote"]), corpus)
        self.assertEqual(inc["indicators"]["suggested_by_system"]["origin"], "system_template")

    def test_trend_article_does_not_merge_into_the_arrest(self):
        self.assertIsNone(self.q("SELECT incident_id FROM threat_intel WHERE article_id = ?", self.aid[TREND_ID])[0][0])
        self.assertEqual(self.ctype(TREND_ID), fi.CT_WARNING)

    def test_dedup_is_not_based_on_headline(self):
        """Same facts under a completely different headline/outlet still joins the same incident;
        an almost identical headline with different facts (city, count, date) does not."""
        base = self.items[117]
        before = [r for r in self.incident_rows() if r["article_count"] == len(RIYADH_IDS)][0]["id"]
        # (a) rewritten headline, new outlet, new URL, same facts + same days -> merges
        store_item(base, url="https://example-news.test/a1", source="موقع تجريبي",
                   title="خبر مختلف تماماً في الصياغة عن الموضوع نفسه")
        # (b) near-identical headline, but another city, another suspect count, a month later -> separate
        other = dict(base)
        other["excerpt"] = ("أوقفت شرطة جدة 3 مقيمين من الجنسية المصرية بتهمة الاحتيال المالي عبر بطاقات مصرفية "
                            "وضبط مبلغ 12,000 ريال.")
        other["summary"] = "شرطة جدة توقف 3 مقيمين بتهمة الاحتيال المالي"
        store_item(other, url="https://example-news.test/b1", source="موقع آخر", title=base["title"],
                   published_at="2026-11-20T08:00:00+00:00")
        incidents.rebuild()
        rows = self.incident_rows()
        riyadh = [r for r in rows if r["id"] == before]
        self.assertEqual(len(riyadh), 1, "incident id must be stable")
        self.assertEqual(riyadh[0]["article_count"], len(RIYADH_IDS) + 1)
        self.assertEqual(len(rows), 3, "Riyadh + Okaz + the unrelated Jeddah case")

    def test_same_facts_far_apart_in_time_are_different_incidents(self):
        base = self.items[117]
        store_item(base, url="https://example-news.test/late", source="موقع متأخر",
                   published_at="2026-12-30T08:00:00+00:00")
        incidents.rebuild()
        counts = sorted(r["article_count"] for r in self.incident_rows())
        self.assertEqual(counts, [1, 1, len(RIYADH_IDS)])


# --------------------------------------------------------------------------- #
class CybersecurityLectureIsExcluded(IsolatedDB):
    def test_classified_as_general_cybersecurity(self):
        self.assertEqual(self.ctype(LECTURE_ID), fi.CT_GENERAL)

    def test_stored_but_not_an_incident_nor_in_any_indicator(self):
        self.assertEqual(self.q("SELECT COUNT(*) FROM articles WHERE id = ?", self.aid[LECTURE_ID])[0][0], 1)
        self.assertIsNone(self.q("SELECT incident_id FROM threat_intel WHERE article_id = ?", self.aid[LECTURE_ID])[0][0])
        c = incidents.counts(db_mod.get_db())
        self.assertEqual(c["excluded_general_cybersecurity"], 1)
        titles = [a["title"] for r in self.incident_rows() for a in incidents.incident_to_dict(r)["articles"]]
        self.assertNotIn(self.items[LECTURE_ID]["title"], titles)

    def test_hidden_from_default_listing_but_reachable_explicitly(self):
        client = _client()
        default = client.get("/saudi-fraud/news", params={"page_size": 100}).json()
        self.assertNotIn(LECTURE_ID, [i["article_id"] and self._iid(i["article_id"]) for i in default["items"]])
        self.assertTrue(all(i["content_type"] in fi.VISIBLE_CONTENT_TYPES for i in default["items"]))
        explicit = client.get("/saudi-fraud/news", params={"content_type": "general_cybersecurity"}).json()
        self.assertEqual([i["article_id"] for i in explicit["items"]], [self.aid[LECTURE_ID]])

    def _iid(self, article_id):
        return next((k for k, v in self.aid.items() if v == article_id), None)


# --------------------------------------------------------------------------- #
class ShikCampaignIsAwarenessNotIncident(IsolatedDB):
    def test_classified_as_awareness_campaign(self):
        for i in SHIK_IDS:
            self.assertEqual(self.ctype(i), fi.CT_AWARENESS, self.items[i]["title"])

    def test_not_counted_as_confirmed_incident(self):
        c = incidents.counts(db_mod.get_db())
        self.assertEqual(c["awareness_total"], 4)  # شيّك x2 + Meta interview + NCA school campaign
        for i in SHIK_IDS:
            self.assertIsNone(self.q("SELECT incident_id FROM threat_intel WHERE article_id = ?", self.aid[i])[0][0])
        urls = {a["url"] for r in self.incident_rows() for a in incidents.incident_to_dict(r)["articles"]}
        for i in SHIK_IDS:
            self.assertNotIn(self.items[i]["url"], urls)

    def test_listed_in_its_own_section(self):
        client = _client()
        r = client.get("/saudi-fraud/news", params={"content_type": "awareness_campaign", "page_size": 50}).json()
        self.assertEqual(r["total"], 4)
        self.assertTrue(all(i["content_type"] == "awareness_campaign" for i in r["items"]))


# --------------------------------------------------------------------------- #
class WarningsAndHeadlineNumbers(IsolatedDB):
    def test_warnings_are_separate_from_incidents(self):
        for i in (115, 116, 125):
            self.assertEqual(self.ctype(i), fi.CT_WARNING, self.items[i]["title"])
        c = incidents.counts(db_mod.get_db())
        self.assertEqual(c["warnings_total"], 3)

    def test_success_metric_is_incidents_not_articles(self):
        ov = _client().get("/saudi-fraud/overview").json()
        self.assertEqual(ov["incidents_total"], 2)            # Riyadh network + Okaz impersonation
        self.assertEqual(ov["incident_articles_total"], 9)    # 8 + 1 articles behind those 2 incidents
        self.assertEqual(ov["warnings_total"], 3)
        self.assertEqual(ov["awareness_total"], 4)
        self.assertEqual(ov["excluded_general_cybersecurity"], 1)
        self.assertEqual(ov["total_items"], 16)               # 17 stored - 1 excluded
        self.assertEqual(sum(c["count"] for c in ov["categories"]), 2, "categories count incidents")
        self.assertEqual(len(self.items), 17)
        self.assertEqual(self.q("SELECT COUNT(*) FROM articles")[0][0], 17, "no article is ever removed")

    def test_overview_keeps_legacy_fields(self):
        ov = _client().get("/saudi-fraud/overview").json()
        for k in ("total_items", "last_7_days", "high_bank_relevance", "distinct_sources", "categories",
                  "last_fetch_at", "sources_ok", "sources_total", "auto_ingest_enabled", "ingest_running"):
            self.assertIn(k, ov)


# --------------------------------------------------------------------------- #
class PatternsAndRules(IsolatedDB):
    def _corpus(self):
        return _words(" ".join(" ".join([i["title"], i["summary"] or "", i["excerpt"] or ""])
                               for i in self.items.values()))

    def test_patterns_are_stated_methods_with_evidence(self):
        pats = _client().get("/saudi-fraud/patterns").json()["patterns"]
        self.assertTrue(pats)
        corpus = self._corpus()
        for p in pats:
            self.assertEqual(p["origin"], "stated_in_source")
            self.assertTrue(p["evidence"], p["key"])
            for e in p["evidence"]:
                self.assertIn(_words(e["quote"]), corpus)
                self.assertTrue(e["url"])
            self.assertEqual(p["incident_count"], len(p["incidents"]))
        top = {p["key"] for p in pats if p["incident_count"] >= 1}
        self.assertTrue({"victim_data_misuse", "online_purchase", "illicit_transfer"} <= top)

    def test_rules_link_to_evidence_and_are_untested_by_default(self):
        body = _client().get("/saudi-fraud/rules").json()
        self.assertTrue(body["rules"])
        for r in body["rules"]:
            self.assertEqual(r["status"], "proposed_untested")
            self.assertTrue(r["is_inference"])
            self.assertEqual(r["test_results"], [])
            ev = r["evidence"]
            self.assertTrue(ev["incident_ids"] or ev["warning_articles"], r["rule_id"])
            for inc in ev["incidents"]:
                self.assertTrue(inc["article_urls"])
        self.assertEqual(body["summary"]["tested"], 0)
        fin = next(r for r in body["rules"] if r["category"] == "financial_fraud")
        self.assertTrue(fin["evidence"]["incident_ids"])
        self.assertTrue(fin["evidence"]["patterns"])

    def test_rule_becomes_tested_only_after_a_real_test_is_recorded(self):
        rid = next(r for r in _client().get("/saudi-fraud/rules").json()["rules"]
                   if r["category"] == "financial_fraud")["rule_id"]
        incidents.record_rule_test(rid, dataset="backtest-2026Q3", precision=0.42, recall=0.7, alerts=123,
                                   notes="unit-test row", tester="pytest")
        body = _client().get("/saudi-fraud/rules").json()
        rule = next(r for r in body["rules"] if r["rule_id"] == rid)
        self.assertEqual(rule["status"], "tested")
        self.assertEqual(rule["test_results"][0]["dataset"], "backtest-2026Q3")
        self.assertEqual(body["summary"]["tested"], 1)
        others = [r for r in body["rules"] if r["rule_id"] != rid]
        self.assertTrue(all(r["status"] == "proposed_untested" for r in others))


# --------------------------------------------------------------------------- #
class RebuildAndUpgradeSafety(IsolatedDB):
    def test_rebuild_is_idempotent_and_ids_are_stable(self):
        ids = [r["id"] for r in self.incident_rows()]
        incidents.rebuild()
        incidents.rebuild()
        self.assertEqual([r["id"] for r in self.incident_rows()], ids)

    def test_upgrade_of_an_existing_db_keeps_every_article(self):
        """Simulate rows written by the previous release (no content_type / incident_id)."""
        before = self.q("SELECT id, url, headline, content FROM articles ORDER BY id")
        con = db_mod.get_db()
        con.execute("UPDATE threat_intel SET content_type = NULL, incident_id = NULL")
        con.execute("DELETE FROM incidents")
        con.commit()
        con.close()
        from backend.app import saudi_ingest
        res = saudi_ingest.backfill_incidents()
        self.assertEqual(res["counts"]["incidents_total"], 2)
        self.assertEqual(res["counts"]["unclassified"], 0)
        self.assertEqual(res["demoted"], 0)
        after = self.q("SELECT id, url, headline, content FROM articles ORDER BY id")
        self.assertEqual([tuple(r) for r in before], [tuple(r) for r in after])
        # second run is a no-op
        res2 = saudi_ingest.backfill_incidents()
        self.assertEqual(res2["checked"], 0)
        self.assertEqual(res2["counts"]["incidents_total"], 2)

    def test_init_db_is_additive_on_an_old_schema_table(self):
        path = os.path.join(self.tmp, "old.db")
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE threat_intel (id INTEGER PRIMARY KEY AUTOINCREMENT, article_id INTEGER UNIQUE, "
                    "title TEXT, is_saudi INTEGER)")
        con.execute("INSERT INTO threat_intel (article_id, title, is_saudi) VALUES (1, 'kept', 1)")
        con.commit()
        con.close()
        with mock.patch.object(db_mod, "DB_PATH", path):
            db_mod.init_db()
            c = sqlite3.connect(path)
            cols = {r[1] for r in c.execute("PRAGMA table_info(threat_intel)")}
            self.assertTrue({"content_type", "incident_id"} <= cols)
            self.assertEqual(c.execute("SELECT title FROM threat_intel").fetchone()[0], "kept")
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"incidents", "rule_tests"} <= tables)
            c.close()

    def test_backup_is_taken_before_first_migration_only(self):
        path = os.path.join(self.tmp, "legacy", "legacy.db")
        os.makedirs(os.path.dirname(path))
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, domain TEXT, "
                    "country TEXT, headline TEXT, content TEXT, article_summary TEXT, url TEXT UNIQUE, "
                    "published_at TEXT, created_at TEXT)")
        con.executemany("INSERT INTO articles (url, headline) VALUES (?, ?)",
                        [("https://e.test/%d" % i, "h%d" % i) for i in range(5)])
        con.commit()
        con.close()
        bdir = os.path.join(os.path.dirname(path), "backups")
        with mock.patch.object(db_mod, "DB_PATH", path):
            self.assertTrue(db_mod._needs_incident_migration())
            db_mod.init_db()
            backups = os.listdir(bdir)
            self.assertEqual(len(backups), 1)
            b = sqlite3.connect(os.path.join(bdir, backups[0]))
            self.assertEqual(b.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 5)
            self.assertFalse(b.execute("SELECT name FROM sqlite_master WHERE name='incidents'").fetchall())
            b.close()
            self.assertFalse(db_mod._needs_incident_migration())
            db_mod.init_db()
            self.assertEqual(len(os.listdir(bdir)), 1)
            live = sqlite3.connect(path)
            self.assertEqual(live.execute("SELECT COUNT(*) FROM articles").fetchone()[0], 5)
            live.close()

    def test_reclassify_never_deletes_rows(self):
        from backend.app import saudi_ingest
        n_t = self.q("SELECT COUNT(*) FROM threat_intel")[0][0]
        n_a = self.q("SELECT COUNT(*) FROM articles")[0][0]
        saudi_ingest.reclassify_stored()
        self.assertEqual(self.q("SELECT COUNT(*) FROM threat_intel")[0][0], n_t)
        self.assertEqual(self.q("SELECT COUNT(*) FROM articles")[0][0], n_a)


# --------------------------------------------------------------------------- #
class ContentTypeUnit(unittest.TestCase):
    def _ct(self, title, summary="", body=""):
        a, why = fi.analyze(title, summary, body, source={"saudi_publisher": True, "saudi_scope": True},
                            require_saudi=False)
        self.assertIsNotNone(a, why)
        return a["content_type"]

    def test_arrest_is_confirmed(self):
        self.assertEqual(self._ct("القبض على 5 مقيمين بتهمة الاحتيال المالي في جدة",
                                  "أوقفت شرطة جدة 5 مقيمين بتهمة الاحتيال المالي وانتحال صفة موظفي البنوك"),
                         fi.CT_CONFIRMED)

    def test_campaign_in_title_is_awareness(self):
        self.assertEqual(self._ct("حملة توعوية للتحذير من الاحتيال المالي والروابط المشبوهة",
                                  "تطلق جهة حكومية حملة توعوية للتحذير من الاحتيال الإلكتروني"),
                         fi.CT_AWARENESS)

    def test_warning_without_a_reported_event(self):
        self.assertEqual(self._ct("تحذير من رسائل احتيالية تنتحل صفة البنوك",
                                  "حذّر خبراء من رسائل احتيال مالي تطلب رمز التحقق"), fi.CT_WARNING)

    def test_all_six_types_are_declared(self):
        self.assertEqual(set(fi.CONTENT_TYPES), {fi.CT_CONFIRMED, fi.CT_REPORTED, fi.CT_WARNING, fi.CT_AWARENESS,
                                                 fi.CT_GENERAL, fi.CT_UNRELATED})
        self.assertEqual(set(fi.VISIBLE_CONTENT_TYPES), {fi.CT_CONFIRMED, fi.CT_REPORTED, fi.CT_WARNING, fi.CT_AWARENESS})
        self.assertEqual(set(fi.INCIDENT_CONTENT_TYPES), {fi.CT_CONFIRMED, fi.CT_REPORTED})

    def test_described_occurrence_without_enforcement_is_reported_not_confirmed(self):
        it = load_items()[129]
        self.assertEqual(self._ct(it["title"], it["summary"] or "", it["excerpt"] or ""), fi.CT_REPORTED)

    def test_arrest_phrasing_variants_are_confirmed(self):
        for title in ("أوقفت شرطة جدة 3 مقيمين بتهمة الاحتيال المالي",
                      "شرطة جدة توقف 3 مقيمين بتهمة الاحتيال المالي"):
            self.assertEqual(self._ct(title, "احتيال مالي عبر بطاقات مصرفية"), fi.CT_CONFIRMED, title)


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.app import saudi_api

    app = FastAPI()
    app.include_router(saudi_api.router)
    return TestClient(app)


if __name__ == "__main__":
    unittest.main()
