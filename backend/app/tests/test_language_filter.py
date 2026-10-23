"""Arabic-only language filter: detection, non-destructive backfill and the ?lang= API filter."""
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from backend.app import db as db_mod


class DetectLanguage(unittest.TestCase):
    def test_arabic_english_and_mixed(self):
        self.assertEqual(db_mod.detect_language("شرطة الرياض تطيح بشبكة احتيال", ""), "ar")
        self.assertEqual(db_mod.detect_language("Police arrest fraud network in Riyadh", ""), "en")
        # an Arabic news headline that quotes a Latin brand is still Arabic
        self.assertEqual(db_mod.detect_language("السعودية و Meta تطلقان مبادرة لمواجهة الاحتيال الإلكتروني", ""), "ar")
        self.assertEqual(db_mod.detect_language("", None), "en")


class LangFilterApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "t.db")
        p = mock.patch.object(db_mod, "DB_PATH", self.path)
        p.start()
        self.addCleanup(p.stop)
        # legacy rows: no language column value at all
        con = sqlite3.connect(self.path)
        con.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, domain TEXT, "
                    "country TEXT, headline TEXT, content TEXT, article_summary TEXT, url TEXT UNIQUE, "
                    "published_at TEXT, created_at TEXT)")
        rows = [
            ("bbc", "bbc.com", "UK", "Police arrest fraud network", "body en", "sum en", "https://e/1", "2026-10-01"),
            ("okaz", "okaz.com.sa", "SA", "شرطة الرياض تطيح بشبكة احتيال", "نص عربي", "ملخص عربي", "https://e/2", "2026-10-02"),
            ("cnn", "cnn.com", "US", "Markets rally", "body", "sum", "https://e/3", "2026-10-03"),
        ]
        con.executemany("INSERT INTO articles (source, domain, country, headline, content, article_summary, url, "
                        "published_at) VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        self.before = self._snapshot()
        db_mod.init_db()

        from fastapi.testclient import TestClient
        from backend.app import main
        self.client = TestClient(main.app)

    def _snapshot(self):
        con = sqlite3.connect(self.path)
        try:
            return con.execute("SELECT id, url, headline, content, article_summary FROM articles ORDER BY id").fetchall()
        finally:
            con.close()

    def test_backfill_tags_rows_without_touching_text(self):
        self.assertEqual(self._snapshot(), self.before)
        con = sqlite3.connect(self.path)
        langs = dict(con.execute("SELECT id, language FROM articles").fetchall())
        con.close()
        self.assertEqual(langs, {1: "en", 2: "ar", 3: "en"})

    def test_a_backup_precedes_the_tagging(self):
        bdir = os.path.join(self.tmp, "backups")
        self.assertTrue(any(f.startswith("metaview-pre-") for f in os.listdir(bdir)))

    def test_articles_and_search_filter_by_lang(self):
        ar = self.client.get("/articles?lang=ar").json()
        self.assertEqual([r["url"] for r in ar["results"]], ["https://e/2"])
        self.assertEqual(ar["total_rows"], 1)
        allr = self.client.get("/articles").json()
        self.assertEqual(allr["total_rows"], 3)
        s = self.client.get("/search-text?lang=ar&q=احتيال").json()
        self.assertEqual(s["total_rows"], 1)
        self.assertEqual(self.client.get("/search-text?lang=en&q=احتيال").json()["total_rows"], 0)

    def test_stats_and_countries_respect_lang(self):
        self.assertEqual(self.client.get("/stats?lang=ar").json()["total_articles"], 1)
        self.assertEqual(self.client.get("/stats").json()["total_articles"], 3)
        self.assertEqual(self.client.get("/countries?lang=ar").json(), [{"country": "SA", "count": 1}])

    def test_invalid_lang_is_rejected(self):
        self.assertEqual(self.client.get("/articles?lang=xx").status_code, 422)


if __name__ == "__main__":
    unittest.main()
