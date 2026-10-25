# backend/app/db.py
import re
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from .config import DB_PATH

def get_db() -> sqlite3.Connection:
    # timeout: the Saudi scheduler thread and API requests may write concurrently
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    con.row_factory = sqlite3.Row
    return con

def _ensure_column(con: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r["name"] for r in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

BACKUP_KEEP = 3

_AR_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_ANY_LETTER = re.compile(r"[^\W\d_]")


def detect_language(*texts: Optional[str]) -> str:
    """'ar' when Arabic letters dominate the given texts (> 40% of all letters), else 'en'.
    Used only to tag articles; it never alters their text."""
    joined = " ".join(t for t in texts if t)
    letters = len(_ANY_LETTER.findall(joined))
    if not letters:
        return "en"
    return "ar" if len(_AR_LETTER.findall(joined)) / letters > 0.4 else "en"


def backup_database(tag: str = "manual") -> Optional[str]:
    """Consistent copy of the live database (sqlite online-backup API) into <db dir>/backups/."""
    import logging
    import os
    from datetime import datetime, timezone
    try:
        if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
            return None
        bdir = os.path.join(os.path.dirname(os.path.abspath(DB_PATH)), "backups")
        os.makedirs(bdir, exist_ok=True)
        dest = os.path.join(bdir, f"metaview-{tag}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.db")
        src = sqlite3.connect(DB_PATH, timeout=30)
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        old = sorted(f for f in os.listdir(bdir) if f.startswith(f"metaview-{tag}-") and f.endswith(".db"))
        for f in old[:-BACKUP_KEEP]:
            os.remove(os.path.join(bdir, f))
        logging.getLogger("metaview.db").info("database backup written: %s", dest)
        return dest
    except Exception:  # pragma: no cover - backup must never block startup
        logging.getLogger("metaview.db").exception("database backup failed")
        return None


def _needs_incident_migration() -> bool:
    """True when an existing, non-empty DB predates the incident layer (no ``incidents`` table, or a
    threat_intel table without ``content_type``). A brand-new/empty DB needs no backup."""
    import os
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return False
    try:
        con = sqlite3.connect(DB_PATH, timeout=30)
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not tables & {"articles", "threat_intel"}:
                return False
            if "incidents" not in tables:
                return True
            cols = [r[1] for r in con.execute("PRAGMA table_info(threat_intel)")]
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return bool(cols) and "content_type" not in cols


def _needs_language_tag() -> bool:
    """True when an existing DB has articles whose ``language`` is still untagged."""
    import os
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        return False
    try:
        con = sqlite3.connect(DB_PATH, timeout=30)
        try:
            cols = [r[1] for r in con.execute("PRAGMA table_info(articles)")]
            if "language" not in cols:
                return bool(con.execute("SELECT 1 FROM articles LIMIT 1").fetchone())
            return bool(con.execute(
                "SELECT 1 FROM articles WHERE language IS NULL OR language = '' LIMIT 1").fetchone())
        finally:
            con.close()
    except sqlite3.Error:
        return False


def init_db() -> None:
    if _needs_incident_migration():
        backup_database("pre-incident-layer")
    elif _needs_language_tag():
        backup_database("pre-language-tag")
    con = get_db()
    cur = con.cursor()
    # Use executescript for multiple CREATE statements
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            domain TEXT,
            country TEXT,
            headline TEXT,
            content TEXT,
            article_summary TEXT,
            url TEXT UNIQUE,
            published_at TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS threat_intel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            threat_type TEXT NOT NULL,
            event_type TEXT,
            evidence_confidence REAL,
            analysis_confidence REAL,
            verified_facts TEXT,
            analytical_assessment TEXT,
            detection_indicators TEXT,
            preventive_recommendations TEXT,
            evidence_urls TEXT,
            confidence REAL,
            original_url TEXT,
            related_links TEXT,
            is_saudi INTEGER,
            is_financial INTEGER,
            created_at TEXT,
            updated_at TEXT,
            FOREIGN KEY(article_id) REFERENCES articles(id),
            UNIQUE(article_id)
        );

        CREATE TABLE IF NOT EXISTS source_trust (
            source_id INTEGER PRIMARY KEY,
            trust_score REAL,
            last_checked TEXT,
            notes TEXT,
            FOREIGN KEY(source_id) REFERENCES articles(id)
        );

        CREATE TABLE IF NOT EXISTS cve_reference (
            cve_id TEXT PRIMARY KEY,
            description TEXT,
            severity TEXT,
            published_at TEXT,
            reference_url TEXT
        );

        CREATE TABLE IF NOT EXISTS incident_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            threat_id INTEGER,
            related_threat_id INTEGER,
            relation_type TEXT,
            FOREIGN KEY(threat_id) REFERENCES threat_intel(id),
            FOREIGN KEY(related_threat_id) REFERENCES threat_intel(id)
        );

        -- Saudi Fraud Intelligence: registry of ingestion sources and their health
        CREATE TABLE IF NOT EXISTS source_registry (
            code TEXT PRIMARY KEY,
            name TEXT,
            url TEXT,
            kind TEXT,               -- rss | html | aggregator
            country TEXT,
            lang TEXT,
            official INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            last_status TEXT,        -- ok | failed | empty
            last_success_at TEXT,
            last_failure_at TEXT,
            last_error TEXT,
            last_entries INTEGER DEFAULT 0,
            consecutive_failures INTEGER DEFAULT 0
        );

        -- One row per source per ingestion run (success AND failure)
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            source_code TEXT,
            started_at TEXT,
            finished_at TEXT,
            ok INTEGER,
            http_status INTEGER,
            entries_found INTEGER DEFAULT 0,
            new_articles INTEGER DEFAULT 0,
            duplicates INTEGER DEFAULT 0,
            excluded INTEGER DEFAULT 0,
            fraud_matched INTEGER DEFAULT 0,
            error TEXT
        );

        -- Every raw link ever evaluated by the Saudi pipeline (stored OR excluded).
        -- Lets us skip re-processing, count duplicates exactly and audit exclusions
        -- without storing non-fraud article bodies.
        CREATE TABLE IF NOT EXISTS ingest_seen (
            raw_link TEXT PRIMARY KEY,
            canonical_url TEXT,
            source_code TEXT,
            decision TEXT,           -- stored | excluded | duplicate
            reason TEXT,
            article_id INTEGER,
            first_seen_at TEXT,
            last_seen_at TEXT
        );

        -- Cross-process mutual exclusion for the Saudi ingestion (one row, expiring lease)
        CREATE TABLE IF NOT EXISTS ingest_lease (
            name TEXT PRIMARY KEY,
            holder TEXT,
            acquired_at TEXT,
            expires_at TEXT
        );
        """
    )
    try:
        cur.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass


    # Ensure additional columns safely
    _ensure_column(con, "articles", "category", "TEXT")
    _ensure_column(con, "articles", "entity", "TEXT")
    _ensure_column(con, "articles", "image_url", "TEXT")
    # Saudi ingestion: content fingerprint for de-duplication + fetch timestamp
    _ensure_column(con, "articles", "content_hash", "TEXT")
    _ensure_column(con, "articles", "fetched_at", "TEXT")
    _ensure_column(con, "articles", "source_code", "TEXT")
    _ensure_column(con, "articles", "language", "TEXT")

    # Source registry extra metadata
    _ensure_column(con, "source_registry", "handle", "TEXT")
    _ensure_column(con, "source_registry", "notes", "TEXT")

    # Fraud intelligence fields on the existing threat_intel table
    _ensure_column(con, "threat_intel", "fraud_category", "TEXT")
    _ensure_column(con, "threat_intel", "fraud_category_ar", "TEXT")
    _ensure_column(con, "threat_intel", "modus_operandi", "TEXT")
    _ensure_column(con, "threat_intel", "channels", "TEXT")
    _ensure_column(con, "threat_intel", "target_group", "TEXT")
    _ensure_column(con, "threat_intel", "bank_relevance", "TEXT")
    _ensure_column(con, "threat_intel", "detection_rule", "TEXT")
    _ensure_column(con, "threat_intel", "required_data", "TEXT")
    _ensure_column(con, "threat_intel", "matched_terms", "TEXT")
    # Incident layer: content type per article + link to the (derived) incident it belongs to
    _ensure_column(con, "threat_intel", "content_type", "TEXT")
    _ensure_column(con, "threat_intel", "incident_id", "INTEGER")

    # Incidents are DERIVED from articles (rebuildable); they never replace or delete articles.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, summary TEXT, fraud_category TEXT, fraud_category_ar TEXT,
            first_reported_at TEXT, last_reported_at TEXT,
            article_count INTEGER, source_count INTEGER,
            bank_relevance TEXT, confidence REAL, evidence_confidence REAL,
            locations TEXT, agencies TEXT, signature TEXT, facts TEXT, methods TEXT,
            channels TEXT, indicators TEXT, articles TEXT, match_basis TEXT,
            record_type TEXT, review_status TEXT, pending_review INTEGER DEFAULT 0,
            created_at TEXT, updated_at TEXT
        )
        """
    )
    _ensure_column(con, "incidents", "record_type", "TEXT")
    _ensure_column(con, "incidents", "review_status", "TEXT")
    _ensure_column(con, "incidents", "pending_review", "INTEGER DEFAULT 0")

    # Intelligence Enrichment columns for Fraud Intelligence Dashboard
    _ensure_column(con, "incidents", "fraud_type", "TEXT")
    _ensure_column(con, "incidents", "fraud_type_ar", "TEXT")
    _ensure_column(con, "incidents", "fraud_type_en", "TEXT")
    _ensure_column(con, "incidents", "secondary_fraud_types", "TEXT")
    _ensure_column(con, "incidents", "incident_country", "TEXT")
    _ensure_column(con, "incidents", "incident_city", "TEXT")
    _ensure_column(con, "incidents", "incident_region", "TEXT")
    _ensure_column(con, "incidents", "victim_location", "TEXT")
    _ensure_column(con, "incidents", "suspect_location", "TEXT")
    _ensure_column(con, "incidents", "channel", "TEXT")
    _ensure_column(con, "incidents", "channel_ar", "TEXT")
    _ensure_column(con, "incidents", "channels_list", "TEXT")
    _ensure_column(con, "incidents", "targeted_sector", "TEXT")
    _ensure_column(con, "incidents", "targeted_sector_ar", "TEXT")
    _ensure_column(con, "incidents", "targeted_sectors", "TEXT")
    _ensure_column(con, "incidents", "impersonated_entity", "TEXT")
    _ensure_column(con, "incidents", "reporting_agency", "TEXT")
    _ensure_column(con, "incidents", "modus_operandi_detailed", "TEXT")
    _ensure_column(con, "incidents", "financial_loss_amount", "REAL")
    _ensure_column(con, "incidents", "financial_loss_currency", "TEXT")
    _ensure_column(con, "incidents", "financial_loss_raw", "TEXT")
    _ensure_column(con, "incidents", "occurred_at", "TEXT")
    _ensure_column(con, "incidents", "evidence_text", "TEXT")
    _ensure_column(con, "incidents", "evidence_sources", "TEXT")
    _ensure_column(con, "incidents", "enrichment_confidence", "REAL")
    _ensure_column(con, "incidents", "is_emerging_threat", "INTEGER DEFAULT 0")
    _ensure_column(con, "incidents", "emerging_rationale", "TEXT")

    # Safe migration: audit log table for all intelligence enrichment & rebuilding operations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS enrichment_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            articles_evaluated INTEGER DEFAULT 0,
            incidents_created INTEGER DEFAULT 0,
            incidents_updated INTEGER DEFAULT 0,
            details TEXT,
            executed_by TEXT DEFAULT 'system',
            created_at TEXT NOT NULL
        )
        """
    )
    # Coverage link: which articles back which incident, and WHY they were linked (articles are never merged).
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS incident_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER NOT NULL, article_id INTEGER NOT NULL UNIQUE, threat_id INTEGER,
            role TEXT, content_type TEXT, match_basis TEXT, published_at TEXT, linked_at TEXT
        )
        """
    )
    # Uncertain merges are NOT applied: they wait here for a human decision, which is then kept forever.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS merge_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_a INTEGER NOT NULL, article_b INTEGER NOT NULL,
            score REAL, shared TEXT, conflicts TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            reviewer TEXT, reason TEXT, created_at TEXT, decided_at TEXT,
            UNIQUE(article_a, article_b)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_incident_articles_inc ON incident_articles(incident_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_merge_suggestions_status ON merge_suggestions(status)")
    # Results of REAL rule backtests / shadow runs. A rule is 'tested' only if it has rows here.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id TEXT NOT NULL, tested_at TEXT, dataset TEXT,
            precision_ REAL, recall REAL, alerts INTEGER, notes TEXT, tester TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_threat_incident ON threat_intel(incident_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_threat_content_type ON threat_intel(content_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rule_tests_rule ON rule_tests(rule_id)")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_country ON articles(country)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_entity ON articles(entity)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(content_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fetch_log_source ON fetch_log(source_code, started_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_threat_saudi ON threat_intel(is_saudi, fraud_category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_language ON articles(language)")

    # Derived metadata only (never touches headline/content/url): tag rows that have no language yet.
    untagged = cur.execute(
        "SELECT id, headline, article_summary FROM articles WHERE language IS NULL OR language = ''"
    ).fetchall()
    if untagged:
        cur.executemany(
            "UPDATE articles SET language = ? WHERE id = ?",
            [(detect_language(r["headline"], r["article_summary"]), r["id"]) for r in untagged],
        )

    seed_source_registry(con)

    con.commit()
    con.close()


def seed_source_registry(con: sqlite3.Connection) -> None:
    """Populate source_registry with Saudi sources and X social sources."""
    cur = con.cursor()
    # 1. Saudi sources
    try:
        from .saudi_sources import SAUDI_SOURCES
        for s in SAUDI_SOURCES:
            cur.execute(
                """
                INSERT INTO source_registry (code, name, url, kind, country, lang, official, enabled, notes)
                VALUES (?, ?, ?, ?, 'SA', ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    name=excluded.name, url=excluded.url, kind=excluded.kind,
                    lang=excluded.lang, official=excluded.official, notes=excluded.notes
                """,
                (s.code, s.name, s.url, s.kind, s.lang, int(s.official), int(s.enabled), s.notes),
            )
    except Exception:
        pass

    # 2. X social accounts
    try:
        from .x_sources import X_SOURCES, get_x_bearer_token
        has_token = bool(get_x_bearer_token())
        for x in X_SOURCES:
            initial_status = "requires_integration" if (not has_token and not x.publisher_feed) else "ok"
            initial_error = (
                "يتطلب تكامل X API (Requires Integration - X_BEARER_TOKEN) للوصول المصرح إلى المنشورات."
                if (not has_token and not x.publisher_feed)
                else None
            )
            cur.execute("SELECT code, last_status FROM source_registry WHERE code = ?", (x.code,))
            existing = cur.fetchone()
            if not existing:
                cur.execute(
                    """
                    INSERT INTO source_registry (
                        code, name, url, kind, country, lang, official, enabled, handle, notes, last_status, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (x.code, x.name, x.url, x.kind, x.country, x.lang, int(x.official), x.handle, x.notes, initial_status, initial_error),
                )
            else:
                cur.execute(
                    """
                    UPDATE source_registry
                    SET name = ?, url = ?, kind = ?, country = ?, lang = ?, official = ?, handle = ?, notes = ?
                    WHERE code = ?
                    """,
                    (x.name, x.url, x.kind, x.country, x.lang, int(x.official), x.handle, x.notes, x.code),
                )
    except Exception:
        pass

    # Link existing articles to source_code if not yet set
    try:
        cur.execute(
            """
            UPDATE articles
            SET source_code = 'x_thehackersnews'
            WHERE (source_code IS NULL OR source_code = '')
              AND (domain = 'thehackernews.com' OR url LIKE '%thehackernews%')
            """
        )
    except Exception:
        pass


# Auto-initialize database on import
init_db()

def row_count() -> int:
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM articles")
    c = int(cur.fetchone()["c"])
    con.close()
    return c

def upsert_article(item: Dict[str, Any]) -> Optional[int]:
    """Insert article if new and return its id.
    Returns the newly inserted row id, or the existing article id if the URL
    already exists. Returns None on failure.
    """
    con = get_db()
    cur = con.cursor()
    try:
        cur.execute(
            """
            INSERT INTO articles (
                source, domain, country, headline, content, article_summary, url,
                published_at, created_at, category, entity, image_url,
                content_hash, fetched_at, source_code, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.get("source", ""),
                item.get("domain", ""),
                item.get("country", ""),
                item.get("headline", ""),
                item.get("content", ""),
                item.get("article_summary", ""),
                item.get("url", ""),
                item.get("published_at", ""),
                item.get("created_at", ""),
                item.get("category", ""),
                item.get("entity", None),
                item.get("image_url", ""),
                item.get("content_hash"),
                item.get("fetched_at"),
                item.get("source_code"),
                item.get("language") or detect_language(item.get("headline"), item.get("article_summary")),
            ),
        )
        article_id = cur.lastrowid
    except Exception:
        # Assume unique constraint violation – fetch existing id
        cur.execute("SELECT id FROM articles WHERE url = ?", (item.get("url", ""),))
        row = cur.fetchone()
        article_id = row["id"] if row else None
    con.commit()
    con.close()
    return article_id


def find_duplicate_article(url: str, content_hash: Optional[str]) -> Optional[sqlite3.Row]:
    """Return an existing article row matching the canonical URL or the content hash."""
    con = get_db()
    cur = con.cursor()
    cur.execute(
        "SELECT id, url, content_hash FROM articles WHERE url = ? OR (content_hash IS NOT NULL AND content_hash = ?) LIMIT 1",
        (url, content_hash or "\x00"),
    )
    row = cur.fetchone()
    con.close()
    return row


def insert_article_if_new(item: Dict[str, Any]) -> Tuple[Optional[int], bool]:
    """Insert an article only if it does not exist yet.

    Unlike :func:`upsert_article` this tells the caller whether the row is new, so
    ingestion statistics (new vs duplicate) are exact. Old articles are never modified.
    Returns ``(article_id, created)``.
    """
    dup = find_duplicate_article(item.get("url", ""), item.get("content_hash"))
    if dup is not None:
        return dup["id"], False
    return upsert_article(item), True


def get_stats_summary(lang: Optional[str] = None) -> Dict[str, Any]:
    con = get_db()
    cur = con.cursor()
    # Optional language scope ('ar' / 'en'); None keeps every article.
    lw, lp = (" AND language = ?", [lang]) if lang else ("", [])

    cur.execute(f"SELECT COUNT(*) AS total FROM articles WHERE 1=1{lw}", lp)
    total = int(cur.fetchone()["total"])

    cur.execute(f"SELECT COUNT(DISTINCT source) AS total_sources FROM articles WHERE source != ''{lw}", lp)
    total_sources = int(cur.fetchone()["total_sources"])

    cur.execute(f"""
        SELECT COUNT(*) AS last_24h
        FROM articles
        WHERE datetime(published_at) >= datetime('now', '-1 day'){lw}
    """, lp)
    last_24h = int(cur.fetchone()["last_24h"])

    cur.execute(f"SELECT MAX(created_at) AS last_sync FROM articles WHERE 1=1{lw}", lp)
    last_sync_row = cur.fetchone()
    last_sync = last_sync_row["last_sync"] if last_sync_row and last_sync_row["last_sync"] else None

    # Sources distribution
    cur.execute(f"""
        SELECT source, COUNT(*) AS count
        FROM articles
        WHERE source != ''{lw}
        GROUP BY source
        ORDER BY count DESC
        LIMIT 10
    """, lp)
    source_dist = [{"source": r["source"], "count": r["count"]} for r in cur.fetchall()]

    # Countries distribution
    cur.execute(f"""
        SELECT country, COUNT(*) AS count
        FROM articles
        WHERE country != '' AND country IS NOT NULL{lw}
        GROUP BY country
        ORDER BY count DESC
        LIMIT 10
    """, lp)
    country_dist = [{"country": r["country"], "count": r["count"]} for r in cur.fetchall()]

    con.close()
    return {
        "total_articles": total,
        "total_sources": total_sources,
        "articles_last_24h": last_24h,
        "last_sync": last_sync,
        "source_distribution": source_dist,
        "country_distribution": country_dist
    }
