# backend/app/db.py
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

def init_db() -> None:
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

    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_country ON articles(country)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_entity ON articles(entity)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(content_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fetch_log_source ON fetch_log(source_code, started_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_threat_saudi ON threat_intel(is_saudi, fraud_category)")

    con.commit()
    con.close()

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
                item.get("language"),
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


def get_stats_summary() -> Dict[str, Any]:
    con = get_db()
    cur = con.cursor()
    
    cur.execute("SELECT COUNT(*) AS total FROM articles")
    total = int(cur.fetchone()["total"])
    
    cur.execute("SELECT COUNT(DISTINCT source) AS total_sources FROM articles WHERE source != ''")
    total_sources = int(cur.fetchone()["total_sources"])
    
    cur.execute("""
        SELECT COUNT(*) AS last_24h 
        FROM articles 
        WHERE datetime(published_at) >= datetime('now', '-1 day')
    """)
    last_24h = int(cur.fetchone()["last_24h"])

    cur.execute("SELECT MAX(created_at) AS last_sync FROM articles")
    last_sync_row = cur.fetchone()
    last_sync = last_sync_row["last_sync"] if last_sync_row and last_sync_row["last_sync"] else None

    # Sources distribution
    cur.execute("""
        SELECT source, COUNT(*) AS count 
        FROM articles 
        WHERE source != '' 
        GROUP BY source 
        ORDER BY count DESC 
        LIMIT 10
    """)
    source_dist = [{"source": r["source"], "count": r["count"]} for r in cur.fetchall()]

    # Countries distribution
    cur.execute("""
        SELECT country, COUNT(*) AS count 
        FROM articles 
        WHERE country != '' AND country IS NOT NULL 
        GROUP BY country 
        ORDER BY count DESC 
        LIMIT 10
    """)
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
