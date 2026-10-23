# backend/app/db.py
import sqlite3
from typing import Dict, Any, List
from .config import DB_PATH

def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
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
        """
    )


    # Ensure additional columns safely
    _ensure_column(con, "articles", "category", "TEXT")
    _ensure_column(con, "articles", "entity", "TEXT")
    _ensure_column(con, "articles", "image_url", "TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_country ON articles(country)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_entity ON articles(entity)")

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
                published_at, created_at, category, entity, image_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
