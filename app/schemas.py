# app/db.py
import sqlite3
from typing import Dict, Any, List
from .config import DB_PATH


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _get_columns(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [row["name"] for row in cur.fetchall()]


def init_db() -> None:
    con = get_db()
    cur = con.cursor()

    # 1) Create table if not exists (base)
    cur.execute("""
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
    )
    """)

    # 2) Lightweight migration: add missing columns safely
    cols = set(_get_columns(cur, "articles"))

    if "category" not in cols:
        cur.execute("ALTER TABLE articles ADD COLUMN category TEXT")

    if "entity" not in cols:
        cur.execute("ALTER TABLE articles ADD COLUMN entity TEXT")

    # 3) Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")

    # New indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_entity ON articles(entity)")

    con.commit()
    con.close()


def row_count() -> int:
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM articles")
    c = int(cur.fetchone()["c"])
    con.close()
    return c


def upsert_article(item: Dict[str, Any]) -> bool:
    con = get_db()
    cur = con.cursor()
    try:
        cur.execute("""
        INSERT OR IGNORE INTO articles
        (source, domain, country, headline, content, article_summary, url, published_at, created_at, category, entity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
            item.get("entity", ""),
        ))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()
