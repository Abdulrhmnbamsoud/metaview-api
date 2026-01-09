# app/db.py
import sqlite3
from typing import Dict, Any
from .config import DB_PATH

def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db() -> None:
    con = get_db()
    cur = con.cursor()
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain)")
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
        (source, domain, country, headline, content, article_summary, url, published_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()
