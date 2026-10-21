# backend/app/config.py
import os

APP_NAME = "MetaView API"
APP_VERSION = "2.0.0"

APP_DIR = os.path.dirname(os.path.abspath(__file__))  # .../app
BACKEND_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BACKEND_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 MetaViewBot/2.0")
AUTO_INGEST = os.getenv("AUTO_INGEST", "false").lower() == "true"
AUTO_INGEST_EVERY_MIN = int(os.getenv("AUTO_INGEST_EVERY_MIN", "15"))

MAX_ITEMS_PER_FEED = int(os.getenv("MAX_ITEMS_PER_FEED", "40"))
