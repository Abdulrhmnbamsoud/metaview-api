# app/config.py
import os

APP_NAME = "MetaView API"
APP_VERSION = "1.0.0"

APP_DIR = os.path.dirname(os.path.abspath(__file__))  # .../app
DATA_DIR = os.getenv("DATA_DIR", os.path.join(APP_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "metaview.db"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))
USER_AGENT = os.getenv("USER_AGENT", "MetaViewBot/1.0 (+https://metaview)")
AUTO_INGEST = os.getenv("AUTO_INGEST", "false").lower() == "true"
AUTO_INGEST_EVERY_MIN = int(os.getenv("AUTO_INGEST_EVERY_MIN", "15"))

# حماية من كراش: حد أعلى لكل Feed
MAX_ITEMS_PER_FEED = int(os.getenv("MAX_ITEMS_PER_FEED", "50"))
