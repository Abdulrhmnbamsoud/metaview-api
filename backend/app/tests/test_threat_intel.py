import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure a fresh isolated SQLite DB for each test case
class ThreatIntelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory within the workspace
        temp_dir = Path(__file__).parent / "tmp_db"
        temp_dir.mkdir(parents=True, exist_ok=True)
        db_file = temp_dir / "test_metaview.db"
        os.environ["DB_PATH"] = str(db_file)
        # Reinitialize the database schema
        from backend.app import db as db_module
        cls._orig_db_path = db_module.DB_PATH
        db_module.DB_PATH = str(db_file)  # config is read once at import; env var alone is not enough
        if db_file.exists():
            db_file.unlink()
        db_module.init_db()
        cls.db_file = db_file
        cls.temp_dir = temp_dir
        # Import the FastAPI app after DB_PATH is set
        from backend.app.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        from backend.app import db as db_module
        db_module.DB_PATH = cls._orig_db_path
        if cls.db_file.exists():
            cls.db_file.unlink()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_threat_intel_flow(self):
        # Insert a dummy article directly via the db layer
        from backend.app import db as db_module
        article = {
            "source": "test_source",
            "domain": "example.com",
            "country": "Saudi Arabia",
            "headline": "Financial fraud alert",
            "content": "",
            "article_summary": "A new banking fraud scheme detected.",
            "url": "https://example.com/fraud",
            "published_at": "2024-01-01T00:00:00Z",
            "created_at": "2024-01-01T00:00:00Z",
            "category": "Finance",
            "entity": None,
            "image_url": "",
        }
        article_id = db_module.upsert_article(article)
        self.assertIsNotNone(article_id)

        # Run threat intelligence processing on the article
        from backend.app.intel import process_article_for_intel
        threat_id = process_article_for_intel({**article, "article_id": article_id})
        self.assertIsNotNone(threat_id)

        # Verify the /threats endpoint returns the created threat
        resp = self.client.get("/threats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total"], 1)
        threat = data["threats"][0]
        self.assertEqual(threat["id"], threat_id)
        self.assertEqual(threat["article_id"], article_id)
        self.assertIn(threat["threat_type"], ["financial_fraud", "phishing", "malware", "ransomware", "Other"])

        # Verify the single threat endpoint
        resp2 = self.client.get(f"/threats/{threat_id}")
        self.assertEqual(resp2.status_code, 200)
        threat_detail = resp2.json()
        self.assertEqual(threat_detail["id"], threat_id)
        self.assertEqual(threat_detail["article_id"], article_id)
