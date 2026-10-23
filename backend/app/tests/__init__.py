"""Test package guard: importing ``backend.app.db`` runs ``init_db()`` against ``DB_PATH``.

Point it at a throwaway file *before* any test module imports the app, so running the test-suite can
never open, migrate or back up the tracked development database. An explicit ``DB_PATH`` wins.
"""
import os
import tempfile

if not os.environ.get("DB_PATH"):
    os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="metaview-tests-"), "tests.db")
