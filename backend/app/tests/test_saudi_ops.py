# backend/app/tests/test_saudi_ops.py
"""Operational safety tests: 429 handling, cross-process lease, protected manual ingest."""
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import httpx

from backend.app import db as db_mod
from backend.app import saudi_ingest as si


class RetryAfterParsing(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(si.parse_retry_after("30"), 30.0)

    def test_http_date(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(si.parse_retry_after("Thu, 01 Jan 2026 12:01:00 GMT", now), 60.0)

    def test_past_date_and_garbage(self):
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(si.parse_retry_after("Thu, 01 Jan 2026 11:00:00 GMT", now), 0.0)
        self.assertIsNone(si.parse_retry_after("soon-ish"))
        self.assertIsNone(si.parse_retry_after(None))


class Http429Handling(unittest.TestCase):
    def setUp(self):
        si._NET = si.NetPolicy()
        self.sleeps = []
        p = mock.patch.object(si.time, "sleep", lambda s: self.sleeps.append(s))
        p.start()
        self.addCleanup(p.stop)

    def _client(self, handler):
        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_honours_retry_after_then_succeeds(self):
        calls = {"n": 0}

        def h(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "7"})
            return httpx.Response(200, text="ok")

        r, err = si._http_get(self._client(h), "https://example.test/feed")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(calls["n"], 2)
        self.assertIn(7.0, self.sleeps)  # waited exactly Retry-After

    def test_exponential_backoff_without_header(self):
        calls = {"n": 0}

        def h(req):
            calls["n"] += 1
            return httpx.Response(429) if calls["n"] < 3 else httpx.Response(200)

        r, _ = si._http_get(self._client(h), "https://example.test/feed")
        self.assertEqual(r.status_code, 200)
        waits = [s for s in self.sleeps if s >= 2.0]
        self.assertEqual(len(waits), 2)
        self.assertLess(waits[0], waits[1] + 1.0)  # grows (jitter < 1s)

    def test_gives_up_sets_cooldown_and_stops_hitting_host(self):
        calls = {"n": 0}

        def h(req):
            calls["n"] += 1
            return httpx.Response(429, headers={"Retry-After": "3600"})

        c = self._client(h)
        r, _ = si._http_get(c, "https://example.test/feed")
        self.assertEqual(r.status_code, 429)
        self.assertEqual(calls["n"], 1)  # Retry-After > cap: no wait, no hammering
        self.assertGreater(si._NET.cooling("example.test"), 0)
        r2, err = si._http_get(c, "https://example.test/other")
        self.assertIsNone(r2)
        self.assertTrue(err.startswith(si.RATE_LIMITED_PREFIX))
        self.assertEqual(calls["n"], 1)  # no further request during cooldown
        self.assertEqual(si._net_failure(err)["state"], "rate_limited")
        self.assertEqual(si._bad_status(r)["state"], "rate_limited")

    def test_other_errors_are_failed_not_rate_limited(self):
        r, _ = si._http_get(self._client(lambda req: httpx.Response(404)), "https://example.test/x")
        self.assertEqual(si._bad_status(r)["state"], "failed")


class LeaseMutualExclusion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        p = mock.patch.object(db_mod, "DB_PATH", os.path.join(self.tmp, "t.db"))
        p.start()
        self.addCleanup(p.stop)
        db_mod.init_db()

    def test_second_holder_is_refused_until_release(self):
        ok, _, _ = si._acquire_lease("run-A")
        self.assertTrue(ok)
        ok2, holder, _ = si._acquire_lease("run-B")
        self.assertFalse(ok2)
        self.assertEqual(holder, "run-A")
        self.assertTrue(si.lease_status()["active"])
        si._release_lease("run-A")
        self.assertFalse(si.lease_status()["active"])
        ok3, _, _ = si._acquire_lease("run-B")
        self.assertTrue(ok3)

    def test_expired_lease_is_taken_over(self):
        si._acquire_lease("crashed")
        con = db_mod.get_db()
        con.execute("UPDATE ingest_lease SET expires_at=?",
                    ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),))
        con.commit()
        con.close()
        self.assertFalse(si.lease_status()["active"])
        ok, _, _ = si._acquire_lease("new-run")
        self.assertTrue(ok)

    def test_run_refuses_when_other_process_holds_lease(self):
        si._acquire_lease("other-process")
        res = si.run_saudi_ingest("manual")
        self.assertEqual(res["status"], "already_running")
        self.assertEqual(si.start_background_run("manual")["scope"], "other_process")


class ManualIngestProtection(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.app import saudi_api

        self.api = saudi_api
        app = FastAPI()
        app.include_router(saudi_api.router)
        self.client = TestClient(app)
        self.start = mock.patch.object(saudi_api.saudi_ingest, "start_background_run",
                                       return_value={"status": "started"})
        self.start_mock = self.start.start()
        self.addCleanup(self.start.stop)
        for name, val in (("RUN_STATE", {"running": False}),):
            p = mock.patch.object(saudi_api.saudi_ingest, name, val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(saudi_api.saudi_ingest, "lease_status", return_value={"active": False})
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(saudi_api.saudi_ingest, "_last_finished", return_value=None)
        p.start()
        self.addCleanup(p.stop)

    def test_disabled_when_token_not_configured(self):
        with mock.patch.object(self.api, "ADMIN_TOKEN", ""):
            r = self.client.post("/saudi-fraud/ingest/run", headers={"X-Admin-Token": "anything"})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(self.client.post("/saudi-fraud/ingest/run").status_code, 503)
        self.start_mock.assert_not_called()

    def test_rejects_missing_or_wrong_token(self):
        with mock.patch.object(self.api, "ADMIN_TOKEN", "s3cret"):
            self.assertEqual(self.client.post("/saudi-fraud/ingest/run").status_code, 401)
            r = self.client.post("/saudi-fraud/ingest/run", headers={"X-Admin-Token": "nope"})
            self.assertEqual(r.status_code, 401)
        self.start_mock.assert_not_called()

    def test_accepts_correct_token(self):
        with mock.patch.object(self.api, "ADMIN_TOKEN", "s3cret"):
            r = self.client.post("/saudi-fraud/ingest/run", headers={"X-Admin-Token": "s3cret"})
        self.assertEqual(r.status_code, 200)
        self.start_mock.assert_called_once()

    def test_manual_cooldown_returns_429_with_retry_after(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=30)
        with mock.patch.object(self.api, "ADMIN_TOKEN", "s3cret"), \
                mock.patch.object(self.api.saudi_ingest, "_last_finished", return_value=recent):
            r = self.client.post("/saudi-fraud/ingest/run", headers={"X-Admin-Token": "s3cret"})
        self.assertEqual(r.status_code, 429)
        self.assertGreater(int(r.headers["Retry-After"]), 0)
        self.start_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
