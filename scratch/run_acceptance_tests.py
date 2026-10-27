# scratch/run_acceptance_tests.py
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from backend.app.saudi_api import overview, news, incidents_list, fraud_countries
from backend.app.fraud_intelligence_api import get_dashboard_view, get_incidents_drilldown
from datetime import datetime, timezone

def run_tests():
    print("=" * 70)
    print("RUNNING REAL ACCEPTANCE TESTS ON CURRENT DATABASE")
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # 1. العالم + آخر 24 ساعة (Global + 24h)
    print("\n--- Test 1: العالم + آخر 24 ساعة (Global + 24h) ---")
    ov1 = overview(scope="all", window_days=1)
    n1 = news(scope="all", window_days=1, page_size=5)
    print(f"  Incidents Count: {ov1['incidents_total']}")
    print(f"  Monitored Items Count: {n1['total']}")
    print(f"  Newest Item Date in Telemetry: {ov1.get('last_new_item')}")
    if n1['items']:
        print(f"  Sample Item 1: [{n1['items'][0]['country']}] {n1['items'][0]['title'][:50]} (Published: {n1['items'][0]['published_at']})")
    else:
        print("  Notice: No articles published strictly within the last 24h in current DB (coverage gap before latest ingest).")

    # 2. السعودية + آخر 24 ساعة (Saudi + 24h)
    print("\n--- Test 2: السعودية + آخر 24 ساعة (Saudi + 24h) ---")
    ov2 = overview(scope="saudi", country="SA", window_days=1)
    n2 = news(scope="saudi", country="SA", window_days=1, page_size=5)
    print(f"  Incidents Count: {ov2['incidents_total']}")
    print(f"  Monitored Items Count: {n2['total']}")
    print(f"  Newest Item Date: {ov2.get('last_new_item')}")

    # 3. العالم + آخر 7 أيام (Global + 7d)
    print("\n--- Test 3: العالم + آخر 7 أيام (Global + 7d) ---")
    ov3 = overview(scope="all", window_days=7)
    n3 = news(scope="all", window_days=7, page_size=5)
    print(f"  Incidents Count: {ov3['incidents_total']}")
    print(f"  Monitored Items Count: {n3['total']}")
    print(f"  Newest Item Date: {ov3.get('last_new_item')}")
    if n3['items']:
        print(f"  Sample Item 1: [{n3['items'][0]['country']}] {n3['items'][0]['title'][:50]} (Published: {n3['items'][0]['published_at']})")

    # 4. السعودية + آخر 30 يومًا (Saudi + 30d)
    print("\n--- Test 4: السعودية + آخر 30 يومًا (Saudi + 30d) ---")
    ov4 = overview(scope="saudi", country="SA", window_days=30)
    n4 = news(scope="saudi", country="SA", window_days=30, page_size=5)
    print(f"  Incidents Count: {ov4['incidents_total']}")
    print(f"  Monitored Items Count: {n4['total']}")
    print(f"  Newest Item Date: {ov4.get('last_new_item')}")
    if n4['items']:
        print(f"  Sample Item 1: [{n4['items'][0]['country']}] {n4['items'][0]['title'][:50]} (Published: {n4['items'][0]['published_at']})")

    # 5. دولة أخرى تتوفر لها بيانات (e.g. US, AE, EG)
    print("\n--- Test 5: دولة أخرى (مثلاً الإمارات AE ومصر EG والولايات المتحدة US) ---")
    for test_c in ["AE", "EG", "US"]:
        ov_c = overview(scope="all", country=test_c, window_days=30)
        n_c = news(scope="all", country=test_c, window_days=30, page_size=3)
        print(f"  [{test_c}] Incidents: {ov_c['incidents_total']} | Items: {n_c['total']} | Newest: {ov_c.get('last_new_item')}")
        if n_c['items']:
            print(f"       Top Item: {n_c['items'][0]['title'][:60]}")

    # 8. تطابق الحوادث بين الرئيسية والداشبورد والخريطة
    print("\n--- Test 8: تطابق الحوادث بين الرئيسية وMetaView 3 الداشبورد والخريطة ---")
    # Saudi 30d comparison
    home_sa_inc = overview(scope="saudi", country="SA", window_days=30)["incidents_total"]
    inc_api_sa = incidents_list(scope="saudi", country="SA", window_days=30)["total"]
    dash_sa = get_dashboard_view(geo="saudi", country="SA", window_days=30)["total_scope_incidents"]

    print(f"  Saudi 30d: Home={home_sa_inc}, IncidentsRegistry={inc_api_sa}, Dashboard={dash_sa}")
    assert home_sa_inc == inc_api_sa == dash_sa, f"Mismatch in Saudi 30d: {home_sa_inc} vs {inc_api_sa} vs {dash_sa}"
    print("  -> RECONCILIATION 100% MATCH for Saudi 30d!")

    # Global 30d comparison
    home_global_inc = overview(scope="all", window_days=30)["incidents_total"]
    inc_api_global = incidents_list(scope="all", window_days=30)["total"]
    dash_global = get_dashboard_view(geo="all", window_days=30)["total_scope_incidents"]

    print(f"  Global 30d: Home={home_global_inc}, IncidentsRegistry={inc_api_global}, Dashboard={dash_global}")
    assert home_global_inc == inc_api_global == dash_global, f"Mismatch in Global 30d: {home_global_inc} vs {inc_api_global} vs {dash_global}"
    print("  -> RECONCILIATION 100% MATCH for Global 30d!")

if __name__ == "__main__":
    run_tests()
