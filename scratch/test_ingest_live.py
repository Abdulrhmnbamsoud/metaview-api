import asyncio
import json
from backend.app.sources import GLOBAL_FRAUD_FEEDS
from backend.app.ingest import ingest_once

async def main():
    print(f"Testing ingestion of {len(GLOBAL_FRAUD_FEEDS)} global fraud feeds...")
    res = await ingest_once(GLOBAL_FRAUD_FEEDS[:6], limit_per_feed=5, allow_web_image_scrape=False)
    print("Ingest Result Summary:")
    print("Started at:", res.get("started_at"))
    print("Finished at:", res.get("finished_at"))
    print("Inserted total:", res.get("inserted_total"))
    print("Skipped dup total:", res.get("skipped_dup_total"))
    print("Errors total:", res.get("errors_total"))
    for s in res.get("sources", []):
        print(f" - {s['source']}: fetched={s['fetched_entries']} inserted={s['inserted']} dups={s['skipped_dup']} err={s['error']}")

if __name__ == "__main__":
    asyncio.run(main())
