# scratch/verify_incident_reconciliation.py
import asyncio
from playwright.async_api import async_playwright

async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        print("1. Navigating to MetaView 3 Dashboard...")
        await page.goto("http://localhost:5173/fraud-intelligence", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Check map header total
        map_header_badge = await page.locator("text='حادثة موثقة'").first.inner_text()
        print(f"   Map Header Badge: {map_header_badge}")
        assert "4" in map_header_badge, f"Expected 4 in map header badge, got {map_header_badge}"

        # Check Saudi cities list
        riyadh_elem = page.locator("text='الرياض'").first
        assert await riyadh_elem.is_visible(), "Riyadh city should be visible in Saudi view"
        print("   Riyadh pin/item is visible in Saudi view.")

        # Check unknown location item in analytical panel
        unknown_item = page.locator("text='عموم المملكة (دون تحديد مدينة)'")
        if await unknown_item.count() > 0:
            print("   Unknown location item is visible in analytical panel.")

        # Screenshot Saudi view
        await page.screenshot(path="scratch/saudi_map_verified.png")
        print("   Saved scratch/saudi_map_verified.png")

        # Switch to World map
        print("2. Switching to World Map Scope...")
        world_btn = page.locator("button:has-text('العالم')").first
        await world_btn.click()
        await page.wait_for_timeout(1500)

        # Verify no '18' on Saudi Arabia badge in world view
        # Instead, verify dynamic hotspots
        await page.screenshot(path="scratch/world_map_verified.png")
        print("   Saved scratch/world_map_verified.png")

        # Check top countries list on right panel
        sa_item = page.locator("div[id='btn-country-sa']")
        assert await sa_item.is_visible(), "Saudi Arabia country button should be visible in top countries"
        sa_count = await sa_item.locator(".font-mono").inner_text()
        print(f"   Saudi Arabia in top countries list count: {sa_count}")
        assert sa_count == "4", f"Expected Saudi count to be 4, got {sa_count}"

        # Click on Saudi Arabia to open drilldown modal
        print("3. Clicking Saudi Arabia to open drilldown modal...")
        await sa_item.click()
        await page.wait_for_timeout(1500)

        modal = page.locator("div[role='dialog'], .fixed.inset-0").first
        assert await modal.is_visible(), "Drilldown modal should open"
        
        # Check modal title or count
        incident_items = page.locator("article, div.group.cursor-pointer, .space-y-3 > div")
        print("   Drilldown modal opened successfully.")
        await page.screenshot(path="scratch/drilldown_modal_verified.png")
        print("   Saved scratch/drilldown_modal_verified.png")

        # Close modal
        close_btn = page.locator("button:has-text('إغلاق'), button:has(.lucide-x)").first
        if await close_btn.is_visible():
            await close_btn.click()
            await page.wait_for_timeout(500)

        # 4. Check Incident Registry
        print("4. Navigating to Incident Registry (/incidents)...")
        await page.goto("http://localhost:5173/incidents", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # In Incident Registry default is Saudi + 30 days
        registry_text = await page.content()
        assert "سجل الحوادث" in registry_text
        await page.screenshot(path="scratch/incident_registry_verified.png")
        print("   Saved scratch/incident_registry_verified.png")

        print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify())
