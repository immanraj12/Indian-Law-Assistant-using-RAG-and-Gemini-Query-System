# playwright_download.py
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

if len(sys.argv) < 2:
    print("Usage: python playwright_download.py urls.txt")
    raise SystemExit(1)

urls = [
    u.strip()
    for u in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if u.strip() and not u.strip().startswith("#")
]
OUT = Path("data")
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)  # set False to watch
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )
    for url in urls:
        fname = url.split("/")[-1].split("?")[0]
        outp = OUT / fname
        if outp.exists():
            print(f"Exists: {fname}")
            continue
        print("Fetching", url)
        # Use browser context request (sends real browser-like headers & cookies)
        resp = context.request.get(
            url, headers={"Referer": "https://www.indiacode.nic.in/"}
        )
        if resp.ok:
            content = resp.body()
            outp.write_bytes(content)
            print("Saved:", outp)
        else:
            print("Failed:", resp.status, url)
    browser.close()
