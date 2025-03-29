
from collections import defaultdict
import tldextract
import asyncio
import os
import random
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from duckduckgo_search import DDGS
from googlesearch import search
from autogen import AssistantAgent
import asyncio


c1 = defaultdict(int)
mapper = {}
link_count = 0
images = []

deepsearch_llm_config = {
    "model" : "gpt-3.5-turbo",

}
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.198 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

async def get_links(topic: str, max_results: int = 7) -> list:
    global link_count
    print(f"[INFO] Searching for links related to query: '{topic}'")
    try:
        with DDGS() as ddgs:
            result = ddgs.news(keywords=topic, max_results=max_results, timelimit="w")
            for links in result:
                link_obj = tldextract.extract(links['url'])
                images.append(links['image'])
                key = link_obj.domain
                c1[key] += 1
                mapper[key] = links['url']
            links = [r['url'] for r in result]
            if links:
                print(f"[INFO] Found {len(links)} links for query: '{topic}' using DuckDuckGo.")
                link_count += len(links)
                return links
    except Exception as e:
        print(f"[ERROR] DuckDuckGo search failed. Error: {e}")
    print("[WARNING] DuckDuckGo failed. Switching to Google Search fallback.")
    try:
        results = list(search(topic, num_results=max_results, unique=True))
        if results:
            print(f"[INFO] Found {len(results)} links using Google Search fallback.")
            for link in results:
                linkl_obj = tldextract.extract(link)
                key = linkl_obj.domain
                c1[key] += 1
                mapper[key] = link
            link_count += len(results)
            return results
        else:
            print("[ERROR] Google Search did not return any results.")
    except Exception as e:
        print(f"[ERROR] Google Search failed: {e}")
    print(f"[ERROR] Failed to retrieve links for query: '{topic}' from both sources.")
    return []

semaphore = asyncio.Semaphore(10)
def is_valid_url(href: str) -> bool:
    if not href or href.startswith(("mailto:", "tel:", "#", "javascript:")):
        return False
    parsed = urlparse(href)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


async def scrape_page(context, url: str) -> tuple[str, list[str]]:
    async with semaphore:
        page = await context.new_page()

        # Block heavy stuff
        async def block_requests(route):
            if route.request.resource_type in ["image", "stylesheet", "font"]:
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", block_requests)

        try:
            print(f"[INFO] Scraping URL: {url}")
            await page.goto(url, wait_until='domcontentloaded', timeout=20000)

            # Wait for some real content to appear (p, h1, or article)
            try:
                await page.locator("p, h1, article").first.wait_for(timeout=10000)
            except Exception:
                print("[WARN] No main content appeared within 5s")

            # Scrape visible text elements (added div + span as fallback)
            text_blocks = await page.locator("body p, body h1, body h2, body h3, body h4, body h5, body h6, body div, body span").all_text_contents()

            # Scrape links
            links = await page.locator("a").all()
            hrefs = []
            for link in links:
                href = await link.get_attribute("href")
                if is_valid_url(href):
                    hrefs.append(href)

            # Keep your middle link logic
            middle_index = len(hrefs) // 2
            middle_hrefs = hrefs[max(0, middle_index - 1): middle_index + 1]

            # Clean + limit text
            cleaned_text = "\n".join([t.strip() for t in text_blocks if t.strip()])
            important_text = cleaned_text[:5000]

            return important_text, middle_hrefs

        except Exception as e:
            print(f"[ERROR] Failed scraping {url}: {e}")
            return f"Error scraping page: {str(e)}", []

        finally:
            await page.close()

async def scrape_contents(links: list) -> str:
    print(f"[INFO] Starting to scrape contents")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        tasks = [scrape_page(context, link) for link in links]
        results = await asyncio.gather(*tasks)
        contents, links = [], []
        for result in results:
            if isinstance(result, tuple):
                content, link = result
                contents.append(content)
                links.extend(link)
            else:
                print(f"[WARNING] Error scraping page: {result}")
        combined = "\n".join(contents)
        await browser.close()
    print(f"[INFO] Completed scraping")
    return combined, links


async def deep_website_scrape(depth: int = 2, links: list = None) -> list[str]:
    global link_count
    scraped_text, webpage_links = await scrape_contents(links)
    link_count += len(webpage_links)
    for link in webpage_links:
        if not link:
            continue
        link_obj = tldextract.extract(link)
        key = link_obj.domain
        c1[key] += 1
        mapper[key] = link
    all_scraped = [scraped_text]
    if depth > 1:
        deeper_scraped = await deep_website_scrape(depth=depth - 1, links=webpage_links)
        all_scraped.extend(deeper_scraped)
    print("Completed deep scraping.")
    return all_scraped


async def search_and_scrape_tool(topic: str, max_results: int = 7):
    links = await get_links(topic, max_results= max_results)
    if not links:
        print("[ERROR] No links found for the given topic.")
        return []
    print(f"[INFO] Found {len(links)} links for topic: '{topic}'")
    data = deep_website_scrape(depth=1, links=links)
    data_dump = "\n".join(data)
    return data_dump



async def main():
    topic = "World News"
    print(f"[INFO] Starting deep search for topic: {topic}")
    #links = await get_links(topic, max_results=15)
    links = [
        "https://www.msn.com/en-us/politics/government/denmark-doesn-t-appreciate-the-tone-of-us-greenland-remarks-minister-says/ar-AA1BT2tJ",
        "https://www.msn.com/en-us/news/us/radio-free-asia-says-it-will-fully-shut-down-by-end-of-april-without-court-intervention/ar-AA1BSCOM",
        "https://www.msn.com/en-us/news/world/hegseth-brought-his-wife-to-sensitive-meetings-with-foreign-military-officials/ar-AA1BSglZ",
        "https://www.msn.com/en-us/news/world/jd-vance-warns-there-s-very-strong-evidence-china-russia-want-greenland/ar-AA1BRzNX",
        "https://www.msn.com/en-us/news/world/four-killed-in-mass-russian-drone-attack-on-dnipro-ukraine/ar-AA1BSDiN",
        "https://www.msn.com/en-us/news/us/a-look-at-who-has-been-detained-or-deported-in-a-us-crackdown-on-pro-palestinian-protesters/ar-AA1BO5wZ",
        "https://www.msn.com/en-us/news/world/vance-scolds-denmark-during-greenland-trip/ar-AA1BS076",
        "https://www.msn.com/en-us/news/world/mass-evacuations-as-israel-strikes-beirut-suburb/ar-AA1BQ7A8",
        "https://www.msn.com/en-us/news/other/new-us-strikes-against-houthi-rebels-kill-at-least-1-in-yemen/ar-AA1BSDbk",
        "https://www.msn.com/en-us/news/politics/remaining-usaid-staff-fired-trump-says-myanmar-will-still-get-earthquake-aid/ar-AA1BSoIL",
        "https://www.msn.com/en-us/politics/government/quakers-say-london-police-arrested-six-people-at-meeting-on-climate-change-gaza/ar-AA1BSojy",
        "https://www.msn.com/en-us/public-safety-and-emergencies/natural-disasters/live-updates-earthquake-hits-myanmar-and-thailand-rescue-efforts-underway-as-death-toll-rises/ar-AA1BSoqj",
    ]
    data = await deep_website_scrape(depth=1, links=links)
    print("[INFO] Deep search completed.")
    print(f"[INFO] Total links scraped: {link_count}")
    print(f"[INFO] Total unique domains: {len(c1)}")
    data_dump = "\n".join(data)
    print(f"data_dump: {data_dump}")

if __name__ == "__main__":
    asyncio.run(main())