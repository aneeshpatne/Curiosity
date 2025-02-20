from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
import os
from playwright.async_api import async_playwright
import asyncio
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ddgs = DDGS()
llm=ChatOpenAI(model='gpt-4o-mini-2024-07-18', api_key=SecretStr(api_key), temperature=0.0)

async def scrape_page(context, url):
    """Scrapes a single page using an existing browser instance."""
    page = await context.new_page()
    try:
        await page.goto(url, wait_until='load')
        text_blocks = await page.locator("body p, body h1, body h2, body h3, body h4, body h5, body h6").all_text_contents()
        cleaned_text = "\n".join([t.strip() for t in text_blocks if t.strip()])
        important_text = cleaned_text[:2000]
        return important_text if important_text else "No text found"
    except Exception as e:
        return f"Error scraping page: {str(e)}"
    finally:
        await page.close()

async def searchAgent(topic):
    result = ddgs.text(topic, max_results=5)
    links = [r['href'] for r in result]
    print("Search Results:", links)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        ls = []
        for index, content in enumerate(scraped_contents, start=1):
            ls.append(str(f"[{index}]", content))
        print("".join(ls))
        await browser.close()
    
def summarize(content, query):
    prompt = (
        f"The user searched for: '{query}'.\n\n"
        f"The following text was retrieved and includes citations (e.g., [1], [2]):\n\n{content}\n\n"
        "Your task is to produce a concise, factual summary of the relevant information from the text. "
        "Ignore any irrelevant content. If the provided text contains little or no relevant information, "
        "return a fallback message such as 'Insufficient relevant information found.' "
        "Ensure that you preserve any citation labels (e.g., [1], [2]) exactly as they appear. "
        "Return only the summarized content with no additional commentary."
    )

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    topic = "Devendra Fadnavis News"
    topic += "ENGLISH ONLY"
    asyncio.run(searchAgent(topic))



# def deepSearch(topic):
#     search_prompt = (
#         f"The user prompt is: {topic}. "
#         "you are a step before the data is fed to the API. you need to take this user prompt and convert this into a search query. "
#         "to deep search, generate 5 more search queries based on the user prompt. "
#         "return the search query, only and nothing else."
#     )
#     response = llm.invoke(search_prompt)
#     print("Search Query:", response.content)