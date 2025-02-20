from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
import os
from playwright.async_api import async_playwright
import asyncio
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_key_groq = os.getenv("GROQ_KEY")
ddgs = DDGS()
llm=ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
groqLLM = ChatOpenAI(model='deepseek-r1-distill-llama-70b', api_key=SecretStr(api_key_groq), temperature=0.0, base_url="https://api.groq.com/openai/v1")

async def scrape_page(context, url):
    """Scrapes a single page using an existing browser instance."""
    page = await context.new_page()
    try:
        await page.goto(url, wait_until='load')
        text_blocks = await page.locator("body p, body h1, body h2, body h3, body h4, body h5, body h6").all_text_contents()
        cleaned_text = "\n".join([t.strip() for t in text_blocks if t.strip()])
        important_text = cleaned_text[:5000]
        return important_text if important_text else "No text found"
    except Exception as e:
        return f"Error scraping page: {str(e)}"
    finally:
        await page.close()

async def searchAgent(topic):
    result = ddgs.text(topic, max_results=4)
    links = [r['href'] for r in result]
    print("Search Results Generated")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        ls = []
        for index, content in enumerate(scraped_contents, start=1):
            ls.append(f"[{index}] {content}")
        print("Scrapped Contents:")
        summary = summarize(ls,topic)
        print("Summary:", summary)
        await browser.close()
    
def summarize(content, query):
    prompt = (
        f"The user has submitted the following query: '{query}'. Using the provided sources: {content}, "
        "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
        "Incorporate a comprehensive summary of the most relevant information, including key points, supporting evidence, contextual explanations, and any additional insights drawn from the sources that enhance the answer. "
        "Ensure the response remains focused by excluding any extraneous or unrelated material, while carefully preserving citation labels (e.g., [1], [2]) to maintain accuracy and traceability to the original content. "
        "Some sources may contain overlapping or redundant information; please synthesize the data to avoid repetition and present a cohesive response that covers all aspects of the query. "
        "you can also include your own knowledge and expertise to provide a complete and well-rounded answer. "
        "Keep a formal and professional tone throughout the response. "
        "Only cite the sources in the format [] [] separately. "
        "If the sources lack sufficient data to fully address the query, conclude with: 'Insufficient relevant information found,' and briefly explain why the available information falls short of providing a complete answer."
    )
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    topic = "Eknath Shinde Fadnavis Rift"
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