from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
import os
from playwright.sync_api import sync_playwright
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ddgs = DDGS()
llm=ChatOpenAI(model='gpt-4o-mini-2024-07-18', api_key=SecretStr(api_key), temperature=0.0)

def searchAgent(topic):
    result = ddgs.text(topic, max_results=5)
    link = [r['href'] for r in result]
    print("Search Results:", link)
def scrape(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page= browser.new_page()
        page.goto(url, wait_until='load')

if __name__ == "__main__":
    topic = "Elon Musk"
    searchAgent(topic)



# def deepSearch(topic):
#     search_prompt = (
#         f"The user prompt is: {topic}. "
#         "you are a step before the data is fed to the API. you need to take this user prompt and convert this into a search query. "
#         "to deep search, generate 5 more search queries based on the user prompt. "
#         "return the search query, only and nothing else."
#     )
#     response = llm.invoke(search_prompt)
#     print("Search Query:", response.content)