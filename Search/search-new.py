from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from browser_use import Controller, Agent
from pydantic import SecretStr
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
ddgs = DDGS()
llm=ChatOpenAI(model='gpt-4o-mini-2024-07-18', api_key=SecretStr(api_key), temperature=0.0)
async def SearchAgent(topic):
    search_prompt = (
        f"The user prompt is: {topic}. "
        "you are a step before the data is fed to the API. you need to take this user prompt and convert this into a search query. "
        "return the search query, only and nothing else."
    )
    response = llm.invoke(search_prompt)
    print("Search Query:", response.content)
if __name__ == "__main__":
    topic = "maharashtra latest news"
    asyncio.run(SearchAgent(topic))