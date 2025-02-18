import os
import asyncio
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Controller, Agent
from pydantic import SecretStr

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=SecretStr(api_key))

class Article(BaseModel):
    url: str
    article_text: str
class Articles(BaseModel):
    articles: List[Article]
class Summary(BaseModel):
    url: str
    summary: str
class Summaries(BaseModel):
    summaries: List[Summary]
class URLResults(BaseModel):
    urls: List[str]

search_controller = Controller(output_model=URLResults)
scrape_controller = Controller(output_model=Articles)
summarize_controller = Controller(output_model=Summaries)

def create_agent(topic):
    search_task = (
        f"Search on duck duck go for the top 5 articles on '{topic}' and return only the article URLs "
        "in JSON format. The JSON should follow this structure: { 'urls': [ 'url1', 'url2', ... ] }."
    )
    scrape_task = (
        "For each URL provided, scrape the full article content and return the important body and ignoring all the irrlevant information in JSON format. "
        "Return a JSON object following this structure: { 'articles': [ { 'url': '...', 'article_text': '...' }, ... ] }."
    )
    summarize_task = (
        "Summarize the content of each article provided. Return a JSON object with a summary for each URL "
        "following this structure: { 'summaries': [ { 'url': '...', 'summary': '...' }, ... ] }."
    )
    search_agent = Agent(search_task, llm= llm, controller=search_controller)
    scrape_agent = Agent(task=scrape_task, llm=llm, controller=scrape_controller)
    summarize_agent = Agent(task=summarize_task, llm=llm, controller=summarize_controller)
    return search_agent, scrape_agent, summarize_agent
        
async def main():
    topic = "Machine Learning"
    search_agent, scrape_agent, summarize_agent = create_agent(topic)
    search_history = await search_agent.run()
    search_result = search_history.final_result()
    print("Search Results:", search_result)



if __name__ == '__main__':
    asyncio.run(main())
