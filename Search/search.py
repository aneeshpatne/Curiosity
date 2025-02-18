import os
import asyncio
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Controller, Agent
from pydantic import SecretStr
import json
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
        f"Search on google for the top 2 articles on '{topic}' and return only the article URLs"
        "Extract the exact URL."
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
    topic = "Aneesh Patne"
    search_task = (
        f"Search on google.com search for {topic} and return the URLs only for the top 2 articles. Dont redirect anywhere. "
        "ignore search results that are ads or irrelevant. Return only the article URLs "
        "in JSON format. The JSON should follow this structure: { 'urls': [ 'url1', 'url2', ... ] }."
    )
    search_agent = Agent(search_task, llm=llm, controller=search_controller)
    search_history = await search_agent.run()
    search_result = search_history.final_result()
    print("Search Results:", search_result)
    urls_data = URLResults.model_validate_json(search_result)
    print("URLs Data:", urls_data.urls)
    urls_str = json.dumps(urls_data.urls)
    content = []
    for url in urls_data.urls:  
        scrape_task = f"""go to URL: {url}
                1. close any popups that appear.
                2. extract all text.
                3. ignore any irrelevant information like ads, comments, etc.
                4. scroll aggresively. 
                5. dont spend more than 2 steps without scrolling.
                6. if u see even 1 advert or irrelevant information, close the tab.
        Return a JSON object following this structure:
        {{
            "articles": [
                {{
                    "url": "{url}",
                    "article_text": "..."
                }}
            ]
        }}

        Important: Do not summarize, modify, or alter the text. Simply extract the article content as it is.
        """


        scrape_agent = Agent(scrape_task, llm=llm, controller=scrape_controller)
        scrape_history = await scrape_agent.run()
        scrape_result = scrape_history.final_result()
        print(scrape_result)
        content.append(scrape_result)
    print(content)






if __name__ == '__main__':
    asyncio.run(main())
