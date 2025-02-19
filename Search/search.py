import os
import asyncio
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from browser_use import Controller, Agent
from pydantic import SecretStr
import json
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
#llm=ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='mistralai/mistral-7b-instruct:free', api_key=SecretStr(api_key))
llm=ChatOpenAI(model='gpt-4o-mini-2024-07-18', api_key=SecretStr(api_key), temperature=0.0)
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

async def main():
    # topic = "Elon musk DOGE NEWS"
    # search_task = (
    #     f"Search on duckduckgo.com search for {topic} and return the URLs only for the top 2 articles. Dont redirect anywhere. "
    #     "Extract the whole URL to the article, not the search result."
    #     "You are forbiddien to click on any links."
    #     "Just Copy the URLS"
    #     "Dont open any links."
    #     "ignore search results that are ads or irrelevant. Return only the article URLs "
    #     "in JSON format. The JSON should follow this structure: { 'urls': [ 'url1', 'url2', ... ] }."
    # )
    # search_agent = Agent(search_task, llm=llm, controller=search_controller)
    # search_history = await search_agent.run()
    # search_result = search_history.final_result()
    # print("Search Results:", search_result)
    # urls_data = URLResults.model_validate_json(search_result)
    # print("URLs Data:", urls_data.urls)
    # urls_str = json.dumps(urls_data.urls)
    urls_data ={"urls":["https://www.theguardian.com/us-news/live/2025/feb/18/donald-trump-elon-musk-eric-adams-us-politics-live-news"]}
    content = []
    for url in urls_data["urls"]:  
        scrape_task = f"""go to URL: {url}
            1. Open the news article URL.
            2. Immediately close any pop-ups or overlays that appear.
            3. Ignore irrelevant advertisements or elements and Scroll.
            4. Process only the content visible on the initially loaded page.
            5. Do not open any additional links or follow any redirects.
            6. Scroll down at most once to load additional content if needed.
            7. Extract the content only once and close the browser tab, your work is done.
            7. If any further scrolling is required, or if any advertisements or irrelevant elements are detected, immediately close the tab.
            8. Extract and return only the main article text.
            9. Optimize resource usage by extracting only the necessary content and terminating the process as soon as the relevant information is captured.
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
