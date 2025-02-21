from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
import os
from playwright.async_api import async_playwright
import asyncio
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openRouterKey = os.getenv("OPEN_ROUTER_KEY")

# Initialize LLMs
summary_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))

# Memory to track conversation history (Deprecated, but still functional for now)
memory = ConversationBufferMemory(memory_key="chat_history")

# DuckDuckGo Search
ddgs = DDGS()

async def scrape_page(context, url):
    """Scrapes a single page using an existing browser instance."""
    page = await context.new_page()
    try:
        await page.goto(url, wait_until='load')
        text_blocks = await page.locator("body p, body h1, body h2, body h3, body h4, body h5, body h6").all_text_contents()
        cleaned_text = "\n".join([t.strip() for t in text_blocks if t.strip()])
        important_text = cleaned_text[:10000]
        return important_text if important_text else "No text found"
    except Exception as e:
        return f"Error scraping page: {str(e)}"
    finally:
        await page.close()

async def searchAgent(topic):
    """Searches for the topic, scrapes data, and summarizes it."""
    result = ddgs.text(topic, max_results=10)
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
        print("Scraped Contents:")
        summary = await summarize(ls, topic)
        print("Summary:", summary)
        await browser.close()
        return summary

async def summarize(content, query):
    """Summarizes the gathered content."""
    prompt = (
        f"The user has submitted the following query: '{query}'. Using the provided sources: {content}, "
        "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
        "Ensure the response remains focused, preserving citation labels (e.g., [1], [2]) for accuracy. "
        "Some sources may contain overlapping or redundant information; synthesize the data to avoid repetition. "
        "If the sources lack sufficient data to fully address the query, conclude with: 'Insufficient relevant information found.' "
    )
    response = await summary_llm.ainvoke(prompt)
    return response.content

# Define the search tool (fixed with func=None)
search_tool = Tool(
    name="WebSearch",
    func=None,  # Explicitly set for async-only tool
    coroutine=searchAgent,  # Async function
    description="Use this tool when additional web search is needed to answer the user's question."
)

# Initialize an agent with function-calling ability
agent = initialize_agent(
    tools=[search_tool],
    llm=agent_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

async def follow_up(question):
    """Handles follow-up questions using conversation history or web search if needed."""
    previous_conversations = memory.load_memory_variables({})
    prompt = (
        f"The user has asked a follow-up question: '{question}'.\n"
        f"Here is the previous conversation for context: {previous_conversations}\n"
        "First, determine whether you can answer this using existing information.\n"
        "If more information is needed, call the WebSearch tool to retrieve data.\n"
        "After you call the WebSearch tool display the answer as-is its a summarised version and you are not needed to do anything with it.\n"
        "Otherwise, respond directly using the provided context."
    )
    response_dict = await agent.ainvoke({"input": prompt})  # Pass dict and await
    response = response_dict["output"]  # Extract output
    memory.save_context({"input": question}, {"output": response})
    print("Follow-Up Response:", response)
    return response

async def main():
    user_query = "What are the latest advancements in AI?"
    result = await follow_up(user_query)
    print("Final Answer:", result)

# Example usage
if __name__ == "__main__":
    asyncio.run(main())