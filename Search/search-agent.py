import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from duckduckgo_search import DDGS
from playwright.async_api import async_playwright
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
import os

# Load API keys from environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set up different LLMs for conversation and summarization
summary_llm = ChatOpenAI(model="o1-mini", api_key=SecretStr(api_key))
agent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=SecretStr(api_key),
    temperature=0.7
)

# System prompt that enforces the correct output format
SYSTEM_PROMPT = """You are a helpful AI assistant. For every response, you MUST use the following format:

Thought: First, think about whether you need to search for information or can answer directly.
Action: Choose one of:
- "Direct Response" (for questions you can answer without searching)
- "WebSearch" (only when you need current information)
Action Input: Your response if direct, or the search query if searching.

Example for direct response:
Thought: This is a greeting, I can respond directly.
Action: Direct Response
Action Input: Hello! How can I help you today?

Example for search needed:
Thought: I need current information about this topic.
Action: WebSearch
Action Input: latest news about artificial intelligence developments

Always maintain this exact format for EVERY response."""

# Initialize memory for contextual follow-ups
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize DuckDuckGo Search
ddgs = DDGS()

async def scrape_page(context, url):
    """Scrapes a webpage using Playwright."""
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="load")
        text_blocks = await page.locator("body p, body h1, body h2, body h3, body h4, body h5, body h6").all_text_contents()
        cleaned_text = "\n".join([t.strip() for t in text_blocks if t.strip()])
        important_text = cleaned_text[:10000]
        return important_text if important_text else "No relevant text found."
    except Exception as e:
        return f"Error scraping page: {str(e)}"
    finally:
        await page.close()

async def searchAgent(topic):
    """Performs a DuckDuckGo search, scrapes results, and summarizes."""
    results = ddgs.text(topic, max_results=3)  # Reduced to 3 results to minimize token usage
    links = [r["href"] for r in results]

    print("🔍 Search Results Found. Scraping pages...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        await browser.close()

    formatted_sources = "\n\n".join([f"[{i+1}] {scraped_contents[i][:1000]}" for i in range(len(scraped_contents))])  # Limited to first 1000 chars per source
    summary = summarize(formatted_sources, topic)
    return summary

def summarize(content, query):
    """Summarizes the scraped search results using o1-mini."""
    prompt = (
        f"Summarize the key information related to: '{query}'. Keep it brief and focused.\n\n"
        f"{content}\n\n"
        "Provide a concise summary with key points only."
    )
    response = summary_llm.invoke(prompt).content
    return response

def web_search_sync(query):
    """Sync wrapper for async searchAgent function."""
    return asyncio.run(searchAgent(query))

web_search = Tool(
    name="WebSearch",
    func=web_search_sync,
    description="Use this tool ONLY when you need current information, news, or specific facts that you're not confident about. For general knowledge questions or logical reasoning, respond directly without using this tool."
)

# Initialize LangChain Agent with the correct system message setup
agent = initialize_agent(
    tools=[web_search],
    llm=agent_llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=2,  # Reduced max iterations
    early_stopping_method="generate",
    agent_kwargs={
        "system_message": SystemMessage(content=SYSTEM_PROMPT),
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
    }
)

def handle_chat(question):
    """Handles user input with improved error handling and response formatting."""
    try:
        response = agent.invoke({"input": question})["output"]
        memory.save_context({"input": question}, {"output": response})
        print(f"\n🤖 ChatBot: {response}\n")
        return response
    except Exception as e:
        error_response = (
            "Thought: There was an error, but I can respond directly.\n"
            "Action: Direct Response\n"
            f"Action Input: I apologize for the error. How can I help you today?"
        )
        memory.save_context({"input": question}, {"output": error_response})
        print(f"\n🤖 ChatBot: {error_response}\n")
        return error_response

def chat_loop():
    """Runs a continuous chat in the terminal, allowing follow-ups."""
    print("💬 ChatBot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("🟢 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        handle_chat(user_input)

if __name__ == "__main__":
    chat_loop()