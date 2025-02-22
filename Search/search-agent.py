from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
import os
from playwright.async_api import async_playwright
import asyncio
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain.schema import OutputParserException
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openRouterKey = os.getenv("OPEN_ROUTER_KEY")

# Initialize LLMs
summary_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='google/gemini-2.0-flash-001', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))

class SummaryFormat(BaseModel):
    content: str = Field(description="The summarized content.")
    moreQtn: list[str] = Field(description="List of 5 follow-up questions based on the content.", min_items=5, max_items=5)
parser = PydanticOutputParser(pydantic_object=SummaryFormat)


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
        important_text = cleaned_text[:3000]
        return important_text if important_text else "No text found"
    except Exception as e:
        return f"Error scraping page: {str(e)}"
    finally:
        await page.close()

async def searchAgent(topic):
    """Searches for the topic, scrapes data, and summarizes it."""
    result = ddgs.text(topic, max_results=3)
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
        print("Scraped Contents")
        summary = await summarize(ls, topic)
        await browser.close()
        return summary

async def summarize(content: str, query: str):
    """Summarizes the gathered content."""
    prompt = PromptTemplate(
        template=(
            "The user has submitted the following query: '{query}'. Using the provided sources: {content}, "
            "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
            "Ensure the response remains focused, preserving citation labels (e.g., [1], [2]) for accuracy. "
            "The sources should be separated and not combined like [1,2,3] but rather formatted individually as [1], [2], [3]. "
            "Some sources may contain overlapping or redundant information; synthesize the data to avoid repetition. "
            "If the sources lack sufficient data to fully address the query, conclude with: 'Insufficient relevant information found.'\n\n"
            "Additionally, generate exactly 5 follow-up questions that the user might ask next, based on the summarized information. "
            "Return the questions as a list.\n\n"
            "### Response Format (Must be valid JSON)\n"
            "{format_instructions}"
        ),
        input_variables=["query", "content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    base_chain = prompt | summary_llm | StrOutputParser()
    retry_parser = RetryWithErrorOutputParser(
        parser=parser,
        retry_chain=base_chain,  # Required in 0.3.x
        max_retries=3
    )
    inputs = {"query": query, "content": content}
    response = await base_chain.ainvoke(inputs)
    structured_output = retry_parser.parse_with_prompt(completion=response, prompt_value=prompt)
    return structured_output

    
async def follow_up(question):
    """Handles questions using conversation history or web search."""
    previous_conversations = memory.load_memory_variables({})
    decision_prompt = (
        f"Based on the conversation history: {previous_conversations}\n"
        f"and the new question: '{question}', determine whether the available information is sufficient to answer comprehensively.\n"
        "If there is any uncertainty, missing details, or if a more up-to-date response could be beneficial, answer 'yes' to perform a WebSearch. Otherwise, answer 'no'."
    )

    decision = (await agent_llm.ainvoke(decision_prompt)).content.strip().lower()
    if "yes" in decision:
        print("Using WebSearch to get the answer...")
        response = await searchAgent(question)
    else:
        answer_prompt = (
            f"Based on the conversation history: {previous_conversations}\n"
            f"answer the question: '{question}'."
        )
        response = (await agent_llm.ainvoke(answer_prompt)).content
    memory.save_context({"input": question}, {"output": response})
    print("Response:", response)
    return response

async def main():
    """Interactive loop for asking questions."""
    print("Welcome to the AI Assistant! Type 'exit' to quit.")
    while True:
        user_query = input("Ask a question: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        result = await follow_up(user_query)
        print("Final Answer:", result)
        print("\n---\n")

# Example usage
if __name__ == "__main__":
    asyncio.run(main())