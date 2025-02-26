import os
import random
import asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, SecretStr
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain.memory import ConversationBufferMemory

# ---------------------------------
# Load Environment Variables
# ---------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openRouterKey = os.getenv("OPEN_ROUTER_KEY")
geminiKey = os.getenv("GEMINI_API_KEY")

# ---------------------------------
# Initialize LLMs
# ---------------------------------
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
summary_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))

# ---------------------------------
# Global Variables and Settings
# ---------------------------------
global_sources = []  # Global list to record each query and its summary.
global_scrape_counter = 0  # Global counter for scraped sources.
memory = ConversationBufferMemory(memory_key="chat_history")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.198 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

semaphore = asyncio.Semaphore(7)

# ---------------------------------
# Pydantic Model for Summarization Output
# ---------------------------------
class SummaryFormat(BaseModel):
    content: str = Field(description="The summarized content.")
    moreQtn: list[str] = Field(description="List of 5 follow-up questions based on the content.")

parser = PydanticOutputParser(pydantic_object=SummaryFormat)

# ---------------------------------
# Helper Functions
# ---------------------------------
def record_source(query: str, summary: str) -> int:
    """
    Records the query and its summary in the global_sources list.
    The source index starts at 1 and increments based on the number of sources recorded.
    Returns the index of the newly recorded source.
    """
    global global_sources
    source_index = len(global_sources) + 1  # Start at 1.
    global_sources.append({
        "source_index": source_index,
        "query": query,
        "summary": summary
    })
    print(f"[INFO] Recorded source [{source_index}] for query: '{query}'")
    return source_index

async def scrape_page(context, url: str) -> str:
    """
    Scrapes a single page while blocking images, stylesheets, and fonts.
    Returns the first 5000 characters of gathered text.
    """
    async with semaphore:
        page = await context.new_page()

        async def block_requests(route):
            if route.request.resource_type in ["image", "stylesheet", "font"]:
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", block_requests)
        try:
            print(f"[INFO] Scraping URL: {url}")
            await page.goto(url, wait_until='domcontentloaded', timeout=10000)
            text_blocks = await page.locator("body p, body h1, body h2, body h3, body h4, body h5, body h6").all_text_contents()
            cleaned_text = "\n".join([t.strip() for t in text_blocks if t.strip()])
            important_text = cleaned_text[:5000]
            return important_text if important_text else "No text found"
        except Exception as e:
            return f"Error scraping page: {str(e)}"
        finally:
            await page.close()

async def get_links(topic: str, max_results: int = 7) -> list:
    """
    Uses DuckDuckGo to fetch search results links for the given topic.
    """
    print(f"[INFO] Searching for links related to query: '{topic}'")
    ddgs = DDGS()
    result = ddgs.text(topic, max_results=max_results)
    links = [r['href'] for r in result]
    print(f"[INFO] Found {len(links)} links for query: '{topic}'")
    return links

async def scrape_contents(topic: str, links: list) -> str:
    """
    Launches a browser via Playwright, scrapes all provided links concurrently,
    and returns concatenated scraped content with each result enumerated using a global counter.
    """
    global global_scrape_counter
    print(f"[INFO] Starting to scrape contents for query: '{topic}'")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        source_lines = []
        for content in scraped_contents:
            global_scrape_counter += 1
            source_lines.append(f"[{global_scrape_counter}] {content}")
        combined = "\n".join(source_lines)
        await browser.close()
    print(f"[INFO] Completed scraping for query: '{topic}'")
    return combined

async def summarize(content: str, query: str) -> SummaryFormat:
    """
    Summarizes the gathered content using the provided query.
    Generates exactly 5 follow-up questions from the summarized content.
    Returns a SummaryFormat object.
    """
    print(f"[INFO] Summarizing content for query: '{query}'")
    prompt = PromptTemplate(
        template=(
            "The user has submitted the following query: '{query}'. Using the provided sources: {content}, "
            "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
            "You will output the summary in markdown format.\n\n"
            "### Instructions\n"
            "Organize the response into clear topics with appropriate headers, and include a concluding section that summarizes the key points. "
            "Ensure the response remains focused, preserving citation labels (e.g., [1], [2]) for accuracy. "
            "The sources should be separated and not combined like [1 2 3] but rather formatted individually as [1] [2] [3]. "
            "The citations should be placed in their own square brackets, and separated by ONLY SPACE no COMMA. "
            "Dont mention the source in the response, only use the citation label. "
            "you will not say 'according to source 1', 'source 2 says', etc. "
            "Do not add citations at the end of the response or make a list of citations at the end.\n\n"
            "Additionally, generate exactly 5 follow-up questions that the user might ask next, based on the summarized information. "
            "Return the questions as a list.\n\n"
            "### Response Format (Must be valid JSON):\n"
            "content: \"The summarized content.\"\n"
            "moreQtn: [\"Question 1\", \"Question 2\", \"Question 3\", \"Question 4\", \"Question 5\"]\n"
            "{format_instructions}"
        ),
        input_variables=["query", "content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    base_chain = prompt | summary_llm | StrOutputParser()
    retry_parser = RetryWithErrorOutputParser(
        parser=parser,
        retry_chain=base_chain,
        max_retries=3
    )
    inputs = {"query": query, "content": content}
    try:
        response = await base_chain.ainvoke(inputs)
        structured_output = retry_parser.parse_with_prompt(
            completion=response, prompt_value=prompt
        )
        print(f"[INFO] Summarization completed for query: '{query}'")
    except Exception as e:
        structured_output = SummaryFormat(content=f"Error parsing structured output. {e}", moreQtn=[])
        print(f"[ERROR] Summarization failed for query: '{query}'")
    return structured_output

# ---------------------------------
# Deep Search and Final Summary Functions
# ---------------------------------
async def deep_search(query: str, depth: int = 2) -> dict:
    """
    Recursively performs a deep search for the given query.
    
    Retrieves the top 3 links, scrapes and summarizes the content,
    records the query and summary, and recursively processes each of the 5 follow-up questions.
    """
    print(f"[INFO] Starting deep search for query: '{query}' with depth: {depth}")
    links = await get_links(query, max_results=3)
    scraped_text = await scrape_contents(query, links)
    summary_obj = await summarize(scraped_text, query)
    source_index = record_source(query, summary_obj.content)

    result = {
        "query": query,
        "summary": summary_obj.content,
        "source_index": source_index,
        "follow_up": {}  # Will hold recursive results.
    }

    if depth > 1:
        for follow_q in summary_obj.moreQtn:
            print(f"[INFO] Recursing for follow-up question: '{follow_q}'")
            result["follow_up"][follow_q] = await deep_search(follow_q, depth=depth - 1)
    print(f"[INFO] Completed deep search for query: '{query}'")
    return result

async def generate_final_summary() -> SummaryFormat:
    """
    Uses the global_sources list to generate an in-depth final summary.
    
    Combines all recorded query-summary pairs (with citations) and passes them to the summarization function.
    The prompt explicitly instructs the LLM to preserve all citation labels exactly as provided in the content.
    """
    print("[INFO] Generating final summary from all sources...")
    combined_content = ""
    for source in global_sources:
        combined_content += (
            f"[{source['source_index']}] Query: {source['query']}\n"
            f"Summary: {source['summary']}\n\n"
        )
    
    # Create a prompt that instructs the summarizer to maintain all citation labels.
    prompt = PromptTemplate(
        template=(
            "Below is a compilation of query summaries with their corresponding source citations. "
            "Ensure that your final answer preserves all citation labels exactly as they appear in the content. \n\n"
            "{combined_content}\n\n"
            "Based on the above, please generate a comprehensive in-depth summary that cites all the sources using their citation numbers. "
            "The final answer must include all unique citation numbers as provided in the content and be well-organized in markdown format."
            "\n\n### Response Format (Must be valid JSON):\n"
            "content: \"The final summarized content with citations preserved\"\n"
            "moreQtn: [\"Follow-up question 1\", \"Follow-up question 2\", \"Follow-up question 3\", \"Follow-up question 4\", \"Follow-up question 5\"]\n"
        ),
        input_variables=["combined_content"],
    )
    
    base_chain = prompt | summary_llm | StrOutputParser()
    retry_parser = RetryWithErrorOutputParser(
        parser=parser,
        retry_chain=base_chain,
        max_retries=3
    )
    inputs = {"combined_content": combined_content}
    try:
        response = await base_chain.ainvoke(inputs)
        final_summary_obj = retry_parser.parse_with_prompt(completion=response, prompt_value=prompt)
        print("[INFO] Final summary generation complete.")
    except Exception as e:
        final_summary_obj = SummaryFormat(content=f"Error generating final summary: {e}", moreQtn=[])
        print(f"[ERROR] Final summary generation failed: {e}")
    
    return final_summary_obj


# ---------------------------------
# Main Routine to Run the Code
# ---------------------------------
async def main():
    query = input("Enter your query: ")
    print("\n[START] Running deep search...\n")
    deep_results = await deep_search(query, depth=2)
    print("\n[RESULT] Deep Search Results:")
    print(deep_results)

    print("\n[START] Generating final in-depth summary with citations...\n")
    final_summary_obj = await generate_final_summary()
    print("\n[FINAL SUMMARY]")
    print(final_summary_obj.content)

if __name__ == "__main__":
    asyncio.run(main())
