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

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openRouterKey = os.getenv("OPEN_ROUTER_KEY")
geminiKey = os.getenv("GEMINI_API_KEY")

# ---------------------------------
# LLM Options
#summary_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='meta-llama/llama-3.3-70b-instruct:nitro', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
#deep_search_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp-01-21', api_key=SecretStr(geminiKey))
summary_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
deep_search_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))

# ---------------------------------
# User Agent options to avoid blocking.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.198 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

semaphore = asyncio.Semaphore(7)
ddgs = DDGS()


class SummaryFormat(BaseModel):
    content: str = Field(description="The summarized content.")
    moreQtn: list[str] = Field(description="List of n follow-up questions based on the content.")

parser = PydanticOutputParser(pydantic_object=SummaryFormat)

async def scrape_page(context, url: str) -> str:
    """
    Scrapes a single page while blocking images, stylesheets, and fonts.
    Returns up to the first 5000 characters of gathered text.
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
    Uses DuckDuckGo to fetch search result links for the given topic.
    """
    print(f"[INFO] Searching for links related to query: '{topic}'")
    while True:
        try:
            result = ddgs.text(topic, max_results=max_results)
            links = [r['href'] for r in result]
            print(f"[INFO] Found {len(links)} links for query: '{topic}'")
            return links
        except Exception as e:
            print(f"[ERROR] DuckDuckGo rate limit reached. Waiting for 3 minutes... {e}")
            await asyncio.sleep(180)
        
    

async def scrape_contents(topic: str, links: list) -> str:
    """
    Launches a headless browser via Playwright and concurrently scrapes all provided links.
    Returns concatenated scraped content with each result labeled with sequential citations.
    """
    print(f"[INFO] Starting to scrape contents for query: '{topic}'")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        source_lines = [f"[{i+1}] {content}" for i, content in enumerate(scraped_contents)]
        combined = "\n".join(source_lines)
        await browser.close()
    print(f"[INFO] Completed scraping for query: '{topic}'")
    return combined

async def summarize(content: str, query: str) -> SummaryFormat:
    """
    Summarizes the provided content using the given query.
    Generates a markdown-formatted summary with exactly 5 follow-up questions.
    Returns a SummaryFormat object.
    """
    print(f"[INFO] Summarizing content for query: '{query}'")

    prompt = PromptTemplate(
        template=(
            "The user has submitted the following query: '{query}'. Using the provided sources: {content}, "
            f"Today is {datetime.now().strftime('%Y-%m-%d')}"
            "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
            "You will output the summary in markdown format.\n\n"
            "### Instructions\n"
            "Organize the response into clear topics with appropriate headers, and include a concluding section that summarizes the key points. "
            "Ensure the response remains focused, preserving citation labels (e.g., [1], [2]) for accuracy. "
            "The sources should be separated and not combined like [1 2 3] but rather formatted individually as [1] [2] [3]. "
            "The citations should be placed in their own square brackets, and separated by ONLY SPACE no COMMA. "
            "Dont mention the source in the response, only use the citation label. "
            "You will not say 'according to source 1', 'source 2 says', etc. "
            "Do not add citations at the end of the response or make a list of citations at the end.\n\n"
            "Additionally, generate exactly 20 follow-up questions that go to the root of the problem. "
            "The follow up questions should essentially point to questions that can be answered by articles not research papers. "
            "These questions are the questions that might arise after reading the summarized content. "
            "Breakdown the summrized content and then go a step back to generate the follow-up questions. "
            "The questions should be like 'What is the impact of X on Y?' or 'How does X affect Y?' or 'How are X?' "
            "The follow up questions should be standalone as in they should generate appropriate answers, when that same query is pasted on a search engine. "
            "Return the questions as a list.\n\n"
            "### Response Format (Must be valid JSON):\n"
            "content: \"The summarized content.\"\n"
            "moreQtn: [\"Question 1\", \"Question 2\", \"Question 3\", \"Question 4\", \"Question 5\", \"Question 6\", "
            "\"Question 7\", \"Question 8\", \"Question 9\", \"Question 10\", \"Question 11\", \"Question 12\", "
            "\"Question 13\", \"Question 14\", \"Question 15\", \"Question 16\", \"Question 17\", \"Question 18\", "
            "\"Question 19\", \"Question 20\"]\n"
            "{format_instructions}"
        ),
        input_variables=["query", "content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Build the chain
    base_chain = prompt | summary_llm | StrOutputParser()
    retry_parser = RetryWithErrorOutputParser(
        parser=parser,
        retry_chain=base_chain,
        max_retries=3
    )

    # Prepare inputs
    inputs = {"query": query, "content": content}

    try:
        # 1) Format the prompt to get a string or PromptValue
        formatted_prompt_text = prompt.format(**inputs)  # a string
        response = await base_chain.ainvoke(inputs)
        structured_output = retry_parser.parse_with_prompt(
            completion=response,
            prompt_value=formatted_prompt_text
        )
        print(f"[INFO] Summarization completed for query: '{query}'")
    except Exception as e:
        structured_output = SummaryFormat(
            content=f"Error parsing structured output. {e}",
            moreQtn=[]
        )
        print(f"[ERROR] Summarization failed for query: '{query}'")
    return structured_output