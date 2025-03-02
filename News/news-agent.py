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
from datetime import datetime
import markdown
import webbrowser
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openRouterKey = os.getenv("OPEN_ROUTER_KEY")
geminiKey = os.getenv("GEMINI_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# ---------------------------------
# LLM Options
summary_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='google/gemini-2.0-flash-lite-001', api_key=SecretStr(openRouterKey))
deep_search_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='google/gemini-2.0-flash-lite-001', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
#deep_search_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp-01-21', api_key=SecretStr(geminiKey))
# summary_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
# deep_search_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))

# ---------------------------------
# User Agent options to avoid blocking.
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.198 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]

semaphore = asyncio.Semaphore(7)
ddgs = DDGS()
global_deep_summaries = []

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
        combined = "\n".join(scraped_contents)
        await browser.close()
    print(f"[INFO] Completed scraping for query: '{topic}'")
    return combined

async def summarize(content: str, query: str) -> SummaryFormat:
    """
    Summarizes the provided content using the given query.
    Generates a markdown-formatted summary with exactly 20 follow-up questions.
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
            "Dont mention the source in the response, or use the citation label. "
            "You will not say 'according to source 1', 'source 2 says', etc. "
            "Do not add citations at the end of the response or make a list of citations at the end.\n\n"
            "Additionally, generate exactly 20 follow-up questions that go to the root of the problem. "
            "The follow up questions should break down the summarized content and then go a step back to generate the follow-up questions. "
            "The follow up questions should essentially point to questions that can be answered by articles not research papers. "
            "These questions are the questions that might arise after reading the summarized content. "
            "Breakdown the summrized content and then go a step back to generate the follow-up questions. "
            "The questions should be like 'What is the impact of X on Y?' or 'How does X affect Y?' or 'How are X?' "
            "Each follow-up question should be self-contained and formulated as a complete search query that would yield relevant results when entered directly into a search engine."
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


async def deep_search(query: str, depth: int = 2, links: list = None) -> dict:
    """
    Recursively performs a deep search for the given query with frequent status updates.
    If 'links' is provided, it is used instead of fetching links again.
    Emits source links (URLs) as soon as they are retrieved at every level.
    For each query, it scrapes via scrape_contents_for_deep and then summarizes the results.
    The query and its summary are stored (un-numbered) in a global list.
    Returns a dictionary with keys: query, summary, and follow_up.
    """
    # Fetch links if not provided.
    if links is None:
        print("Fetching links for query: ", query)
        links = await get_links(query, max_results=5)
    scraped_text = await scrape_contents(query, links)
    print("Sending to summarizer, ", scraped_text)
    summary_obj = await summarize(scraped_text, query)
    # Save the query and its summary (without extra numbering) to our global deep summaries.
    global_deep_summaries.append({
        "query": query,
        "summary": summary_obj.content
    })
    
    result = {
        "query": query,
        "summary": summary_obj.content,
        "follow_up": {}
    }
    
    # Recurse if more depth is required.
    if depth > 1:
        for follow_q in summary_obj.moreQtn:
            print(f"Deep searching follow-up query: '{follow_q}'")
            result["follow_up"][follow_q] = await deep_search(
                follow_q,
                depth=depth - 1,
            )
    print(f"Deep search completed for query: '{query}'")
    return result

async def generate_final_summary() -> SummaryFormat:
    """
    Uses the global_deep_summaries list to generate an in-depth final summary.
    Combines all recorded query-summary pairs (without added citation numbers) and passes them to the summarization function.
    Returns a SummaryFormat object.
    """
    print("[INFO] Generating final summary from all query summaries...")
    combined_content = ""
    for entry in global_deep_summaries:
        # No numbering is added here to avoid confusing the LLM.
        combined_content += (
            f"Query: {entry['query']}\n"
            f"Summary: {entry['summary']}\n\n"
        )
    
    # (Optional) Save combined content to a text file for debugging.
    try:
        with open('combined_sources.txt', 'w', encoding='utf-8') as f:
            f.write(combined_content)
        print("[INFO] Saved combined query summaries to combined_sources.txt")
    except Exception as e:
        print(f"[ERROR] Failed to save combined sources: {e}")
    # print(combined_content)
    max_retries = 3
    retry_count = 0
    
    prompt = PromptTemplate(
        template=(
            "Below is a compilation of query summaries:\n\n"
            "{combined_content}\n\n"
            f"Today is {datetime.now().strftime('%Y-%m-%d')}"
            "Mention the date if the user is query is news related. "
            "Please generate a comprehensive in-depth summary that fully explains the topics from all directions. "
            "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
            "You will output the summary in markdown format.\n\n"
            "### Instructions\n"
            "Organize the response into clear topics with appropriate headers, and include a concluding section that summarizes the key points. "
            "Do not mention the source in the response, and dont use the citation label. "
            "Do not add citations at the end of the response or make a list of citations at the end.\n\n"
            "Produce a LONG response that addresses the query in depth and produce exactly 5 follow-up questions.\n\n"
            "Produce a nuanced answer that covers all aspects of the query. "
            "Take a step back and think about the query from all angles. "
            "You're free to add your own knowledge and experience to the response.\n\n"
            "This is a deep research, user expects a detailed and nuanced answer. "
            "Don't just provide a simple answer, provide a detailed and nuanced answer.\n\n"
            "The context may or may not have all the information needed to answer the query, at that time you can use your own knowledge and experience to answer the query. "
            "For each provided query and its summary, generate a comprehensive summary that covers all the topics in depth. "
            "If you are capable of adding in your own knowledge and experience to the response, please do so, when you think it is necessary. "
            "Your experience should not be used to answer the query, but to add depth to the response. "
            "For the things that can change over time, you should not use your experience to answer the query. "
            "User expects a readable long answer with no citations preserved.\n\n"
            "Break down the response into clear sections with appropriate headers and subheaders.\n\n"
            "Dont write long paragraphs, break down the response into clear sections with appropriate headers and subheaders.\n\n"
            "DONT FORGET TO RETURN THE RESPONSE IN JSON FORMAT.\n\n"
            "### Response Format (Must be valid JSON):\n"
            "content: \"The final summarized content with citations preserved\"\n"
            "moreQtn: [\"Follow-up question 1\", \"Follow-up question 2\", \"Follow-up question 3\", \"Follow-up question 4\", \"Follow-up question 5\"]\n"
            "{format_instructions}"
        ),
        input_variables=["combined_content"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    base_chain = prompt | deep_search_llm | StrOutputParser()
    retry_parser = RetryWithErrorOutputParser(
        parser=parser,
        retry_chain=base_chain,
        max_retries=3
    )
    inputs = {"combined_content": combined_content}
    while retry_count < max_retries:
        try:
            formatted_prompt_text = prompt.format(**inputs)
            response = await base_chain.ainvoke(inputs)

            # print("[DEBUG] Raw response from LLM:")
            # print(response if response.strip() else "Empty response")
            final_summary_obj = retry_parser.parse_with_prompt(
                completion=response,
                prompt_value=formatted_prompt_text
            )
            print("[INFO] Final summary generation complete.")
            return final_summary_obj
        except Exception as e:
            print(f"[ERROR] Final summary generation failed: {e}")
            retry_count += 1
    print(f"[ERROR] Final summary generation failed even after {max_retries} retries.")
    final_summary_obj = SummaryFormat(
        content=f"Error generating final summary:",
        moreQtn=[]
    )
    return final_summary_obj
async def sendMail(html_content):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = "Curiosity Testing"
    msg.attach(MIMEText(html_content, "html"))
    try:
        # Establish SMTP connection
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Secure connection
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

        print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ Error sending email: {e}")

async def main():
    query = "Latest Global News"
    links = await get_links(query, max_results=2)
    _ = await deep_search(links, depth=1,links=links)
    final_summary_obj = await generate_final_summary()
    html_content = markdown.markdown(final_summary_obj.content)
    await sendMail(html_content)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
        f.write(html_content)
        temp_file_path = f.name 
    webbrowser.open(f"file://{temp_file_path}")
    print(final_summary_obj)
    return final_summary_obj
if __name__ == "__main__":
    asyncio.run(main())