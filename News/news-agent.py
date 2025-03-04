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
#deep_search_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='google/gemini-2.0-flash-001', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
#deep_search_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp-01-21', api_key=SecretStr(geminiKey))
# summary_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
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
link_count = 0
async def get_links(topic: str, max_results: int = 7) -> list:
    global link_count
    """
    Uses DuckDuckGo to fetch search result links for the given topic.
    """
    print(f"[INFO] Searching for links related to query: '{topic}'")
    retries = 0
    max_retries = 20
    while retries < max_retries:
        try:
            result = ddgs.text(topic, max_results=max_results)
            links = [r['href'] for r in result]
            print(f"[INFO] Found {len(links)} links for query: '{topic}'")
            link_count += max_results
            return links
        except Exception as e:
            retries += 1
            print(f"[ERROR] DuckDuckGo rate limit reached. Attempt {retries}/{max_retries}. Waiting for 3 minutes... {e}")
            wait_time = random.randint(60, 300)
            print(f"[INFO] Waiting for {wait_time // 60} minutes and {wait_time % 60} seconds before retrying...")
            await asyncio.sleep(wait_time)
    print(f"[ERROR] Failed to retrieve links after {max_retries} attempts.")
    return []
        
    

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
            "You are summarizing a news"
            "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
            "You will output the summary in markdown format.\n\n"
            "### Instructions\n"
            "Organize the response into clear topics with appropriate headers, and include a concluding section that summarizes the key points. "
            "Dont mention the source in the response, or use the citation label. "
            "You will not say 'according to source 1', 'source 2 says', etc. "
            "Do not add citations at the end of the response or make a list of citations at the end.\n\n"
            "Additionally, generate exactly 20 follow-up questions that go to the root of the problem. "
            "The follow-up questions should be related to those news articles. "
            "The follow-up questions should build upon the news, and provide more follow up about the news"
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
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            # 1) Format the prompt to get a string or PromptValue
            formatted_prompt_text = prompt.format(**inputs)  # a string
            response = await base_chain.ainvoke(inputs)
            structured_output = retry_parser.parse_with_prompt(
                completion=response,
                prompt_value=formatted_prompt_text
            )
            print(f"[INFO] Summarization completed for query: '{query}'")
            return structured_output
        except Exception as e:
            print(f"[ERROR] Summarization failed for query: '{query}' after {retry_count + 1} retries: {e}")
            retry_count += 1
            await asyncio.sleep(180)
    structured_output = SummaryFormat(
        content=f"Error parsing structured output, after max retries. Please check the logs for more details.",
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
    
    prompt = (
            "Below is a compilation of query summaries:\n\n"
            f"data: {combined_content}\n\n"
            "If data is empty, end your response NOW. Telling user to check Cron logs."
            f"Today is {datetime.now().strftime('%Y-%m-%d')}"
            "Mention the date if the user is query is news related. "
            "Dont fixate on a specific topic its general news so explain all the topics, nicely. "
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
            "You can not use own knowledge and experience to the response.\n\n"
            "This is a deep research, user expects a detailed and nuanced answer. "
            "Don't just provide a simple answer, provide a detailed and nuanced answer.\n\n"
            "For each provided query and its summary, generate a comprehensive summary that covers all the topics in depth. "
            "User expects a readable long answer with no citations preserved.\n\n"
            "Break down the response into clear sections with appropriate headers and subheaders.\n\n"
            "Dont write long paragraphs, break down the response into clear sections with appropriate headers and subheaders.\n\n"
            
    )
    while retry_count < max_retries:
        try:
            # formatted_prompt_text = prompt.format(**inputs)
            response = await deep_search_llm.ainvoke(prompt)
            # print("[DEBUG] Raw response from LLM:")
            # print(response if response.strip() else "Empty response")
            # final_summary_obj = retry_parser.parse_with_prompt(
            #     completion=response,
            #     prompt_value=formatted_prompt_text
            # )
            print("[INFO] Final summary generation complete.")
            return response.content
        except Exception as e:
            print(f"[ERROR] Final summary generation failed: {e}")
            retry_count += 1
    print(f"[ERROR] Final summary generation failed even after {max_retries} retries.")
    return "The output could not be generated."
async def sendMail(html_content):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = f"Curiosity Daily Global Updates:  {datetime.now().strftime('%d-%m-%Y')}"
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
    query = "Global News"
    links = await get_links(query, max_results=20)
    _ = await deep_search(links, depth=2,links=links)
    final_summary_obj = await generate_final_summary()
    html_body = markdown.markdown(final_summary_obj)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Preview</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }}
            .email-container {{
                max-width: 650px;
                background: white;
                margin: auto;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
            }}
            h2 {{
                color: #1a73e8;
                text-align: center;
                margin-top: 0;
                margin-bottom: 20px;
                font-size: 22px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }}
            h3 {{
                color: #333;
                font-size: 18px;
                margin-top: 25px;
                margin-bottom: 15px;
                border-left: 4px solid #1a73e8;
                padding-left: 10px;
            }}
            p {{
                color: #555;
                font-size: 16px;
                margin-bottom: 15px;
            }}
            ul {{
                background: #f9fafc;
                padding: 20px 20px 20px 40px;
                border-radius: 8px;
                border-left: 3px solid #dfe1e5;
                margin: 20px 0;
            }}
            li {{
                margin: 10px 0;
                color: #444;
            }}
            .highlight {{
                background: #fff6dd;
                padding: 3px 6px;
                border-radius: 4px;
                font-weight: 500;
                color: #d67d00;
            }}
            a {{
                color: #1a73e8;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .button {{
                display: block;
                width: fit-content;
                margin: 25px auto;
                padding: 12px 25px;
                background: #1a73e8;
                color: white !important;
                text-decoration: none;
                border-radius: 6px;
                font-weight: bold;
                text-align: center;
                transition: background 0.3s ease;
            }}
            .button:hover {{
                background: #0d5bbd;
                text-decoration: none;
            }}
            blockquote {{
                border-left: 4px solid #ddd;
                padding: 10px 15px;
                margin: 20px 0;
                background: #f9f9f9;
                font-style: italic;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: monospace;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                margin: 15px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f8ff;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .footer {{
                text-align: center;
                font-size: 13px;
                color: #777;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="email-container">
            <h2>📢 Curiosity Daily News</h2>
            <p>Curiosity Scraped, Read and summarised {link_count} pages today</p>
            {html_body}
            <p class="footer">Never Stop fighting!</p>
        </div>
    </body>
    </html>
    """
    await sendMail(html_content)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html",  encoding="utf-8") as f:
        f.write(html_content)
        temp_file_path = f.name 
    webbrowser.open(f"file://{temp_file_path}")
    print(final_summary_obj)
    return final_summary_obj
if __name__ == "__main__":
    asyncio.run(main())