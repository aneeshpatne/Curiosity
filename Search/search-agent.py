import os
import random
import asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, SecretStr
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # if needed
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.retry import RetryWithErrorOutputParser
from langchain.memory import ConversationBufferMemory
from datetime import datetime

global_deep_summaries = []  
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
# Uncomment below to use a different LLM if needed
#summary_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='google/gemini-2.0-flash-lite-001', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
#deep_search_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp-01-21', api_key=SecretStr(geminiKey))
summary_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))
deep_search_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))
# ---------------------------------
# Global Variables and Settings
# ---------------------------------

memory = ConversationBufferMemory(memory_key="chat_history")
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.5414.120 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.198 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]
semaphore = asyncio.Semaphore(7)
ddgs = DDGS()  # DuckDuckGo Search instance

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
    result = ddgs.text(topic, max_results=max_results)
    links = [r['href'] for r in result]
    print(f"[INFO] Found {len(links)} links for query: '{topic}'")
    return links

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
            "Additionally, generate exactly 5 follow-up questions that the user might ask next, based on the summarized information. "
            "The follow up questions must be SEO friendly and should be relevant to the topic. "
            "Return the questions as a list.\n\n"
            "### Response Format (Must be valid JSON):\n"
            "content: \"The summarized content.\"\n"
            "moreQtn: [\"Question 1\", \"Question 2\", \"Question 3\", \"Question 4\", \"Question 5\"]\n"
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
            "Additionally, generate exactly 5 follow-up questions that the user might ask next, based on the summarized information. "
            "The follow up questions must be SEO friendly and should be relevant to the topic. "
            "Return the questions as a list.\n\n"
            "### Response Format (Must be valid JSON):\n"
            "content: \"The summarized content.\"\n"
            "moreQtn: [\"Question 1\", \"Question 2\", \"Question 3\", \"Question 4\", \"Question 5\"]\n"
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

# Define the global citation counter at the module level.
global_citation_counter = 0

async def scrape_contents_for_deep(topic: str, links: list) -> str:
    """
    Launches a headless browser via Playwright and concurrently scrapes all provided links.
    Returns concatenated scraped content with each result labeled with sequential citations.
    Uses a global citation counter to maintain numbering across calls.
    """
    global global_citation_counter
    print(f"[INFO] Starting to scrape contents for query: '{topic}'")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        # Use the global citation counter to number each scraped page.
        source_lines = []
        for content in scraped_contents:
            global_citation_counter += 1
            source_lines.append(f"[{global_citation_counter}] {content}")
        combined = "\n".join(source_lines)
        await browser.close()
    print(f"[INFO] Completed scraping for query: '{topic}'")
    return combined


async def deep_search(query: str, depth: int = 4, sid=None, msg_id=None, 
                      emit_status=None, emit_sources=None, links: list = None) -> dict:
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
        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, f"Searching links for query: '{query}'")
        links = await get_links(query, max_results=2)
    
    # Emit the source URLs for this level.
    if emit_sources and sid and msg_id:
        await emit_sources(sid, msg_id, links)
    
    if emit_status and sid and msg_id:
        await emit_status(sid, msg_id, f"Scraping content for query: '{query}'")
    # Use our dedicated function to scrape and number each URL's content.
    scraped_text = await scrape_contents_for_deep(query, links)
    
    if emit_status and sid and msg_id:
        await emit_status(sid, msg_id, f"Summarizing content for query: '{query}'")
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
        for follow_q in summary_obj.moreQtn[:2]:
            if emit_status and sid and msg_id:
                await emit_status(sid, msg_id, f"Recursing for follow-up question: '{follow_q}'")
            result["follow_up"][follow_q] = await deep_search(
                follow_q,
                depth=depth - 1,
                sid=sid,
                msg_id=msg_id,
                emit_status=emit_status,
                emit_sources=emit_sources
            )
    
    if emit_status and sid and msg_id:
        await emit_status(sid, msg_id, f"Completed deep search for query: '{query}'")
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
    prompt = PromptTemplate(
        template=(
            "Below is a compilation of query summaries:\n\n"
            "{combined_content}\n\n"
            "Please generate a comprehensive in-depth summary that fully explains the topics from all directions. "
            "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
            "You will output the summary in markdown format.\n\n"
            "### Instructions\n"
            "Organize the response into clear topics with appropriate headers, and include a concluding section that summarizes the key points. "
            "Ensure the response remains focused, preserving citation labels (e.g., [1], [2]) for accuracy. "
            "The sources should be separated and not combined like [1 2 3] but rather formatted individually as [1] [2] [3]. "
            "The citations should be placed in their own square brackets, and separated by ONLY SPACE no COMMA. "
            "Each and every point you make in the response should be backed by the sources. "
            "Dont mention the source in the response, only use the citation label. "
            "You will not say 'according to source 1', 'source 2 says', etc. "
            "Do not add citations at the end of the response or make a list of citations at the end.\n\n"
            "Produce a LONG response that addresses the query in depth and produce exactly 5 follow-up questions.\n\n"
            "User expects a readable long answer with citations preserved.\n\n"
            "Break down the response into clear sections with appropriate headers and subheaders.\n\n"
            "Dont write long paragraphs, break down the response into clear sections with appropriate headers and subheaders.\n\n"
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
    except Exception as e:
        final_summary_obj = SummaryFormat(
            content=f"Error generating final summary: {e}",
            moreQtn=[]
        )
        print(f"[ERROR] Final summary generation failed: {e}")
    
    return final_summary_obj


# -----------------------------------------------
# FASTAPI + SOCKET.IO SERVER
# -----------------------------------------------
from fastapi import FastAPI
import socketio
import uvicorn

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Helper functions to emit status and sources to the client.
async def emit_status(sid, msg_id, status_value):
    await sio.emit("status", {"id": msg_id, "status": status_value}, to=sid)

async def emit_sources(sid, msg_id, links):
    await sio.emit("sources", {"id": msg_id, "sources": links}, to=sid)

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

@sio.event
async def message(sid, data):
    msg_id = data.get("id")
    text = data.get("text")
    search_type = data.get("searchType", "normal")
    if not text:
        return

    # Emit initial status: 'waiting'
    await emit_status(sid, msg_id, "waiting")
    try:
        result = await follow_up(
            question=text,
            sid=sid,
            msg_id=msg_id,
            emit_status=emit_status,
            emit_sources=emit_sources,
            search_type=search_type
        )
        final_answer = f"{result.content}"
        moreQtn = getattr(result, "moreQtn", [])
        await sio.emit("message", {"id": msg_id, "text": final_answer, "moreQtn": moreQtn, "status": "finished"}, to=sid)
    except Exception as e:
        final_answer = f"Error processing message: {str(e)}"
        await sio.emit("message", {"id": msg_id, "text": final_answer, "status": "finished", "moreQtn": []}, to=sid)

# -----------------------------------------------
# follow_up Function (Used by the Socket.IO handler)
# -----------------------------------------------
async def follow_up(
    question: str,
    sid=None,
    msg_id=None,
    emit_status=None,
    emit_sources=None,
    search_type: str = "normal"
):
    """
    Processes a question by checking conversation history and deciding whether a web search is needed.
    If needed, it retrieves sources, scrapes content, summarizes, and updates conversation memory.
    Additionally, if the search type is "deep", it emits source links for every search level,
    calls deep_search with frequent status updates, and finally calls generate_final_summary.
    """
    if search_type == "deep":
        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "Deep search selected; starting deep search")
        # Retrieve and emit sources for this level
        links = await get_links(question, max_results=2)
        # Call deep_search with provided links to avoid duplicate search, ensuring nested calls emit their sources as well.
        _ = await deep_search(question, depth=4, sid=sid, msg_id=msg_id, emit_status=emit_status, emit_sources=emit_sources, links=links)
        final_summary_obj = await generate_final_summary()
        return final_summary_obj

    previous_conversations = memory.load_memory_variables({})
    decision_prompt = (
        f"Based on the conversation history: {previous_conversations}\n"
        f"and the new question: '{question}', determine whether the available information is sufficient to answer comprehensively.\n"
        "If there is any uncertainty or missing details, answer 'yes' to perform a WebSearch. Otherwise, answer 'no'."
    )
    decision = (await agent_llm.ainvoke(decision_prompt)).content.strip().lower()
    if "yes" in decision:
        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "searching")
        max_results = 25 if search_type == "pro" else 7
        links = await get_links(question, max_results)
        if emit_sources and sid and msg_id:
            await emit_sources(sid, msg_id, links)
        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "scraping")
        scraped_text = await scrape_contents(question, links)
        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "thinking")
        summary_obj = await summarize(scraped_text, question)
        response_str = f"{summary_obj.content}"
        memory.save_context({"input": question}, {"output": response_str})
        return summary_obj
    else:
        response_prompt = (
            f"Based on the conversation history: {previous_conversations}\n"
            f"answer the question: '{question}'."
        )
        response_obj = await agent_llm.ainvoke(response_prompt)
        response_str = response_obj.content
        memory.save_context({"input": question}, {"output": response_str})
        return SummaryFormat(content=response_str, moreQtn=[])

# -----------------------------------------------
# Run Socket.IO Server
# -----------------------------------------------
if __name__ == "__main__":
    uvicorn.run(socket_app, host="0.0.0.0", port=4000)
