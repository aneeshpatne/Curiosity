from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
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
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openRouterKey = os.getenv("OPEN_ROUTER_KEY")
geminiKey = os.getenv("GEMINI_API_KEY")

# Initialize LLMs
#summary_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='openai/gpt-4o-mini', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
#summary_llm = ChatGoogleGenerativeAI(model='gemini-2.0-pro-exp-02-05', api_key=SecretStr(geminiKey))
summary_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))
# gemini-2.0-pro-exp-02-05

class SummaryFormat(BaseModel):
    content: str = Field(description="The summarized content.")
    moreQtn: list[str] = Field(description="List of 5 follow-up questions based on the content.")

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
        important_text = cleaned_text[:5000]
        return important_text if important_text else "No text found"
    except Exception as e:
        return f"Error scraping page: {str(e)}"
    finally:
        await page.close()

async def summarize(content: str, query: str):
    """Summarizes the gathered content."""
    prompt = PromptTemplate(
        template=(
        "The user has submitted the following query: '{query}'. Using the provided sources: {content}, "
        "please craft a thorough, well-rounded, and detailed response that directly addresses the user's question with clarity and depth. "
        "You will output the summary in markdown format.\n\n"
        "### Instructions\n"
        "Organize the response into clear topics with appropriate headers, and include a concluding section that summarizes the key points. "
        "Ensure the response remains focused, preserving citation labels (e.g., [1], [2]) for accuracy. "
        "The sources should be separated and not combined like [1 ,2 ,3] but rather formatted individually as [1] [2] [3]. "
        "The citations should be placed in their own square brackets, and separated by ONLY SPACE no COMMA. "
        "Dont mention the source in the response, only use the citation label. "
        "you will not say 'according to source 1', 'source 2 says', etc. "
        "Do not add citations at the end of the response or make a list of citations at the end."
        "Some sources may contain overlapping or redundant information; synthesize the data to avoid repetition. "
        "If the sources lack sufficient data to fully address the query, conclude with: 'Insufficient relevant information found.'\n\n"
        "Additionally, generate exactly 5 follow-up questions that the user might ask next, based on the summarized information. "
        "Return the questions as a list.\n\n"
        "### Response Format (Must be valid JSON)\n with the following format:\n"
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
        retry_chain=base_chain,  # Required in 0.3.x
        max_retries=3
    )
    inputs = {"query": query, "content": content}
    try:
        response = await base_chain.ainvoke(inputs)
        print(response)
        structured_output = retry_parser.parse_with_prompt(
            completion=response, prompt_value=prompt
        )
    except Exception as e:
        structured_output = SummaryFormat(content=f"Error parsing structured output. {e}", moreQtn=[])
    return structured_output

async def get_links(topic):
    """Get source links for the given topic using DuckDuckGo search."""
    result = ddgs.text(topic, max_results=20)
    links = [r['href'] for r in result]
    return links

async def scrape_contents(topic, links):
    """Scrapes data from provided links and returns the concatenated sources string."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        tasks = [scrape_page(context, link) for link in links]
        scraped_contents = await asyncio.gather(*tasks)
        ls = "\n".join([f"[{index}] {content}" for index, content in enumerate(scraped_contents, start=1)])
        await browser.close()
    return ls

async def follow_up(
    question: str,
    sid=None,
    msg_id=None,
    emit_status=None,
    emit_sources=None
):
    """
    Handles questions using conversation history or web search.
    Emits statuses and sources as soon as they're discovered.
    """
    previous_conversations = memory.load_memory_variables({})
    decision_prompt = (
        f"Based on the conversation history: {previous_conversations}\n"
        f"and the new question: '{question}', determine whether the available information is sufficient to answer comprehensively.\n"
        "If there is any uncertainty, missing details, or if a more up-to-date response could be beneficial, answer 'yes' to perform a WebSearch. Otherwise, answer 'no'."
    )
    decision = (await agent_llm.ainvoke(decision_prompt)).content.strip().lower()

    if "yes" in decision:
        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "searching")

        # Immediately get and emit source links
        links = await get_links(question)
        if emit_sources and sid and msg_id:
            await emit_sources(sid, msg_id, links)

        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "scraping")

        # First, scrape the contents (this completes the scraping phase)
        scraped_text = await scrape_contents(question, links)

        if emit_status and sid and msg_id:
            await emit_status(sid, msg_id, "thinking")

        # Then call LLM summarization
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

# Cleaned-up main function (optional).
async def main():
    """Interactive loop for asking questions (CLI mode)."""
    pass
    # Uncomment for CLI usage:
    # print("Welcome to the AI Assistant! Type 'exit' to quit.")
    # while True:
    #     user_query = input("Ask a question: ").strip()
    #     if user_query.lower() == "exit":
    #         print("Goodbye!")
    #         break
    #     result = await follow_up(user_query)
    #     print("Final Answer:", result.content)
    #     print("Follow up Questions:", getattr(result, "moreQtn", "Not available"))
    #     print("\n---\n")


# -----------------------------------------------
#          FASTAPI + SOCKET.IO SERVER
# -----------------------------------------------
from fastapi import FastAPI
import socketio
import uvicorn

# Create a Socket.IO "AsyncServer" and FastAPI app
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Helper emit functions for statuses and sources
async def emit_status(sid, msg_id, status_value):
    """Emit a 'status' event to one specific client."""
    await sio.emit("status", {"id": msg_id, "status": status_value}, to=sid)

async def emit_sources(sid, msg_id, links):
    """Emit a 'sources' event to one specific client with the given links."""
    await sio.emit("sources", {"id": msg_id, "sources": links}, to=sid)

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

@sio.event
async def message(sid, data):
    """
    Expects data in the format:
      { "id": <message id>, "text": <user message> }
    """
    msg_id = data.get("id")
    text = data.get("text")
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
            emit_sources=emit_sources
        )
        final_answer = f"{result.content}"
    except Exception as e:
        final_answer = f"Error processing message: {str(e)}"

    # Emit the final answer with status: 'finished'
    await sio.emit(
        "message",
        {"id": msg_id, "text": final_answer, "status": "finished"},
        to=sid
    )

if __name__ == "__main__":
    # If you want to run the CLI loop instead, uncomment:
    # asyncio.run(main())
    
    uvicorn.run(socket_app, host="0.0.0.0", port=4000)
