from langchain_ollama import ChatOllama
from browser_use import Agent
from pydantic import SecretStr
import asyncio

async def main():
    llm = ChatOllama(model="mistral:latest", num_ctx=32000)
    agent = Agent(
        task="Go to Reddit, search for 'browser-use', click on the first post and return the first comment.",
        llm = llm
    )
    result = await agent.run()
    print(result)
asyncio.run(main())
