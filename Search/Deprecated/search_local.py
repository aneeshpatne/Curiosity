from langchain_ollama import ChatOllama
from browser_use import Agent
from pydantic import SecretStr
import asyncio

# Load environment variables


async def main():
    llm = ChatOllama(model="llama3.2:1b")

    agent = Agent(
        task="Go to Reddit, search for 'browser-use', click on the first post and return the first comment.",
        llm=llm
    )

    result = await agent.run()  # Ensure Agent supports async execution
    print(result)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
