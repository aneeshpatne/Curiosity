import os
import asyncio
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Controller, Agent
from pydantic import SecretStr

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Define the output format as a Pydantic model
class URLResults(BaseModel):
    urls: List[str]

# Initialize the controller with the desired output model
controller = Controller(output_model=URLResults)

async def main():
    task = (
        "Go to DuckDuckGo, search for 'Maharashtra Latest news', and return only the top 10 result URLs "
        "in JSON format. The JSON should follow this structure: { 'urls': [ 'url1', 'url2', ... ] }."
    )
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=SecretStr(api_key))
    # Create the Agent with the task, llm, and controller
    agent = Agent(task=task, llm=llm, controller=controller)
    history = await agent.run()
    result = history.final_result()
    print(result)

if __name__ == '__main__':
    asyncio.run(main())
