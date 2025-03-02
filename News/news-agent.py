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


#summary_llm = ChatOpenAI(base_url='https://openrouter.ai/api/v1', model='meta-llama/llama-3.3-70b-instruct:nitro', api_key=SecretStr(openRouterKey))
agent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
#deep_search_llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-thinking-exp-01-21', api_key=SecretStr(geminiKey))
summary_llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key))
deep_search_llm = ChatOpenAI(model='o1-mini', api_key=SecretStr(api_key))