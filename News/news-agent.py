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