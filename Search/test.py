import os
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from dotenv import load_dotenv
from pydantic import SecretStr
# Load environment variables from a .env file if present
load_dotenv()

# Retrieve your API key from the environment
api_key = os.getenv("GROQ_KEY")
if not api_key:
    raise ValueError("GROQ_KEY is not set in the environment variables.")

# Initialize the ChatOpenAI LLM with your API details
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.2-1b-preview",
    api_key=SecretStr(api_key)
)

def main():
    prompt = "Tell me a short joke."
    # Send the prompt to the LLM
    response = llm(prompt)
    print("Response:", response)

if __name__ == "__main__":
    main()
