# Curiosity: Search Agent Project

This project implements multiple search and scraping functionalities using several LLMs (e.g., GPT variants, Gemini) and tools such as DuckDuckGo, Playwright, and LangChain. The agents can perform web searches, scrape page content, and generate detailed summaries with follow-up questions.

## Project Structure

- **Search/**
  - **.env**
  - [search-agent.py](Search/search-agent.py)  
     The primary agent script that performs a web search using DuckDuckGo, scrapes multiple pages, and summarizes the results along with generating follow-up questions.
- **Frontend/**  
   _(This directory will contain the frontend code for the project.)_

2. **Configure Environment Variables**  
   Create a `.env` file in the `Search/` folder with your API keys

## Usage

### Running the Main Agent

The primary entry point is `search-agent.py`. It allows an interactive loop where you can ask questions, and the agent will use web search and scraping to produce a detailed summary.

Run the agent with:

```sh
python search-agent.py
```

## Features

- **Web Search & Scraping**  
   Fetches URLs from DuckDuckGo and extracts text from web pages using Playwright.

- **Advanced Summarization**  
   Leverages LLMs from OpenAI and Google Gemini to create cohesive summaries with citation labels and generate follow-up questions.

- **Modular Design**  
   The project is organized using controllers, agents, and chains from LangChain, making it extensible for future improvements.
