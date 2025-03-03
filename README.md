# Curiosity: AI-Powered Search Agent

Curiosity is an AI-powered search agent that combines web search capabilities with a modern chat interface. The project uses LLMs to scrape, analyze, and summarize web content in response to user queries in real-time.

![Curiosity Bot](Frontend/curiosity/public/assets/bot.png)

## Project Structure

The repository is organized into three main components:

```
.
├── Frontend/
│   └── curiosity/        # Next.js frontend application
├── Search/               # Python backend search agent
│   ├── search-agent.py   # Main search agent implementation
│   ├── deep-search.py    # Deep search implementation
│   └── Deprecated/       # Older search implementations
├── News/
|   ├─ news-agent.py      # Global news deep search mailer
|   ├─ run-newsletter.sh  # Shell script for automating newsletter execution
└── README.md
```

# Curiosity Search

- **Multiple Search Modes**:

  - **Normal Search**: Quick searches with standard depth
  - **Pro Search**: Enhanced search with more sources
  - **Deep Search**: Recursive search that explores follow-up questions

- **Real-time Updates**: See sources and progress as the agent works
- **Source Citations**: Results include references to source materials
- **Follow-up Questions**: Automatically generates relevant follow-up questions
- **Modern UI**: Clean, responsive interface with dark mode

## Tech Stack

### Frontend

- [Next.js](https://nextjs.org/) with React 19
- [Socket.io](https://socket.io/) for real-time communication
- [Tailwind CSS](https://tailwindcss.com/) for styling
- [shadcn/ui](https://ui.shadcn.com/) for UI components

### Backend

- Python with [FastAPI](https://fastapi.tiangolo.com/) and Socket.io
- [Playwright](https://playwright.dev/) for headless browsing and web scraping
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) for web searches
- [LangChain](https://langchain.readthedocs.io/) for LLM integration
- Multiple LLM support (OpenAI, Google Gemini, etc.)

## How It Works

1.  **Query Transmission**: The frontend sends user queries to the backend via Socket.io
2.  **Web Search**: Backend performs searches using DuckDuckGo API
3.  **Real-time Updates**: Retrieved URLs are streamed back to frontend instantly
4.  **Content Scraping**: Backend uses Playwright to extract content from URLs
5.  **AI Processing**: Content is analyzed and summarized by LLMs with source citations
6.  **Deep Search**: For comprehensive queries, follow-up questions are searched recursively
7.  **Result Delivery**: Final summarized content is sent back to frontend

# Curiosity Newsletter

Curiosity Newsletter is an automated daily global news summary generator that fetches the latest news, processes and summarizes it using AI, and then sends it as an email. It utilizes a **modified version of the deep search** algorithm to aggregate information efficiently from multiple sources. It leverages **DuckDuckGo Search**, **Playwright**, and **AI-powered summarization models** to provide detailed, structured, and insightful news updates.

## Features

- **Web Scraping**: Uses DuckDuckGo Search to find relevant news links.
- **Content Extraction**: Scrapes website content efficiently with Playwright.
- **AI Summarization**: Uses OpenAI and Gemini models to generate structured news summaries.
- **Email Automation**: Sends daily news updates directly to your inbox.
- **Markdown-to-HTML Conversion**: Formats the content into a visually appealing email.

## Technologies Used

- **Python**
- **Playwright** for web scraping
- **DuckDuckGo Search API** for fetching news links
- **LangChain & OpenAI API** for AI-driven summarization
- **Markdown & SMTP** for formatting and email delivery
- **Pydantic** for structured data parsing

## How It Works

1. **Fetch Links**: The script searches for top news links using DuckDuckGo.
2. **Scrape Content**: Uses Playwright to extract important text from the pages.
3. **AI Summarization**: Summarizes the news articles using LLMs.
4. **Format & Send**: Converts the summary into HTML and emails it to the receiver.

---
