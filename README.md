# Curiosity: AI-Powered Search Agent

Curiosity is an AI-powered search agent that combines web search capabilities with a modern chat interface. The project uses LLMs to scrape, analyze, and summarize web content in response to user queries in real-time.

![Curiosity Bot](Frontend/curiosity/public/assets/bot.png)

## Project Structure

The repository is organized into two main components:

```
.
├── Frontend/
│   └── curiosity/        # Next.js frontend application
├── Search/               # Python backend search agent
│   ├── search-agent.py   # Main search agent implementation
│   ├── deep-search.py    # Deep search implementation
│   └── Deprecated/       # Older search implementations
├── News/
|   ├─ news-agent.py      # Global news Deep seach mailer
└── README.md
```

## Features

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

## Setup Instructions

### Backend Setup

1. Navigate to the Search directory:

   ```bash
   cd Search
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install langchain langchain-openai langchain-google-genai playwright python-socketio python-dotenv duckduckgo-search fastapi uvicorn pydantic
   ```

4. Install Playwright browsers:

   ```bash
   playwright install chromium
   ```

5. Create a `.env` file with your API keys:

   ```
   OPENAI_API_KEY=your_openai_key
   OPEN_ROUTER_KEY=your_openrouter_key
   GEMINI_API_KEY=your_gemini_key
   ```

6. Start the backend server:
   ```bash
   python search-agent.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd Frontend/curiosity
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. Open the Curiosity interface in your browser
2. Select a search type from the dropdown (Normal, Pro, or Deep)
3. Enter your query in the text field and press Enter or click Send
4. Watch as Curiosity searches the web, finds relevant sources, and generates a comprehensive answer
5. Follow-up questions will be displayed at the bottom of the response for further exploration

## How It Works

1. The frontend sends the user query to the backend via Socket.io
2. The backend performs a web search using DuckDuckGo
3. Retrieved URLs are sent back to the frontend in real-time
4. The backend scrapes content from the URLs using Playwright
5. Content is summarized using LLMs with appropriate citations
6. For deep searches, follow-up questions are also searched recursively
7. The final summarized result is sent back to the frontend
