# Simple LLM Service

A lightweight service for working with LLMs, built with FastAPI and Streamlit. This project provides a simple and clean approach to building LLM-powered applications without the complexity of frameworks like LangChain or LangGraph.

## Features

- **Direct LLM Integration**: Connect directly to OpenAI, Anthropic, and Google Gemini models
- **FastAPI Backend**: Fast and modern API service with async support
- **Streamlit Frontend**: User-friendly chat interface
- **Streaming Support**: Real-time token streaming for better user experience
- **Docker Support**: Easy deployment with Docker Compose
- **Conversation History**: Track and share conversation threads

## Getting Started

### Prerequisites

- Python 3.13+ (required by project configuration)
- At least one LLM API key (OpenAI, Anthropic, or Google)
- UV package manager (`pip install uv`)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/simple-llm-service.git
   cd simple-llm-service
   ```

2. Create a `.env` file with your API keys:
   ```
   cp .env.example .env
   ```
   Edit the `.env` file to add your API keys.

3. Create virtual environment and install dependencies:
   ```
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   uv sync
   ```

### Running Locally

1. Start the FastAPI service:
   ```
   python src/run_service.py
   ```

2. In a separate terminal, start the Streamlit app:
   ```
   streamlit run src/streamlit_app.py
   ```

3. Open your browser and navigate to `http://localhost:8501` to interact with the UI.

### Running with Docker

1. Build and start the services:
   ```
   docker compose up --build
   ```

2. For development with automatic reloading:
   ```
   docker compose watch
   ```

3. Access the Streamlit app at `http://localhost:8501`.

## Project Structure

```
simple-llm-service/
│
├── src/
│   ├── core/               # Core functionality
│   │   ├── settings.py     # Configuration settings
│   │   └── llm.py          # LLM integration
│   │
│   ├── schema/             # Data models
│   │   └── schema.py       # Pydantic models
│   │
│   ├── service/            # API service
│   │   └── service.py      # FastAPI service
│   │
│   ├── client/             # Client for the API
│   │   └── client.py       # API client
│   │
│   ├── run_service.py      # Script to run the API service
│   └── streamlit_app.py    # Streamlit frontend
│
├── docker/                 # Docker configuration
│   ├── Dockerfile.service  # Dockerfile for API service
│   └── Dockerfile.app      # Dockerfile for Streamlit app
│
├── compose.yaml            # Docker Compose configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Customization

This project is designed to be simple and extensible. Here are some ways to customize it:

- Add new LLM providers in `src/core/llm.py`
- Extend the API in `src/service/service.py`
- Modify the chat interface in `src/streamlit_app.py`
- Add authentication or rate limiting to the API

## License

This project is licensed under the MIT License - see the LICENSE file for details.