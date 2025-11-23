# BrowseAgent

BrowseAgent is an intelligent search assistant that leverages large language models to provide contextual, relevant search results. It combines the power of SearXNG for privacy-respecting web searches with OpenRouter for advanced language model processing.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Frontend Development](#frontend-development)
- [Contributing](#contributing)
- [License](#license)

## Features

- Privacy-respecting web searches using SearXNG
- Integration with OpenRouter for advanced language models
- Session management with API key handling
- Model selection and configuration
- Rate limiting and usage tracking
- Comprehensive search result processing pipeline
- Semantic search result ranking
- Frontend with ChatGPT-like interface

## Architecture

BrowseAgent follows a modular architecture with clear separation of concerns:

- **Backend API**: FastAPI application handling all server-side logic
- **Search Tools**: SearXNG and DuckDuckGo integration for web searches
- **LLM Integration**: OpenRouter client for language model processing
- **Session Management**: In-memory session storage with auto-cleanup
- **Search Pipeline**: Comprehensive processing of search results
- **Frontend**: React-based ChatGPT-like interface

## Project Structure

```
browseagent/
├── app.py                 # Main FastAPI application
├── agent.py              # BrowseAgent implementation
├── search_tool.py        # Search tool implementations
├── search_result_pipeline.py # Search result processing pipeline
├── openrouter.py         # OpenRouter API integration
├── session_manager.py    # Session management
├── requirements.txt      # Backend dependencies
├── frontend/             # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── utils/        # Utility functions
│   │   └── services/     # API services
│   └── public/           # Public assets
├── tests/                # Test files
├── docker-compose.yml    # Docker configuration
├── Dockerfile            # Backend Docker configuration
├── Dockerfile.frontend   # Frontend Docker configuration
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend development)
- Docker and docker-compose (for containerized deployment)

### Backend Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd browseagent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   SITE_URL=https://your-site.com
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the frontend directory:
   ```env
   REACT_APP_API_BASE_URL=http://localhost:8000
   ```

## Usage

### Running the Backend

1. Start the backend server:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

### Running the Frontend

1. In a separate terminal, navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Start the development server:
   ```bash
   npm start
   ```

### Using Docker

1. Build and start the entire application:
   ```bash
   docker-compose up --build
   ```

The application will be available at `http://localhost:3000` (frontend) and `http://localhost:8000` (backend).

## API Endpoints

### Authentication & Sessions

- `POST /session/create` - Create a new session with API key
- `DELETE /session` - End current session
- `GET /session/info` - Get session information

### Query Processing

- `POST /query` - Process a query with the BrowseAgent
- `POST /validate-api-key` - Validate an API key
- `GET /models` - Get available models

### Account Information

- `GET /account` - Get account usage and balance
- `POST /model/select` - Update selected model
- `GET/POST /model/config` - Manage model configurations

### Information

- `GET /health` - Health check
- `GET /` - Root endpoint

## Frontend Development

The frontend is built with React and follows a component-based architecture. Key features include:

- ChatGPT-like conversation interface
- Real-time streaming responses
- Model selection dropdown
- API key management
- Responsive design for all devices

### Key Components

- `ChatInterface` - Main chat interface component
- `Message` - Individual message bubble component
- `SearchInput` - Input field with advanced options
- `ModelSelector` - Model selection component
- `SettingsPanel` - User settings and preferences

### Running Development Server

```bash
cd frontend
npm start
```

### Building for Production

```bash
cd frontend
npm run build
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.