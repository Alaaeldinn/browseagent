# BrowseAgent

A web-based AI agent that can browse the web, extract content, and provide intelligent responses based on retrieved information.

## Project Structure

```
browseagent/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API application
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app router
│   │   ├── components/   # React components
│   │   └── lib/          # Utility functions
│   ├── package.json       # Node.js dependencies
│   ├── tailwind.config.js # Tailwind CSS configuration
│   └── postcss.config.js  # PostCSS configuration
├── scrape.py              # Web scraping utility
├── plan.md               # Development roadmap
└── README.md             # This file
```

## Features

### Phase 1: Basic Setup ✅
- [x] Project structure initialization
- [x] Next.js frontend setup
- [x] FastAPI backend setup
- [x] Tailwind CSS v4 configuration
- [x] ESLint configuration
- [x] Basic project documentation

### Phase 2: Frontend UI Components 🚧
- [ ] Main search interface
- [ ] Results display area
- [ ] Source attribution
- [ ] Loading states
- [ ] Error handling
- [ ] Responsive design

### Phase 3: Backend Implementation 🚧
- [ ] Web scraping functionality
- [ ] Content extraction
- [ ] LLM integration
- [ ] Query processing
- [ ] Source management
- [ ] Error handling

### Phase 4: Integration & Testing 🚧
- [ ] Frontend-backend integration
- [ ] API testing
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Security implementation

### Phase 5: Advanced Features 🚧
- [ ] Multi-source aggregation
- [ ] Content summarization
- [ ] Citation generation
- [ ] User preferences
- [ ] History tracking
- [ ] Export functionality

## Technology Stack

### Frontend
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS v4** - Utility-first CSS framework
- **shadcn/ui** - UI component library (to be added)
- **TypeScript** - Type-safe JavaScript

### Backend
- **FastAPI** - Modern Python web framework
- **Python 3.10+** - Programming language
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### AI/ML Stack (Phase 3+)
- **LangChain** - LLM application framework
- **LlamaIndex** - Data framework for LLMs
- **OpenAI API** - LLM provider
- **Chromadb** - Vector database (optional)

## Getting Started

### Prerequisites
- Node.js 18+ (Note: Current setup uses Node.js 10.19.0 with warnings)
- Python 3.10+
- npm or yarn
- pip

### Installation

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Running the Application
1. Start the backend server on `http://localhost:8000`
2. Start the frontend development server on `http://localhost:3000`
3. Access the application at `http://localhost:3000`

## Development Roadmap

See [plan.md](plan.md) for detailed development roadmap and milestones.

## API Endpoints

### Current Endpoints
- `GET /` - Health check
- `POST /query` - Process user queries (placeholder)

### Future Endpoints
- `POST /scrape` - Web scraping endpoint
- `GET /sources` - Retrieve source information
- `POST /settings` - User preferences

## Configuration

### Environment Variables
Create a `.env.local` file in the frontend directory and `.env` in the backend directory with the following variables:

```env
# Backend
OPENAI_API_KEY=your_openai_api_key
NEXT_PUBLIC_API_URL=http://localhost:8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions, please open an issue in the GitHub repository.

## Acknowledgments

- Built with Next.js and FastAPI
- Styled with Tailwind CSS
- AI capabilities powered by OpenAI API
- Web scraping capabilities to be implemented
