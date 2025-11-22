# BrowseAgent: AI-Powered Search Assistant

## Project Overview

BrowseAgent is an intelligent search assistant that leverages large language models to provide contextual, relevant search results. The system extracts optimal search keywords from user prompts, performs searches, ranks results using embedding similarity, and generates comprehensive responses based on the top results.

This project is being developed as a Minimum Viable Product (MVP) to establish a robust, scalable, and user-friendly core platform.

## Guiding Principles for Development

This document serves as the primary guide for the AI coding agent. To ensure the highest quality and modern implementation, the agent is encouraged to:

-   **Actively Research:** Proactively search for the latest documentation, tutorials, and best practices for all technologies used (e.g., FastAPI, LangChain, SearXNG, OpenRouter API).
-   **Seek Optimal Solutions:** Evaluate different libraries or approaches before implementation to choose the most efficient and maintainable one.
-   **Clarify Ambiguities:** If any requirement in this document or the resulting `plan.md` is unclear, seek clarification or propose a well-reasoned solution.

## Current Architecture

### Flow
1.  User enters a prompt.
2.  LLM extracts optimal search keywords.
3.  Agent performs search via a tool call.
4.  Results (body, metadata) are collected.
5.  Embedding similarity scoring is applied to rank results.
6.  Top 5 results are extracted.
7.  These results are used as context to generate a final response.

### Technology Stack
-   **LangChain:** For LLM orchestration and tool management.
-   **FastAPI:** For the high-performance API backend.
-   **FastHTML:** For the initial frontend interface (to be replaced).

## MVP Requirements

### Core Improvements
1.  **Search Infrastructure**
    -   Replace the current search tool with **SearXNG**, an open-source metasearch engine, to gain control and privacy.
    -   Configure and deploy SearXNG for optimal result quality.

2.  **Model Integration**
    -   Implement **OpenRouter** integration.
    -   Allow users to provide their own OpenRouter API key.
    -   Enable users to select a model from a curated list of **free models available on OpenRouter** for the MVP.

3.  **Quality Assurance**
    -   Develop comprehensive unit tests for all core API endpoints.
    -   Implement integration tests for the complete search-and-response flow.

4.  **User Interface**
    -   Design and implement a modern, **ChatGPT-like conversational interface**.
    -   Improve user experience with real-time feedback, streaming responses, and clear result visualization.

5.  **Deployment**
    -   Create a production-ready **Docker image** for the API.
    -   Ensure all dependencies from the virtual environment are correctly managed within the Docker container.

### Future Features (Post-MVP)
-   **Deep Research:** Implement multi-step reasoning for complex, multi-faceted queries.
-   **Deep Think:** Enhanced analytical capabilities using advanced prompting techniques.
-   **Scientific Research Mode:** Specialized search focused exclusively on academic papers, journals, and scholarly databases.
-   **Real-time News Integration:** Focused search and synthesis of current news from various sources.

## Development Approach (Phased)

The project will be developed iteratively, broken down into logical phases. The `plan.md` will detail the specific tasks within each phase.

*   **Phase 1: Infrastructure & Core API**
    *   Integrate and configure SearXNG as the primary search tool.
    *   Set up OpenRouter/LiteLLM for model management.
    *   Restructure the core API to handle user-provided API keys and model selection.

*   **Phase 2: User Experience & Core Logic**
    *   Implement the logic for users to input and securely store their OpenRouter API key.
    *   Create the model selection interface (frontend and backend).
    *   Update the search flow to use the user's chosen model and API key.

*   **Phase 3: Quality & Testing**
    *   Write and execute unit tests for all new and existing API functionality.
    *   Develop integration tests for the end-to-end user journey.
    *   Refactor code for clarity and performance based on test results.

*   **Phase 4: UI/UX Overhaul**
    *   Design the new ChatGPT-like UI.
    *   Implement the frontend, replacing FastHTML with a modern framework (e.g., Typescript, React, Vue, or Svelte)(the easiest , the     fastest).
    *   Connect the frontend to the backend API, ensuring smooth data flow and state management.

*   **Phase 5: Deployment & Documentation**
    *   Write a robust `Dockerfile` and `docker-compose.yml` for easy deployment.
    *   Create comprehensive API documentation (e.g., with Swagger/OpenAPI).
    *   Finalize the project README and user guides.

## MVP Success Criteria

-   A user can sign up, add their OpenRouter API key, and select a free model.
-   Search queries are processed using SearXNG and return relevant, ranked results.
-   The application is fully tested with a target of >90% code coverage for critical paths.
-   The UI provides an intuitive, responsive, ChatGPT-like conversational experience.
-   The entire application can be reliably deployed to a production environment using Docker.

## Next Steps

This README will serve as the foundation for creating a detailed `plan.md`. That plan will break down each phase into granular, actionable tasks for the coding agent to execute.

---

*This document is a living guide and will be updated as the project evolves and new requirements emerge.*