from fasthtml.common import *
import requests
import json

# Define color scheme
primary_color = "#10a37f"  # Green like ChatGPT
primary_hover = "#0d8a6a"
secondary_color = "#f7f7f8"  # Light gray background
text_color = "#202123"
border_color = "#e5e5e5"
input_bg = "#ffffff"
panel_bg = "#ffffff"
sidebar_bg = "#f0f4f8"

# Create the FastHTML app
app, rt = fast_app(
    hdrs=(
        # Add Tailwind CSS
        Link(rel="stylesheet", href="https://cdn.tailwindcss.com"),
        # Font
        Link(rel="preconnect", href="https://fonts.googleapis.com"),
        Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"),
        # Custom styles
        Style(f"""
            :root {{
                --primary: {primary_color};
                --primary-hover: {primary_hover};
                --secondary: {secondary_color};
                --text: {text_color};
                --border: {border_color};
                --input-bg: {input_bg};
                --panel-bg: {panel_bg};
                --sidebar-bg: {sidebar_bg};
            }}

            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--secondary);
                color: var(--text);
                height: 100vh;
                margin: 0;
                display: flex;
                flex-direction: column;
            }}

            .app-container {{
                display: flex;
                height: 100vh;
                max-width: 1600px;
                margin: 0 auto;
                background-color: var(--panel-bg);
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
            }}

            .sidebar {{
                width: 260px;
                background-color: var(--sidebar-bg);
                border-right: 1px solid var(--border);
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }}

            .main-content {{
                flex: 1;
                display: flex;
                flex-direction: column;
            }}

            .header {{
                padding: 1rem;
                border-bottom: 1px solid var(--border);
                background-color: white;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}

            .logo {{
                font-weight: 700;
                font-size: 1.25rem;
                color: var(--primary);
            }}

            .chat-history {{
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }}

            .message {{
                display: flex;
                gap: 1rem;
                padding: 0.75rem 0;
            }}

            .user-message {{
                background-color: #ffffff;
            }}

            .assistant-message {{
                background-color: var(--secondary);
            }}

            .avatar {{
                width: 30px;
                height: 30px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                font-weight: bold;
                color: white;
            }}

            .user-avatar {{
                background-color: #5436da;
            }}

            .assistant-avatar {{
                background-color: var(--primary);
            }}

            .message-content {{
                flex: 1;
            }}

            .input-area {{
                padding: 1rem;
                border-top: 1px solid var(--border);
                background-color: white;
            }}

            .btn {{
                background-color: var(--primary);
                color: white;
                border: none;
                padding: 0.5rem 1.25rem;
                border-radius: 0.5rem;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.2s;
            }}

            .btn:hover {{
                background-color: var(--primary-hover);
            }}

            .btn-outline {{
                background-color: white;
                color: var(--primary);
                border: 1px solid var(--primary);
            }}

            .btn-outline:hover {{
                background-color: var(--primary);
                color: white;
            }}

            .settings-panel {{
                background-color: white;
                border-radius: 0.5rem;
                padding: 1rem;
                border: 1px solid var(--border);
                margin-bottom: 1rem;
            }}

            .input-field {{
                width: 100%;
                padding: 0.6rem 0.8rem;
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                background-color: var(--input-bg);
                font-size: 0.9rem;
            }}

            .input-field:focus {{
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
            }}

            .input-group {{
                margin-bottom: 1rem;
            }}

            .input-group label {{
                display: block;
                margin-bottom: 0.3rem;
                font-weight: 500;
                font-size: 0.9rem;
            }}

            .chat-input-container {{
                display: flex;
                gap: 0.5rem;
            }}

            .chat-input {{
                flex: 1;
                padding: 0.75rem 1rem;
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                resize: none;
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                min-height: 56px;
                max-height: 200px;
            }}

            .chat-input:focus {{
                outline: none;
                border-color: var(--primary);
                box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2);
            }}

            .send-btn {{
                padding: 0.75rem 1.25rem;
            }}

            .provider-tag {{
                display: inline-block;
                background-color: #e6f7f0;
                color: var(--primary);
                padding: 0.25rem 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.75rem;
                font-weight: 500;
                margin-left: 0.5rem;
            }}

            .welcome-card {{
                text-align: center;
                padding: 2rem;
                border: 2px dashed var(--border);
                border-radius: 0.5rem;
                margin: 2rem 0;
            }}

            .welcome-title {{
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}

            .welcome-subtitle {{
                color: #666;
                margin-bottom: 1.5rem;
            }}
        """)
    )
)

# List of supported providers including free-tier options
providers = [
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",  # Free tier available
    "google/gemini-pro",          # Free tier available
    "mistral/mistral-tiny",       # Free tier available
    "groq/llama3-8b-8192",        # High rate limits, often free tier
    "openrouter/openchat-7b:free",     # OpenRouter free model
    "openrouter/google/gemma-7b-it:free",    # OpenRouter free model
    "openrouter/microsoft/wizardlm-2-7b:free", # OpenRouter free model
    "openrouter/nousresearch/nous-hermes-2-mistral-7b-dpo:free",  # OpenRouter free model
    "openrouter/mistralai/mistral-7b-instruct:free",  # OpenRouter free model
    "openai/gpt-4",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-opus",
    "google/gemini-1.5-pro",
    "mistral/mistral-large-latest",
    "perplexity/llama-3-sonar-large-32k-chat",
    "groq/llama3-70b-8192"
]

# Main page route
@rt("/")
def get():
    # Settings panel with provider selection and API key input
    provider_select = Select(
        *[Option(provider, value=provider, selected=(i==0)) for i, provider in enumerate(providers)],
        name="llm_provider",
        id="llm-provider",
        cls="input-field"
    )

    api_key_input = Input(
        type="password",
        name="api_key",
        placeholder="Enter your API key",
        cls="input-field",
        id="api-key"
    )

    settings_panel = Div(
        H3("Settings", cls="text-lg font-semibold mb-3"),
        Div(
            Label("LLM Provider:", cls="input-group"),
            provider_select,
        ),
        Div(
            Label("API Key:", cls="input-group"),
            api_key_input,
        ),
        cls="settings-panel"
    )

    # Sidebar
    sidebar = Div(
        H2("BrowseAgent", cls="logo mb-6"),
        settings_panel,
        Div(
            A("Reset Chat", href="/", cls="btn btn-outline w-full mb-2"),
            cls="mt-auto"
        ),
        cls="sidebar"
    )

    # Header
    header = Div(
        Div("BrowseAgent", cls="logo"),
        Div("Chat with Search"),
        cls="header"
    )

    # Chat history area with welcome message
    welcome_card = Div(
        H2("Welcome to BrowseAgent!", cls="welcome-title"),
        P("Ask anything and I'll search the web to find answers for you.", cls="welcome-subtitle"),
        cls="welcome-card"
    )

    chat_history = Div(welcome_card, id="chat-history", cls="chat-history")

    # Input area with query input and submit button
    query_input = Textarea(
        placeholder="Message BrowseAgent...",
        cls="chat-input",
        id="query-input",
        name="query",
        hx_trigger="keydown[key=='Enter' && !shiftKey]",
        hx_post="/query",
        hx_target="#chat-history",
        hx_swap="beforeend"
    )

    submit_btn = Button(
        "Send",
        cls="btn send-btn",
        id="submit-btn",
        hx_post="/query",
        hx_target="#chat-history",
        hx_swap="beforeend"
    )

    input_area = Div(
        Div(
            query_input,
            submit_btn,
            cls="chat-input-container"
        ),
        cls="input-area"
    )

    # Main content
    main_content = Div(
        header,
        chat_history,
        input_area,
        cls="main-content"
    )

    # Full app container
    app_container = Div(
        sidebar,
        main_content,
        cls="app-container"
    )

    return Titled("BrowseAgent", app_container)

# Route to handle the query
@rt("/query")
def post(query: str, llm_provider: str = "openai/gpt-3.5-turbo", api_key: str = ""):
    # Validate inputs
    if not query.strip():
        return Div(
            Div(
                Div("BrowseAgent", cls="font-semibold"),
                Div("Please enter a query.", cls="mt-1 text-red-500"),
            ),
            cls="message assistant-message"
        )

    if not api_key.strip():
        return Div(
            Div(
                Div("BrowseAgent", cls="font-semibold"),
                Div("Please enter an API key.", cls="mt-1 text-red-500"),
            ),
            cls="message assistant-message"
        )

    # Add user message to chat
    user_msg = Div(
        Div("You", cls="avatar user-avatar", inner_html="Y"),
        Div(
            Div("You", cls="font-semibold"),
            Div(query, cls="mt-1"),
        ),
        cls="message user-message"
    )

    # Prepare API payload
    backend_url = "http://localhost:8000/query"
    payload = {
        "query": query,
        "llm_provider": llm_provider
    }

    # Set the appropriate environment variable based on provider
    if llm_provider.startswith("openai/"):
        import os
        os.environ["OPENAI_API_KEY"] = api_key
    elif llm_provider.startswith("anthropic/"):
        import os
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif llm_provider.startswith("google/"):
        import os
        os.environ["GOOGLE_API_KEY"] = api_key
    elif llm_provider.startswith("mistral/"):
        import os
        os.environ["MISTRAL_API_KEY"] = api_key
    elif llm_provider.startswith("perplexity/"):
        import os
        os.environ["PERPLEXITY_API_KEY"] = api_key
    elif llm_provider.startswith("groq/"):
        import os
        os.environ["GROQ_API_KEY"] = api_key
    elif llm_provider.startswith("openrouter/"):
        import os
        os.environ["OPENROUTER_API_KEY"] = api_key

    try:
        # Make the API call to the backend
        response = requests.post(backend_url, json=payload)
        result = response.json()

        # Format the provider name for display
        provider_name = llm_provider.split('/')[-1].replace('-', ' ').title()

        # Add assistant message to chat
        assistant_msg = Div(
            Div("BA", cls="avatar assistant-avatar", inner_html="BA"),
            Div(
                Div(
                    Span("BrowseAgent", cls="font-semibold"),
                    Span(provider_name, cls="provider-tag")
                ),
                Div(result["result"], cls="mt-1"),
            ),
            cls="message assistant-message"
        )

        return user_msg, assistant_msg

    except Exception as e:
        error_msg = Div(
            Div("BA", cls="avatar assistant-avatar", inner_html="BA"),
            Div(
                Div(
                    Span("BrowseAgent", cls="font-semibold"),
                    Span("Error", cls="provider-tag")
                ),
                Div(f"Error: {str(e)}", cls="mt-1 text-red-500"),
            ),
            cls="message assistant-message"
        )

        return user_msg, error_msg

serve()