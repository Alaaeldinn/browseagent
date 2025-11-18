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

# Create the FastHTML app
app, rt = fast_app(
    hdrs=(
        # Add Tailwind CSS
        Link(rel="stylesheet", href="https://cdn.tailwindcss.com"),
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
            }}

            body {{
                font-family: 'Söhne', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Ubuntu, Cantarell, sans-serif;
                background-color: var(--secondary);
                color: var(--text);
                height: 100vh;
                margin: 0;
                display: flex;
                flex-direction: column;
            }}

            .chat-container {{
                display: flex;
                flex-direction: column;
                height: 100vh;
                max-width: 800px;
                margin: 0 auto;
                background-color: var(--panel-bg);
            }}

            .header {{
                padding: 1rem;
                border-bottom: 1px solid var(--border);
                background-color: white;
                position: sticky;
                top: 0;
                z-index: 10;
            }}

            .chat-history {{
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                background-color: var(--secondary);
            }}

            .message {{
                padding: 1rem 0;
                border-bottom: 1px solid var(--border);
            }}

            .user-message {{
                background-color: #ffffff;
            }}

            .assistant-message {{
                background-color: var(--secondary);
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
                padding: 0.5rem 1rem;
                border-radius: 0.375rem;
                cursor: pointer;
                font-weight: 500;
            }}

            .btn:hover {{
                background-color: var(--primary-hover);
            }}

            .settings-panel {{
                padding: 1rem;
                border-bottom: 1px solid var(--border);
                background-color: white;
            }}

            .input-field {{
                width: 100%;
                padding: 0.5rem;
                border: 1px solid var(--border);
                border-radius: 0.375rem;
                background-color: var(--input-bg);
            }}

            .provider-select {{
                margin-right: 1rem;
            }}
        """)
    )
)

# List of supported providers
providers = [
    "openai/gpt-3.5-turbo",
    "openai/gpt-4",
    "openai/gpt-4-turbo",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-pro",
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
        cls="input-field provider-select"
    )

    api_key_input = Input(
        type="password",
        name="api_key",
        placeholder="Enter API key",
        cls="input-field",
        id="api-key"
    )

    settings_panel = Div(
        H2("Settings", cls="text-lg font-semibold mb-2"),
        Div(
            Label("LLM Provider:", cls="mr-2"),
            provider_select,
            cls="flex items-center mb-2"
        ),
        Div(
            Label("API Key:", cls="mr-2"),
            api_key_input,
            cls="flex items-center"
        ),
        cls="settings-panel"
    )

    # Chat history area
    chat_history = Div(id="chat-history", cls="chat-history")

    # Input area with query input and submit button
    query_input = Input(
        type="text",
        name="query",
        placeholder="Message BrowseAgent...",
        cls="input-field",
        id="query-input",
        hx_trigger="keydown[key=='Enter' && !shiftKey]",
        hx_post="/query",
        hx_target="#chat-history",
        hx_swap="beforeend"
    )

    submit_btn = Button(
        "Send",
        cls="btn",
        id="submit-btn",
        hx_post="/query",
        hx_target="#chat-history",
        hx_swap="beforeend"
    )

    input_area = Div(
        Div(
            query_input,
            submit_btn,
            cls="flex gap-2"
        ),
        cls="input-area"
    )

    # Main chat container
    chat_container = Div(
        settings_panel,
        chat_history,
        input_area,
        cls="chat-container"
    )

    return Titled("BrowseAgent", chat_container)

# Route to handle the query
@rt("/query")
def post(query: str, llm_provider: str = "openai/gpt-3.5-turbo", api_key: str = ""):
    # Validate inputs
    if not query.strip():
        return Div("Please enter a query.", cls="message assistant-message text-red-500")

    if not api_key.strip():
        return Div("Please enter an API key.", cls="message assistant-message text-red-500")

    # Add user message to chat
    user_msg = Div(
        Div("You", cls="font-semibold"),
        Div(query, cls="mt-1"),
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

    try:
        # Make the API call to the backend
        response = requests.post(backend_url, json=payload)
        result = response.json()

        # Add assistant message to chat
        assistant_msg = Div(
            Div("BrowseAgent", cls="font-semibold"),
            Div(result["result"], cls="mt-1"),
            cls="message assistant-message"
        )

        return user_msg, assistant_msg

    except Exception as e:
        error_msg = Div(
            Div("BrowseAgent", cls="font-semibold"),
            Div(f"Error: {str(e)}", cls="mt-1 text-red-500"),
            cls="message assistant-message"
        )

        return user_msg, error_msg

serve()