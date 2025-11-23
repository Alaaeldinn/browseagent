"""
Frontend UI for BrowseAgent using FastHTML
This implements the user interface for API key management and model selection
"""
from fasthtml.common import *
import os
import requests
import json
from urllib.parse import urljoin

# API base URL - in production this would be configurable
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Create the FastHTML app
app, rt = fast_app(
    hdrs=(
        # Add Tailwind CSS for styling
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        # Add custom styles
        Style("""
            .api-key-input { 
                font-family: monospace; 
                letter-spacing: 0.5px; 
            }
            .fade-in {
                animation: fadeIn 0.5s;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        """),
    ),
    # Enable sessions for temporary API key storage
    secret_key=os.getenv("SECRET_KEY", "browseagent-secret-key-change-in-production")
)

# Simple in-memory session for demonstration
# In production, use a proper session management system
active_sessions = {}

def check_auth(session):
    """Check if user is authenticated (has API key)"""
    return 'api_key' in session

def get_session_id_from_cookies():
    """Get session ID from cookies (placeholder implementation)"""
    # In a real implementation, you would get this from cookies
    # For this example, we'll store it in a global dict
    global session_id_store
    if 'session_id_store' not in globals():
        session_id_store = {}
    return session_id_store.get('current_session_id', None)

def set_session_id_in_cookies(session_id):
    """Set session ID in cookies (placeholder implementation)"""
    global session_id_store
    if 'session_id_store' not in globals():
        session_id_store = {}
    session_id_store['current_session_id'] = session_id

def get_models_from_api(session):
    """Fetch available models from the API"""
    session_id = session.get('session_id')
    if not session_id:
        return []

    headers = {
        'X-Session-ID': session_id,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(f"{API_BASE_URL}/models", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get('models', [])
        else:
            return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def create_session_with_api_key(api_key):
    """Create a session with the backend using API key"""
    headers = {
        'Content-Type': 'application/json'
    }

    payload = {
        'api_key': api_key
    }

    try:
        response = requests.post(f"{API_BASE_URL}/session/create", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get('valid', False):
                # Extract session ID from message
                message = result.get('message', '')
                if 'Session ID: ' in message:
                    session_id = message.split('Session ID: ')[1]
                    return session_id
            return None
        return None
    except Exception:
        return None

@rt("/")
def get(session):
    """Main page - API key input or chat interface"""
    if 'session_id' not in session:
        # Show API key input form
        return Div(
            H1("Welcome to BrowseAgent", cls="text-3xl font-bold text-center mb-8 text-gray-800"),
            Div(
                H2("Enter your OpenRouter API Key", cls="text-xl font-semibold mb-4 text-gray-700"),
                Form(
                    Input(
                        type="password",
                        name="api_key",
                        placeholder="sk-or-v1-...",
                        cls="w-full p-3 border border-gray-300 rounded-lg mb-4 api-key-input",
                        required=True
                    ),
                    Button(
                        "Create Session",
                        type="submit",
                        cls="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
                    ),
                    action="/set_api_key",
                    method="post",
                    cls="space-y-4"
                ),
                P("Your API key is used to authenticate with OpenRouter and is stored temporarily. Never shared or saved on our servers.",
                  cls="text-sm text-gray-500 mt-4 text-center"),
                cls="max-w-md mx-auto p-6 bg-white rounded-xl shadow-md"
            ),
            cls="min-h-screen bg-gray-50 flex items-center justify-center p-4"
        )
    else:
        # Show chat interface
        models = get_models_from_api(session)
        model_options = [Option(model['id'], value=model['id']) for model in models] if models else []

        return Div(
            # Header with logout
            Div(
                Div(
                    H1("BrowseAgent", cls="text-2xl font-bold text-gray-800"),
                    P(f"Using: {session.get('selected_model', 'Default Model')}", cls="text-sm text-gray-600"),
                    cls="flex-1"
                ),
                A("Logout", href="/logout",
                  cls="text-blue-600 hover:text-blue-800 font-medium"),
                cls="flex justify-between items-center p-4 border-b"
            ),

            # Model selection
            Div(
                Label("Select Model:", cls="block text-sm font-medium text-gray-700 mb-1"),
                Select(
                    *model_options,
                    name="model",
                    id="model-select",
                    cls="w-full p-2 border border-gray-300 rounded-md",
                    hx_post="/set_model",
                    hx_target="#model-status",
                    hx_swap="innerHTML"
                ) if model_options else P("Loading models...", cls="text-gray-500"),
                Div(id="model-status", cls="mt-1 text-sm text-green-600"),
                cls="p-4 border-b bg-gray-50"
            ),

            # Chat interface
            Div(
                Div(id="chat-messages", cls="flex-1 p-4 overflow-y-auto max-h-96"),
                Form(
                    Div(
                        Input(
                            type="text",
                            name="message",
                            placeholder="Ask me anything...",
                            cls="flex-1 p-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500",
                            id="message-input"
                        ),
                        Button(
                            "Send",
                            type="submit",
                            cls="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-r-lg font-medium"
                        ),
                        cls="flex"
                    ),
                    hx_post="/chat",
                    hx_target="#chat-messages",
                    hx_swap="beforeend",
                    hx_vals=f"javascript:{{'session_id': '{session['session_id']}'}}",
                    cls="flex p-4 border-t"
                ),
                cls="flex flex-col h-[60vh]"
            ),
            Script("""
                // Auto-scroll to bottom of chat
                function scrollToBottom() {
                    const chatDiv = document.getElementById('chat-messages');
                    chatDiv.scrollTop = chatDiv.scrollHeight;
                }
                document.addEventListener('htmx:afterRequest', scrollToBottom);
            """),
            cls="flex flex-col h-screen max-w-4xl mx-auto"
        )

@rt("/set_api_key")
def post(api_key: str, session):
    """Handle API key submission and session creation"""
    session_id = create_session_with_api_key(api_key)
    if session_id:
        session['session_id'] = session_id
        session['api_key'] = api_key  # Still store temporarily for the session

        # Get the session info to initialize model
        headers = {
            'X-Session-ID': session_id,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.get(f"{API_BASE_URL}/session/info", headers=headers)
            if response.status_code == 200:
                info = response.json()
                session['selected_model'] = info.get('selected_model', 'openai/gpt-3.5-turbo')
            else:
                session['selected_model'] = 'openai/gpt-3.5-turbo'
        except:
            session['selected_model'] = 'openai/gpt-3.5-turbo'

        return RedirectResponse("/", status_code=303)
    else:
        return Div(
            H1("Session Creation Failed", cls="text-2xl font-bold text-red-600 mb-4"),
            P("The API key you provided is invalid or could not create a session. Please try again.", cls="text-gray-700 mb-4"),
            A("Go Back", href="/", cls="text-blue-600 hover:underline"),
            cls="min-h-screen flex items-center justify-center p-4 text-center"
        )

@rt("/set_model")
def post(model: str, session):
    """Handle model selection"""
    session_id = session.get('session_id')
    if not session_id:
        return Div("Session expired. Please log in again.", cls="text-red-500 text-sm")

    headers = {
        'X-Session-ID': session_id,
        'Content-Type': 'application/json'
    }

    payload = {
        'model': model
    }

    try:
        response = requests.post(f"{API_BASE_URL}/model/select", headers=headers, json=payload)
        if response.status_code == 200:
            session['selected_model'] = model
            return Div(f"Model set to: {model}", cls="text-green-600 text-sm")
        else:
            return Div(f"Failed to update model: {response.text}", cls="text-red-500 text-sm")
    except Exception as e:
        return Div(f"Error updating model: {str(e)}", cls="text-red-500 text-sm")

@rt("/chat")
def post(message: str, session):
    """Handle chat messages"""
    session_id = session.get('session_id')
    if not session_id:
        return Div("Session expired. Please log in again.", cls="text-red-500")

    # Make request to backend API
    headers = {
        'X-Session-ID': session_id,
        'Content-Type': 'application/json'
    }

    payload = {
        'query': message,
        'llm_provider': session.get('selected_model', 'openai/gpt-3.5-turbo')
    }

    try:
        response = requests.post(f"{API_BASE_URL}/query", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            user_msg = Div(
                Div("You", cls="font-semibold text-blue-600"),
                Div(message, cls="mt-1 text-gray-800"),
                cls="mb-4 text-left"
            )

            agent_msg = Div(
                Div("BrowseAgent", cls="font-semibold text-green-600"),
                Div(result.get('result', 'No response'), cls="mt-1 text-gray-800"),
                cls="mb-4 text-left bg-gray-50 p-3 rounded-lg"
            )

            return user_msg, agent_msg
        else:
            error_msg = response.json().get('detail', 'Error contacting backend')
            return Div(
                Div("System", cls="font-semibold text-red-600"),
                Div(f"Error: {error_msg}", cls="mt-1 text-gray-800"),
                cls="mb-4 text-left"
            )
    except Exception as e:
        return Div(
            Div("System", cls="font-semibold text-red-600"),
            Div(f"Error: {str(e)}", cls="mt-1 text-gray-800"),
            cls="mb-4 text-left"
        )

@rt("/logout")
def get(session):
    """Handle logout"""
    session_id = session.get('session_id')
    if session_id:
        # Optionally notify backend to invalidate session
        try:
            headers = {
                'X-Session-ID': session_id,
                'Content-Type': 'application/json'
            }
            # We could add a backend endpoint to explicitly delete the session
            # For now, we'll just clear the frontend session
        except:
            pass

    # Clear session data
    session.clear()
    return RedirectResponse("/", status_code=303)

# Additional API endpoints for frontend
@rt("/models")
def get(session):
    """API endpoint to fetch models (for frontend use)"""
    if not check_auth(session):
        return {"error": "Not authenticated"}
    
    models = get_models_from_api(session)
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)