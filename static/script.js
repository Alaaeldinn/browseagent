const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = userInput.value.trim();
    if (!prompt) return;

    // Clear input and reset height
    userInput.value = '';
    userInput.style.height = 'auto';

    // Remove welcome message if it exists
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // Add user message
    addMessage(prompt, 'user');

    // Add loading indicator
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Remove loading indicator
        removeMessage(loadingId);

        // Add bot message
        addMessage(data.answer, 'bot', data.results, data.keywords);

    } catch (error) {
        console.error('Error:', error);
        removeMessage(loadingId);
        addMessage('Sorry, something went wrong. Please try again.', 'bot');
    }
});

// Handle Enter key to submit
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

function addMessage(text, sender, sources = null, keywords = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        sourcesHtml = `
            <div class="sources">
                <h4>Sources</h4>
                ${sources.map(s => `<a href="${s.link}" target="_blank" class="source-item" title="${s.snippet}">${s.title}</a>`).join('')}
            </div>
        `;
    }

    let keywordsHtml = '';
    if (keywords && keywords.length > 0) {
        // Optional: display keywords used
        // keywordsHtml = `<div class="keywords">Keywords: ${keywords.join(', ')}</div>`;
    }

    // Parse markdown
    const parsedText = marked.parse(text);

    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="avatar">${sender === 'user' ? 'U' : 'AI'}</div>
            <div class="text">
                ${parsedText}
                ${sourcesHtml}
            </div>
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Highlight code blocks
    messageDiv.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

function addLoadingMessage() {
    const id = 'loading-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot');
    messageDiv.id = id;

    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="avatar">AI</div>
            <div class="text">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return id;
}

function removeMessage(id) {
    const element = document.getElementById(id);
    if (element) {
        element.remove();
    }
}
