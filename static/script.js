// ===== STATE =====
let isFirstMessage = true;

// ===== DOM ELEMENTS =====
const welcomeState = document.getElementById('welcome-state');
const messagesContainer = document.getElementById('messages-container');
const inputArea = document.getElementById('input-area');
const chatForm = document.getElementById('chat-form');
const chatFormBottom = document.getElementById('chat-form-bottom');
const userInput = document.getElementById('user-input');
const userInputBottom = document.getElementById('user-input-bottom');

// ===== AUTO-RESIZE TEXTAREA =====
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

userInput.addEventListener('input', function () {
    autoResize(this);
});

userInputBottom.addEventListener('input', function () {
    autoResize(this);
});

// ===== FORM SUBMISSION =====
chatForm.addEventListener('submit', handleSubmit);
chatFormBottom.addEventListener('submit', handleSubmit);

async function handleSubmit(e) {
    e.preventDefault();

    const input = isFirstMessage ? userInput : userInputBottom;
    const prompt = input.value.trim();

    if (!prompt) return;

    // Clear input
    input.value = '';
    input.style.height = 'auto';

    // Transition from welcome to messages view
    if (isFirstMessage) {
        welcomeState.style.display = 'none';
        messagesContainer.style.display = 'block';
        inputArea.style.display = 'block';
        isFirstMessage = false;
    }

    // Add user message
    addMessage(prompt, 'user');

    // Add loading indicator
    const loadingId = addLoadingIndicator();

    try {
        // Call API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt }),
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        const data = await response.json();

        // Remove loading indicator
        removeLoadingIndicator(loadingId);

        // Add assistant response
        addAssistantMessage(data);

    } catch (error) {
        console.error('Error:', error);
        removeLoadingIndicator(loadingId);
        addMessage('Sorry, something went wrong. Please try again.', 'assistant');
    }
}

// ===== MESSAGE FUNCTIONS =====
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.parentElement.scrollTop = messagesContainer.parentElement.scrollHeight;
}

function addAssistantMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    // Answer
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = data.answer;
    messageDiv.appendChild(contentDiv);

    // Sources
    if (data.results && data.results.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';

        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.textContent = 'Sources';
        sourcesDiv.appendChild(sourcesTitle);

        const sourcesGrid = document.createElement('div');
        sourcesGrid.className = 'sources-grid';

        data.results.forEach((result, index) => {
            const sourceCard = document.createElement('a');
            sourceCard.className = 'source-card';
            sourceCard.href = result.link;
            sourceCard.target = '_blank';
            sourceCard.rel = 'noopener noreferrer';

            const sourceNumber = document.createElement('div');
            sourceNumber.className = 'source-number';
            sourceNumber.textContent = `${index + 1}`;

            const sourceTitle = document.createElement('div');
            sourceTitle.className = 'source-title';
            sourceTitle.textContent = result.title;

            const sourceUrl = document.createElement('div');
            sourceUrl.className = 'source-url';
            try {
                const url = new URL(result.link);
                sourceUrl.textContent = url.hostname;
            } catch {
                sourceUrl.textContent = result.link;
            }

            sourceCard.appendChild(sourceNumber);
            sourceCard.appendChild(sourceTitle);
            sourceCard.appendChild(sourceUrl);
            sourcesGrid.appendChild(sourceCard);
        });

        sourcesDiv.appendChild(sourcesGrid);
        messageDiv.appendChild(sourcesDiv);
    }

    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.parentElement.scrollTop = messagesContainer.parentElement.scrollHeight;
}

function addLoadingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'loading-message';

    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading';
    loadingDiv.innerHTML = `
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
    `;

    messageDiv.appendChild(loadingDiv);
    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.parentElement.scrollTop = messagesContainer.parentElement.scrollHeight;

    return 'loading-message';
}

function removeLoadingIndicator(id) {
    const loadingMessage = document.getElementById(id);
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// ===== KEYBOARD SHORTCUTS =====
userInput.addEventListener('keydown', handleKeydown);
userInputBottom.addEventListener('keydown', handleKeydown);

function handleKeydown(e) {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const form = e.target.closest('form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
}

// ===== INITIAL FOCUS =====
userInput.focus();
