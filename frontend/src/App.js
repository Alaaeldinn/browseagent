import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('openrouter/gpt-3.5-turbo');
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;
    if (!apiKey) {
      alert('Please enter your OpenRouter API key');
      return;
    }

    // Add user message
    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send request to backend
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          api_key: apiKey,
          model: model
        })
      });

      const data = await response.json();

      if (data.success) {
        const aiMessage = { 
          id: Date.now() + 1, 
          text: data.result, 
          sender: 'assistant' 
        };
        setMessages(prev => [...prev, aiMessage]);
      } else {
        const errorMessage = { 
          id: Date.now() + 1, 
          text: `Error: ${data.detail || 'Failed to get response'}`, 
          sender: 'system' 
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      const errorMessage = { 
        id: Date.now() + 1, 
        text: `Error: ${error.message}`, 
        sender: 'system' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const validateApiKey = async () => {
    if (!apiKey) return false;
    
    try {
      const response = await fetch('http://localhost:8000/validate-api-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_key: apiKey,
          model: model
        })
      });
      
      const data = await response.json();
      return data.valid;
    } catch (error) {
      console.error('API key validation error:', error);
      return false;
    }
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>BrowseAgent</h2>
        </div>
        
        <div className="settings-section">
          <button 
            className={`settings-btn ${showSettings ? 'active' : ''}`}
            onClick={() => setShowSettings(!showSettings)}
          >
            {showSettings ? 'Hide Settings' : 'Show Settings'}
          </button>
          
          {showSettings && (
            <div className="settings-panel">
              <div className="input-group">
                <label>OpenRouter API Key:</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter your API key"
                  className="api-key-input"
                />
              </div>
              
              <div className="input-group">
                <label>Model:</label>
                <select 
                  value={model} 
                  onChange={(e) => setModel(e.target.value)}
                  className="model-select"
                >
                  <option value="openrouter/gpt-3.5-turbo">GPT-3.5 Turbo</option>
                  <option value="openrouter/gpt-4">GPT-4</option>
                  <option value="mistralai/mistral-7b-instruct:free">Mistral 7B (Free)</option>
                  <option value="google/gemma-7b-it:free">Google Gemma 7B (Free)</option>
                </select>
              </div>
            </div>
          )}
        </div>
        
        <div className="info-section">
          <p>AI-powered search assistant with web search capabilities</p>
        </div>
      </div>
      
      <div className="chat-container">
        <div className="messages">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <h1>Welcome to BrowseAgent!</h1>
              <p>Ask me anything and I'll search the web to find answers for you.</p>
              <p>Enter your OpenRouter API key in the settings and start chatting!</p>
            </div>
          ) : (
            messages.map((message) => (
              <div 
                key={message.id} 
                className={`message ${message.sender}-message`}
              >
                <div className="message-content">
                  <strong>
                    {message.sender === 'user' ? 'You: ' : 
                     message.sender === 'assistant' ? 'Assistant: ' : 'System: '}
                  </strong>
                  <div className="message-text">
                    {message.text}
                  </div>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message assistant-message">
              <div className="message-content">
                <strong>Assistant: </strong>
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={apiKey ? "Message BrowseAgent..." : "Enter API key first..."}
            disabled={isLoading || !apiKey}
            className="message-input"
          />
          <button 
            type="submit" 
            disabled={isLoading || !inputValue.trim() || !apiKey}
            className="send-button"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;