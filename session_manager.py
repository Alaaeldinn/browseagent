"""
Session Management for BrowseAgent
Handles temporary storage of API keys and user preferences
"""
import os
import time
import secrets
import hashlib
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration specific to a model"""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def to_dict(self):
        """Convert to dictionary for storage/serialization"""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        return cls(
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1000),
            top_p=data.get("top_p", 0.9),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0)
        )

@dataclass
class UserSession:
    """Represents a user session with API key and preferences"""
    session_id: str
    api_key: str
    created_at: datetime
    last_accessed: datetime
    selected_model: str = "openai/gpt-3.5-turbo"
    searx_host: str = "https://search.us.projectsegfau.lt"
    use_searxng: bool = True
    request_count: int = 0
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))  # 24-hour session
    is_active: bool = True
    model_configs: Dict[str, ModelConfig] = field(default_factory=dict)  # Model-specific configurations

    def update_access_time(self):
        """Update the last accessed time"""
        self.last_accessed = datetime.now()

    def increment_request_count(self):
        """Increment the request counter"""
        self.request_count += 1

    def is_expired(self) -> bool:
        """Check if the session has expired"""
        return datetime.now() > self.expires_at

    def extend_session(self, hours: int = 24):
        """Extend the session by specified hours"""
        self.expires_at = datetime.now() + timedelta(hours=hours)

    def set_model_config(self, model_name: str, config: ModelConfig):
        """Set configuration for a specific model"""
        self.model_configs[model_name] = config
        self.update_access_time()

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model, return default if not set"""
        if model_name in self.model_configs:
            return self.model_configs[model_name]
        else:
            # Return a default configuration
            return ModelConfig()


class SessionManager:
    """
    Manages user sessions in memory with automatic cleanup
    """
    def __init__(self, session_timeout_hours: int = 24):
        self.sessions: Dict[str, UserSession] = {}
        self.lock = Lock()  # Thread-safe access
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.cleanup_interval = timedelta(minutes=30)  # Clean up every 30 minutes
        
    def create_session(self, api_key: str, selected_model: str = "openai/gpt-3.5-turbo") -> str:
        """
        Create a new user session and return the session ID
        """
        with self.lock:
            # Generate a secure session ID
            session_id = secrets.token_urlsafe(32)

            # Create the session
            session = UserSession(
                session_id=session_id,
                api_key=api_key,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                selected_model=selected_model
            )

            # Store the session
            self.sessions[session_id] = session

            logger.info(f"Created new session with ID: {session_id[:8]}...")
            return session_id

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get a session by ID, returns None if not found or expired
        """
        with self.lock:
            if session_id not in self.sessions:
                logger.debug(f"Session not found: {session_id[:8]}...")
                return None

            session = self.sessions[session_id]

            # Check if session is expired
            if session.is_expired():
                logger.info(f"Session expired, removing: {session_id[:8]}...")
                self._remove_session(session_id)
                return None

            # Update last accessed time
            session.update_access_time()
            logger.debug(f"Retrieved session: {session_id[:8]}...")
            return session

    def update_session_model(self, session_id: str, model: str) -> bool:
        """
        Update the selected model for a session
        """
        with self.lock:
            session = self.get_session(session_id)
            if session:
                session.selected_model = model
                session.update_access_time()
                return True
            return False

    def update_session_settings(self, session_id: str, **kwargs) -> bool:
        """
        Update session settings such as SearXNG host, etc.
        """
        with self.lock:
            session = self.get_session(session_id)
            if session:
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                session.update_access_time()
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID
        """
        with self.lock:
            return self._remove_session(session_id)

    def _remove_session(self, session_id: str) -> bool:
        """
        Internal method to remove a session (thread-safe)
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self):
        """
        Remove all expired sessions
        """
        with self.lock:
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if session.is_expired()
            ]

            for session_id in expired_sessions:
                self._remove_session(session_id)

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            return len(expired_sessions)

    def get_active_session_count(self) -> int:
        """
        Get the number of active sessions
        """
        with self.lock:
            count = 0
            for session in list(self.sessions.values()):  # Use list() to avoid modification during iteration
                if not session.is_expired():
                    count += 1
                else:
                    self._remove_session(session.session_id)
            return count

    def increment_request_count(self, session_id: str) -> bool:
        """
        Increment the request counter for a session
        """
        with self.lock:
            session = self.get_session(session_id)
            if session:
                session.increment_request_count()
                return True
            return False


# Global session manager instance
session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    return session_manager


def create_user_session(api_key: str, selected_model: str = "openai/gpt-3.5-turbo") -> str:
    """Create a new user session"""
    return session_manager.create_session(api_key, selected_model)


def get_user_session(session_id: str) -> Optional[UserSession]:
    """Get a user session by ID"""
    return session_manager.get_session(session_id)


def update_user_model(session_id: str, model: str) -> bool:
    """Update the selected model for a user session"""
    return session_manager.update_session_model(session_id, model)


def delete_user_session(session_id: str) -> bool:
    """Delete a user session"""
    return session_manager.delete_session(session_id)


if __name__ == "__main__":
    # Example usage
    print("Testing Session Manager...")
    
    # Create a session
    session_id = create_user_session("test-api-key-123", "openai/gpt-4")
    print(f"Created session: {session_id}")
    
    # Get the session
    session = get_user_session(session_id)
    if session:
        print(f"Session found: {session.session_id}")
        print(f"API Key: {session.api_key[:8]}...")
        print(f"Model: {session.selected_model}")
        print(f"Created: {session.created_at}")
    
    # Update model
    success = update_user_model(session_id, "anthropic/claude-3")
    print(f"Model update successful: {success}")
    
    # Get updated session
    updated_session = get_user_session(session_id)
    if updated_session:
        print(f"Updated model: {updated_session.selected_model}")
    
    # Delete session
    deleted = delete_user_session(session_id)
    print(f"Session deleted: {deleted}")