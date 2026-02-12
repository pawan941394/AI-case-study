import os
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

AGNO_DB_URL = os.getenv("AGNO_DB_URL", "sqlite:///tmp/agents.db")
_ENGINE: Optional[Engine] = None


def _get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        # Use the correct database path
        db_url = os.getenv("AGNO_DB_URL", "sqlite:///tmp/agents.db")
        
        try:
            _ENGINE = create_engine(db_url, future=True)
            # Test connection
            with _ENGINE.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            raise RuntimeError(f"Could not connect to database at {db_url}. Error: {e}")
    
    return _ENGINE


def get_session_ids_by_username(username: str) -> List[Dict[str, Any]]:
    if not username:
        raise ValueError("username is required.")
    query = text(
        """
        SELECT session_id, runs
        FROM agno_sessions
        WHERE user_id = :username
        ORDER BY created_at DESC
        """
    )
    with _get_engine().connect() as conn:
        rows = conn.execute(query, {"username": username})
        
        sessions = []
        for row in rows:
            session_id = row[0]
            first_message = "No messages found"
            
            if row[1]:  # Check if runs data exists
                import json
                try:
                    runs_data = json.loads(row[1])
                    
                    if isinstance(runs_data, str):
                        runs_data = json.loads(runs_data)
                    
                    # Get first user message from first run
                    if runs_data and len(runs_data) > 0:
                        first_run = runs_data[0]
                        messages = first_run.get("messages", [])
                        
                        for message in messages:
                            if message.get("role") == "user":
                                first_message = message.get("content", "No content")
                                break
                except json.JSONDecodeError:
                    first_message = "Error parsing messages"
            
            sessions.append({
                "session_id": session_id,
                "first_message": first_message
            })
        
        return sessions


def get_chats_by_session(session_id: str, user_id: str) -> List[Dict[str, Any]]:
    if not session_id or not user_id:
        raise ValueError("session_id and user_id are required.")
    query = text(
        """
        SELECT runs
        FROM agno_sessions
        WHERE session_id = :session_id AND user_id = :user_id
        ORDER BY created_at ASC
        """
    )
    with _get_engine().connect() as conn:
        rows = conn.execute(query, {"session_id": session_id, "user_id": user_id})
        
        chats = []
        for row in rows:
            if row[0]:  # Check if runs data exists
                import json
                try:
                    # First parse the JSON string
                    runs_data = json.loads(row[0])
                    
                    # If runs_data is still a string, parse it again
                    if isinstance(runs_data, str):
                        runs_data = json.loads(runs_data)
                    
                    # Now runs_data should be a list
                    for run in runs_data:
                        messages = run.get("messages", [])
                        
                        # Extract user and assistant messages
                        user_message = None
                        assistant_message = None
                        
                        for message in messages:
                            if message.get("role") == "user":
                                user_message = message.get("content")
                            elif message.get("role") == "assistant":
                                assistant_message = message.get("content")
                        
                        if user_message and assistant_message:
                            chats.append({
                                "user_prompt": user_message,
                                "assistant_response": assistant_message,
                                "timestamp": run.get("created_at")
                            })
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    continue
        
        return chats


def delete_session(session_id: str, user_id: str) -> bool:
    if not session_id or not user_id:
        raise ValueError("session_id and user_id are required.")
    delete_query = text(
        """
        DELETE FROM agno_sessions
        WHERE session_id = :session_id AND user_id = :user_id
        """
    )
    with _get_engine().begin() as conn:
        result = conn.execute(
            delete_query,
            {"session_id": session_id, "user_id": user_id}
        )
    return result.rowcount > 0



def get_session_token_usage(session_id: str, user_id: str) -> Dict[str, Any]:
    query = text(
        """
        SELECT runs
        FROM agno_sessions
        WHERE session_id = :session_id AND user_id = :user_id
        """
    )
    with _get_engine().connect() as conn:
        rows = conn.execute(query, {"session_id": session_id, "user_id": user_id})
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        
        for row in rows:
            if row[0]:
                import json
                try:
                    runs_data = json.loads(row[0])
                    if isinstance(runs_data, str):
                        runs_data = json.loads(runs_data)
                    
                    for run in runs_data:
                        metrics = run.get("metrics", {})
                        total_input_tokens += metrics.get("input_tokens", 0) or 0
                        total_output_tokens += metrics.get("output_tokens", 0) or 0
                        total_tokens += metrics.get("total_tokens", 0) or 0
                except json.JSONDecodeError:
                    continue
        
        return {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        }