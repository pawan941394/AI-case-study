from fastapi.concurrency import run_in_threadpool

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import asyncio
import contextlib
import os
from functools import lru_cache
from typing import AsyncGenerator, Tuple, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query
from chatbot import agent
import random


from chat_memories import get_session_ids_by_username, get_chats_by_session,delete_session,get_session_token_usage

class ChatRequest(BaseModel):
    username: str = Field(..., description="Username to associate with the session")
    prompt: str = Field(..., description="User's latest prompt for the travel agent")


class ContinueChatRequest(BaseModel):

    username: str = Field(..., description="Username to associate with the session")
    prompt: str = Field(..., description="User's latest prompt for the travel agent")



class SessionSummary(BaseModel):
    session_id: str
    first_message: str


class ChatEntry(BaseModel):
    user_prompt: str
    assistant_response: str
    timestamp: Optional[str] = None




import uuid

def generate_conversation_id() -> str:
    return str(uuid.uuid4())


app = FastAPI(title="Bot API", version="1.0.0")


@app.post("/conversations")
async def chat(request: ChatRequest) -> StreamingResponse:
    user_id = request.username
    conversation_id = conversation_id = str(uuid.uuid4())
    res = await run_in_threadpool(
        agent.run,
        request.prompt,
        user_id=request.username,
        session_id=conversation_id,
    )
    return {
        "conversation_id": conversation_id,
        "response": res.content
    }

@app.post("/conversations/{conversation_id}/messages")
async def continue_chat(conversation_id: str, request: ContinueChatRequest):
    user_id = request.username

    res = await run_in_threadpool(
        agent.run,
        request.prompt,
        user_id=request.username,
        session_id=conversation_id,
    )

    return {
        "conversation_id": conversation_id,
        "response": res.content
    }




@app.get("/users/{user_id}/conversations/")
async def get_sessions(user_id: str) -> list[SessionSummary]:
    sessions_data = get_session_ids_by_username(user_id)
    return [SessionSummary(**session) for session in sessions_data]


@app.get("/sessions/{username}/{conversation_id}/messages")
async def get_chats(username: str, conversation_id: str) -> list[ChatEntry]:
    chats = get_chats_by_session(conversation_id, username)
    return [
        ChatEntry(
            user_prompt=chat.get("user_prompt", ""),
            assistant_response=chat.get("assistant_response", ""),
            timestamp=str(chat.get("timestamp")) if chat.get("timestamp") is not None else None,
        )
        for chat in chats
    ]


@app.delete("/conversations/{user_id}/{conversation_id}")
async def delete_session_endpoint(user_id: str, conversation_id: str) -> dict:

    success = delete_session(conversation_id, user_id)
    return {"success": success}


@app.get("/sessions/{username}/{conversation_id}/token_usage")
async def get_token_usage(username: str, conversation_id: str) -> dict:
    token_usage = get_session_token_usage(conversation_id, username)
    return token_usage