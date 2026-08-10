# app/main.py
import os
import uuid
import json
import time
import asyncio
from typing import Optional, List, Dict, Any
from threading import Lock
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from app.github_auth import GitHubTokenManager

load_dotenv()

token_manager = GitHubTokenManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh_task = asyncio.create_task(token_manager.start_auto_refresh())
    yield
    refresh_task.cancel()

app = FastAPI(title="GitHub Copilot OpenAI-compatible API", lifespan=lifespan)

SERVER_API_KEY = os.getenv("SERVER_API_KEY", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", os.getenv("SERVER_API_KEY", ""))
GITHUB_API_BASE = "https://api.individual.githubcopilot.com/github/chat"

SUPPORTED_MODELS = [
    "auto", "github-copilot", "claude-3.5-sonnet", "claude-3.7-sonnet",
    "gpt-4o", "gpt-4.1", "o3-mini", "o4-mini", "gemini-2.5-flash"
]

thread_state: Dict[str, str] = {}
state_lock = Lock()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    thread_id: Optional[str] = None

async def get_github_headers() -> Dict[str, str]:
    token = await token_manager.get_valid_token()
    if not token:
        raise HTTPException(status_code=500, detail="No valid GitHub token available")
    
    if not token.startswith("GitHub-Bearer "):
        token = f"GitHub-Bearer {token}"
    
    return {
        "Authorization": token,
        "copilot-integration-id": "copilot-chat",
        "x-github-api-version": "2025-05-01",
        "Content-Type": "application/json",
    }

async def create_thread(client: httpx.AsyncClient) -> str:
    url = f"{GITHUB_API_BASE}/threads"
    headers = await get_github_headers()
    resp = await client.post(url, headers=headers, json={})
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Failed to create thread: {resp.text}"
        )
    return resp.json()["id"]

async def send_message_stream(
    client: httpx.AsyncClient,
    thread_id: str,
    content: str,
    parent_message_id: str,
    response_message_id: str,
    model: str = "auto",
) -> httpx.Response:
    url = f"{GITHUB_API_BASE}/threads/{thread_id}/messages"
    headers = await get_github_headers()
    headers["Content-Type"] = "text/event-stream"
    body = {
        "responseMessageID": response_message_id,
        "content": content,
        "intent": "conversation",
        "references": [],
        "context": [],
        "currentURL": "https://github.com/copilot",
        "streaming": True,
        "confirmations": [],
        "customInstructions": [],
        "model": model,
        "mode": "immersive",
        "parentMessageID": parent_message_id,
        "mediaContent": [],
        "skillOptions": {"deepCodeSearch": False},
        "requestTrace": False,
    }
    req = client.build_request("POST", url, headers=headers, json=body)
    return await client.send(req, stream=True)

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if SERVER_API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        if authorization.split(" ", 1)[1] != SERVER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def openai_chunk(id: str, model: str, delta: dict, finish_reason: str = None) -> bytes:
    chunk = {
        "id": id, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()

def openai_final_chunk() -> bytes:
    return b"data: [DONE]\n\n"

@app.get("/")
async def root():
    return {
        "service": "GitHub Copilot OpenAI-compatible API",
        "auto_refresh": True,
        "docs": "/docs",
    }

@app.get("/debug/token")
async def debug_token():
    token = await token_manager.get_valid_token()
    if not token:
        return {"error": "No token available"}
    
    masked = token[:25] + "..." + token[-15:] if len(token) > 40 else "too short"
    is_valid = await token_manager._validate_token(token)
    
    return {
        "token_preview": masked,
        "is_valid": is_valid,
        "expires_at": str(token_manager.expires_at) if token_manager.expires_at else None,
        "has_session_cookie": bool(token_manager.github_session),
    }

@app.get("/admin/token-status")
async def token_status():
    token = await token_manager.get_valid_token()
    if not token:
        return {"status": "no_token", "is_valid": False}
    
    is_valid = await token_manager._validate_token(token)
    return {
        "is_valid": is_valid,
        "status": "valid" if is_valid else "invalid",
        "expires_at": str(token_manager.expires_at) if token_manager.expires_at else None,
        "auto_refresh": True,
    }

@app.post("/admin/refresh-token")
async def force_refresh_token(
    admin_key: str = Header(..., alias="X-Admin-Key")
):
    if ADMIN_KEY and admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    from datetime import datetime, timedelta
    new_token = await token_manager._fetch_new_token()
    if new_token:
        token_manager.token = new_token
        token_manager.expires_at = datetime.now() + timedelta(seconds=token_manager.refresh_interval)
        return {"status": "refreshed", "token_valid": True}
    
    return {"status": "failed", "token_valid": False}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "created": 1686935002, "owned_by": "github"},
            {"id": "claude-3.5-sonnet", "object": "model", "created": 1700000000, "owned_by": "anthropic"},
            {"id": "gpt-4o", "object": "model", "created": 1713000000, "owned_by": "openai"},
            {"id": "o3-mini", "object": "model", "created": 1718000000, "owned_by": "openai"},
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _: bool = Depends(verify_api_key),
):
    if body.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{body.model}' not supported")

    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message required")
    last_user_content = user_messages[-1].content

    async with httpx.AsyncClient(timeout=300.0) as client:
        thread_id = body.thread_id
        parent_id = "root"
        with state_lock:
            if thread_id and thread_id in thread_state:
                parent_id = thread_state[thread_id]

        if not thread_id or thread_id not in thread_state:
            thread_id = await create_thread(client)
            with state_lock:
                thread_state[thread_id] = "root"
            parent_id = "root"

        response_message_id = str(uuid.uuid4())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        resp = await send_message_stream(
            client, thread_id, last_user_content, parent_id, response_message_id,
            model=body.model
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text}")

        if body.stream:
            async def event_stream():
                collected_assistant_id = None
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta_content = None
                    finish_reason = None
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "delta" in choice:
                            delta_content = choice["delta"].get("content", "")
                            finish_reason = choice.get("finish_reason")
                        if "message" in choice and "id" in choice["message"]:
                            collected_assistant_id = choice["message"]["id"]
                    elif "body" in data:
                        delta_content = data["body"]

                    if delta_content is not None:
                        yield openai_chunk(completion_id, body.model, {"content": delta_content}, finish_reason)

                if collected_assistant_id:
                    with state_lock:
                        thread_state[thread_id] = collected_assistant_id
                yield openai_final_chunk()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        else:
            full_content = []
            collected_assistant_id = None
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "delta" in choice:
                        content = choice["delta"].get("content", "")
                        if content:
                            full_content.append(content)
                    if "message" in choice and "id" in choice["message"]:
                        collected_assistant_id = choice["message"]["id"]
                elif "body" in data:
                    full_content.append(data["body"])

            if collected_assistant_id:
                with state_lock:
                    thread_state[thread_id] = collected_assistant_id

            return JSONResponse(content={
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(full_content)},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "thread_id": thread_id,
            })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
