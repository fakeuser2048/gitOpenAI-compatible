import os
import uuid
import json
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from threading import Lock

import httpx
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GitHub Copilot OpenAI-compatible API")

# ---------- Config ----------
GITHUB_TOKEN = os.getenv("GITHUB_COPILOT_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_COPILOT_TOKEN environment variable is required")

SERVER_API_KEY = os.getenv("SERVER_API_KEY", "")  # if empty, no client auth needed

GITHUB_API_BASE = "https://api.individual.githubcopilot.com/github/chat"

# Thread state: thread_id -> last_assistant_message_id
thread_state: Dict[str, str] = {}
state_lock = Lock()

# ---------- Models ----------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "github-copilot"
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    # Custom field for thread management
    thread_id: Optional[str] = None

# ---------- Helpers ----------
def get_github_headers() -> Dict[str, str]:
    return {
        "Authorization": f"GitHub-Bearer {GITHUB_TOKEN}",
        "copilot-integration-id": "copilot-chat",
        "x-github-api-version": "2025-05-01",
        "Content-Type": "application/json",
    }

async def create_thread(client: httpx.AsyncClient) -> str:
    url = f"{GITHUB_API_BASE}/threads"
    headers = get_github_headers()
    # GitHub expects empty body {}
    resp = await client.post(url, headers=headers, json={})
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Failed to create thread: {resp.text}")
    data = resp.json()
    # The response is like {"id": "thread-uuid"}
    return data["id"]

async def send_message_stream(
    client: httpx.AsyncClient,
    thread_id: str,
    content: str,
    parent_message_id: str,
    response_message_id: str,
) -> httpx.Response:
    url = f"{GITHUB_API_BASE}/threads/{thread_id}/messages"
    headers = get_github_headers()
    headers["Content-Type"] = "text/event-stream"  # request for SSE
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
        "model": "auto",
        "mode": "immersive",
        "parentMessageID": parent_message_id,
        "mediaContent": [],
        "skillOptions": {"deepCodeSearch": False},
        "requestTrace": False,
    }
    req = client.build_request("POST", url, headers=headers, json=body)
    return await client.send(req, stream=True)

# ---------- Authentication dependency ----------
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if SERVER_API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        token = authorization.split(" ", 1)[1]
        if token != SERVER_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    # If no SERVER_API_KEY set, allow all requests
    return True

# ---------- OpenAI response formatters ----------
def openai_chunk(
    id: str,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> bytes:
    chunk = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(__import__("time").time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()

def openai_final_chunk() -> bytes:
    return b"data: [DONE]\n\n"

# ---------- Routes ----------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "github-copilot",
                "object": "model",
                "created": 1686935002,
                "owned_by": "github",
            }
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _: bool = Depends(verify_api_key),
):
    # Extract last user message (assuming it's the new prompt)
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message required")
    last_user_content = user_messages[-1].content

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Thread management
        thread_id = body.thread_id
        parent_id = "root"
        with state_lock:
            if thread_id and thread_id in thread_state:
                parent_id = thread_state[thread_id]
            # else: new thread, parent stays root, will be created below

        if not thread_id or thread_id not in thread_state:
            # Create new thread
            thread_id = await create_thread(client)
            with state_lock:
                thread_state[thread_id] = "root"  # initial parent
            parent_id = "root"

        # Generate unique IDs
        response_message_id = str(uuid.uuid4())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        # Send message to GitHub API and stream
        try:
            resp = await send_message_stream(
                client, thread_id, last_user_content, parent_id, response_message_id
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"GitHub API request failed: {str(e)}")

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

                    # Try to extract delta content
                    delta_content = None
                    finish_reason = None
                    # GitHub Copilot may wrap in choices format
                    if "choices" in data and len(data["choices"]) > 0:
                        choice = data["choices"][0]
                        if "delta" in choice:
                            delta_content = choice["delta"].get("content", "")
                            finish_reason = choice.get("finish_reason")
                        # Update assistant message id if available (usually in message metadata)
                        if "message" in choice and "id" in choice["message"]:
                            collected_assistant_id = choice["message"]["id"]
                    elif "body" in data:
                        # Alternate format: simple body text
                        delta_content = data["body"]
                    elif data.get("type") == "content":
                        delta_content = data.get("body", "")

                    if delta_content is not None:
                        delta = {"content": delta_content}
                        yield openai_chunk(completion_id, body.model, delta, finish_reason)

                # After stream ends, store the assistant message id for next turn
                if collected_assistant_id:
                    with state_lock:
                        thread_state[thread_id] = collected_assistant_id
                yield openai_final_chunk()

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        else:
            # Non-streaming: collect full content
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

            final_text = "".join(full_content)
            response_obj = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "thread_id": thread_id,  # custom field for session continuity
            }
            return JSONResponse(content=response_obj)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
