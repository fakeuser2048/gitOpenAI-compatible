import os
import json
import time
import uuid
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


app = FastAPI(
    title="OpenAI Compatible Proxy",
    version="1.0.0"
)


UPSTREAM_URL = os.getenv("UPSTREAM_URL", "").rstrip("/")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "auto")

REQUEST_TIMEOUT = float(
    os.getenv("REQUEST_TIMEOUT", "300")
)


class Message(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


def check_auth(authorization: str | None):
    if not PROXY_API_KEY:
        return

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    expected = f"Bearer {PROXY_API_KEY}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


def upstream_headers():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    return headers


def convert_messages(messages: list[Message]):
    return [
        {
            "role": message.role,
            "content": message.content
        }
        for message in messages
    ]


@app.get("/")
async def root():
    return {
        "name": "OpenAI Compatible Proxy",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "upstream_configured": bool(UPSTREAM_URL)
    }


@app.get("/v1/models")
async def models(
    authorization: str | None = Header(default=None)
):
    check_auth(authorization)

    model = DEFAULT_MODEL

    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "upstream"
            }
        ]
    }


async def normal_request(
    request: ChatCompletionRequest
):
    if not UPSTREAM_URL:
        raise HTTPException(
            status_code=500,
            detail="UPSTREAM_URL is not configured"
        )

    payload = {
        "model": request.model,
        "messages": convert_messages(request.messages),
        "stream": False
    }

    if request.temperature is not None:
        payload["temperature"] = request.temperature

    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    if request.top_p is not None:
        payload["top_p"] = request.top_p

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT,
        connect=30
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            UPSTREAM_URL,
            headers=upstream_headers(),
            json=payload
        )

    if response.status_code >= 400:
        return JSONResponse(
            status_code=response.status_code,
            content={
                "error": {
                    "message": response.text,
                    "type": "upstream_error"
                }
            }
        )

    try:
        data = response.json()
    except Exception:
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.text
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    return JSONResponse(content=data)


async def stream_request(
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:

    if not UPSTREAM_URL:
        error = {
            "error": {
                "message": "UPSTREAM_URL is not configured",
                "type": "configuration_error"
            }
        }

        yield f"data: {json.dumps(error)}\n\n"
        yield "data: [DONE]\n\n"
        return

    payload = {
        "model": request.model,
        "messages": convert_messages(request.messages),
        "stream": True
    }

    if request.temperature is not None:
        payload["temperature"] = request.temperature

    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    if request.top_p is not None:
        payload["top_p"] = request.top_p

    timeout = httpx.Timeout(
        REQUEST_TIMEOUT,
        connect=30
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:

            async with client.stream(
                "POST",
                UPSTREAM_URL,
                headers=upstream_headers(),
                json=payload
            ) as response:

                if response.status_code >= 400:

                    body = await response.aread()

                    error = {
                        "error": {
                            "message": body.decode(
                                "utf-8",
                                errors="replace"
                            ),
                            "type": "upstream_error"
                        }
                    }

                    yield (
                        "data: "
                        + json.dumps(error, ensure_ascii=False)
                        + "\n\n"
                    )

                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():

                    if not line:
                        continue

                    # Upstream already uses SSE
                    if line.startswith("data:"):
                        raw = line[5:].strip()

                        if raw == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return

                        try:
                            parsed = json.loads(raw)

                            # If already OpenAI-compatible,
                            # forward it directly.
                            if (
                                isinstance(parsed, dict)
                                and (
                                    "choices" in parsed
                                    or "error" in parsed
                                )
                            ):
                                yield (
                                    "data: "
                                    + json.dumps(
                                        parsed,
                                        ensure_ascii=False
                                    )
                                    + "\n\n"
                                )
                                continue

                        except Exception:
                            pass

                        # Generic upstream data
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": raw
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }

                        yield (
                            "data: "
                            + json.dumps(
                                chunk,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )

                    else:
                        # Non-SSE upstream
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": line
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }

                        yield (
                            "data: "
                            + json.dumps(
                                chunk,
                                ensure_ascii=False
                            )
                            + "\n\n"
                        )

    except httpx.TimeoutException:

        error = {
            "error": {
                "message": "Upstream request timed out",
                "type": "timeout_error"
            }
        }

        yield (
            "data: "
            + json.dumps(error)
            + "\n\n"
        )

    except Exception as exc:

        error = {
            "error": {
                "message": str(exc),
                "type": "proxy_error"
            }
        }

        yield (
            "data: "
            + json.dumps(error)
            + "\n\n"
        )

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None)
):

    check_auth(authorization)

    if request.stream:
        return StreamingResponse(
            stream_request(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    return await normal_request(request)
