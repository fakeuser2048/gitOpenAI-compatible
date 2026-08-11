import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Union

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ==================== Configuration ====================
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/"

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# ==================== Available Models ====================
AVAILABLE_MODELS = [
    "meta/llama-3.2-1b-instruct",
    "meta/llama-3.2-3b-instruct",
    "meta/llama-3.1-8b-instruct-fp8-fast",
    "mistralai/mistral-small-3.1-24b-instruct",
    "meta/llama-3.1-70b-instruct-fp8-fast",
    "meta/llama-3.3-70b-instruct-fp8-fast",
    "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "zai-org/glm-4.7-flash",
    "qwen/qwq-32b",
    "qwen/qwen2.5-coder-32b-instruct"
]

# ==================== OpenAI-Compatible Schemas ====================

class Message(BaseModel):
    role: str = Field(..., description="Role of the message: system, user, or assistant")
    content: str = Field(..., description="Content of the message")

class ToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    function: Dict[str, Any]

class ToolCallMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use")
    messages: List[Union[Message, ToolCallMessage]] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(256, ge=1, le=8192)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    user: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

# ==================== Helper Functions ====================

def generate_id() -> str:
    """Generate a unique ID for completions."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"

def get_timestamp() -> int:
    """Get current timestamp."""
    return int(time.time())

def extract_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and format messages for Cloudflare API."""
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content is not None:
                formatted_messages.append({"role": role, "content": content})
            elif role == "assistant" and msg.get("tool_calls"):
                # Handle tool calls
                formatted_messages.append(msg)
    return formatted_messages

def prepare_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Prepare payload for Cloudflare API."""
    messages = extract_messages(request.messages)
    
    payload = {
        "messages": messages
    }
    
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens
    
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    
    if request.tools:
        payload["tools"] = [tool.model_dump() for tool in request.tools]
    
    if request.tool_choice:
        payload["tool_choice"] = request.tool_choice
    
    return payload

def call_cloudflare_api(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Cloudflare AI API."""
    url = f"{BASE_URL}{model}"
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cloudflare API error: {str(e)}"
        )

def format_response(
    cloudflare_response: Dict[str, Any],
    model: str,
    request_id: str,
    created_at: int
) -> Dict[str, Any]:
    """Format Cloudflare response to OpenAI-compatible format."""
    
    if not cloudflare_response.get("success", False):
        errors = cloudflare_response.get("errors", [])
        error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
    
    result = cloudflare_response.get("result", {})
    response_text = result.get("response", "")
    usage_data = result.get("usage", {})
    
    # Determine finish reason
    finish_reason = "stop"
    if len(response_text) >= (usage_data.get("completion_tokens", 0) * 4):
        finish_reason = "length"
    
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0)
        }
    }

# ==================== FastAPI App ====================

app = FastAPI(
    title="Cloudflare AI API (OpenAI-Compatible)",
    description="OpenAI-compatible API for Cloudflare AI models",
    version="1.0.0"
)

# ==================== Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "Cloudflare AI API (OpenAI-Compatible)",
        "models": AVAILABLE_MODELS,
        "endpoints": {
            "/v1/models": "List available models",
            "/v1/chat/completions": "Chat completions endpoint"
        }
    }

@app.get("/v1/models")
async def list_models():
    """List all available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "cloudflare"
            }
            for model in AVAILABLE_MODELS
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    
    # Validate model
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not found. Available: {AVAILABLE_MODELS}"
        )
    
    request_id = generate_id()
    created_at = get_timestamp()
    
    # Prepare payload
    payload = prepare_payload(request)
    
    # Handle streaming
    if request.stream:
        # Cloudflare doesn't support streaming natively, so we simulate it
        return await stream_response(request, payload, request_id, created_at)
    
    # Non-streaming response
    try:
        cloudflare_response = call_cloudflare_api(request.model, payload)
        formatted = format_response(cloudflare_response, request.model, request_id, created_at)
        return JSONResponse(content=formatted)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

async def stream_response(
    request: ChatCompletionRequest,
    payload: Dict[str, Any],
    request_id: str,
    created_at: int
):
    """Simulate streaming response."""
    import asyncio
    
    try:
        cloudflare_response = call_cloudflare_api(request.model, payload)
        result = cloudflare_response.get("result", {})
        response_text = result.get("response", "")
        usage_data = result.get("usage", {})
        
        # Chunk the response into words
        words = response_text.split()
        total_words = len(words)
        
        async def generate():
            # Send initial chunk
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_at, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            
            # Send chunks
            for i, word in enumerate(words):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": word + " "},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.02)  # Small delay
            
            # Send final chunk with usage
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_at,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming error: {str(e)}"
        )

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
