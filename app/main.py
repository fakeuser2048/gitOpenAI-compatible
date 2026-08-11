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
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "8702d5bfa8a2f290dd5fa041a132541f")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "cfut_nnMOYOMizqMPjoVnGxFUBtrlp3mUwlvHSQgX0vpDfb01fb7e")
BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/"

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# ==================== Available Models ====================
AVAILABLE_MODELS = [
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/meta/llama-3.1-8b-instruct-fp8-fast",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-3.1-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/qwen/qwq-32b",
    "@cf/qwen/qwen2.5-coder-32b-instruct"
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
                formatted_messages.append(msg)
    return formatted_messages

def prepare_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Prepare payload for Cloudflare API."""
    messages = extract_messages(request.messages)
    
    # Handle system prompt for models that don't support it directly
    formatted_messages = []
    system_prompt = None
    
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        else:
            formatted_messages.append(msg)
    
    # If system prompt exists, prepend to first user message
    if system_prompt and formatted_messages:
        first_user = next((m for m in formatted_messages if m.get("role") == "user"), None)
        if first_user:
            first_user["content"] = f"{system_prompt}\n\n{first_user['content']}"
    
    payload = {
        "messages": formatted_messages if formatted_messages else messages
    }
    
    if request.max_tokens:
        payload["max_tokens"] = min(request.max_tokens, 2048)  # Cloudflare limit
    
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    
    return payload

def call_cloudflare_api(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Cloudflare AI API."""
    url = f"{BASE_URL}{model}"
    
    try:
        response = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 400:
            error_detail = response.json().get("errors", [{}])[0].get("message", "Bad Request")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cloudflare API error: {error_detail}"
            )
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            error_msg = e.response.text[:200]
        else:
            error_msg = str(e)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cloudflare API error: {error_msg}"
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
        
        # Split into words for streaming
        words = response_text.split()
        
        async def generate():
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_at, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            
            for word in words:
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
                await asyncio.sleep(0.02)
            
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
