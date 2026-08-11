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

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(256, ge=1, le=8192)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
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
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"

def get_timestamp() -> int:
    return int(time.time())

def prepare_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """Prepare payload for Cloudflare API - FIXED VERSION"""
    
    # Extract messages as simple dicts
    messages = []
    
    for msg in request.messages:
        # Skip empty messages
        if not msg.content or msg.content.strip() == "":
            continue
            
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Ensure at least one message exists
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one non-empty message is required"
        )
    
    payload = {
        "messages": messages
    }
    
    # Add optional parameters
    if request.max_tokens:
        payload["max_tokens"] = min(request.max_tokens, 2048)
    
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
        
        # Check for error response
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get("errors", [{}])[0].get("message", error_text)
            except:
                error_msg = error_text
            
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Cloudflare API error: {error_msg}"
            )
        
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
    
    # Validate messages
    if not request.messages or len(request.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required"
        )
    
    # Filter out empty messages
    valid_messages = [msg for msg in request.messages if msg.content and msg.content.strip()]
    if not valid_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one non-empty message is required"
        )
    
    request.messages = valid_messages
    
    request_id = generate_id()
    created_at = get_timestamp()
    
    # Prepare payload
    payload = prepare_payload(request)
    
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
