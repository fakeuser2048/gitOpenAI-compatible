import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any, Union

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
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
    # Meta Llama
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct-fast",
    "@cf/meta/llama-3.1-8b-instruct-fp8",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-guard-3-8b",
    "@cf/meta-llama/llama-2-7b-chat-hf-lora",
    
    # Google Gemma
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/google/gemma-2b-it-lora",
    "@cf/google/gemma-7b-it-lora",
    "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
    
    # Qwen
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/qwen/qwq-32b",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    
    # ZAI
    "@cf/zai-org/glm-4.7-flash",
    
    # OpenAI GPT-OSS
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    
    # NVIDIA
    "@cf/nvidia/nemotron-3-120b-a12b",
    
    # DeepSeek
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    
    # Mistral
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/mistral/mistral-7b-instruct-v0.2-lora",
    
    # IBM Granite
    "@cf/ibm-granite/granite-4.0-h-micro",
    
    # Moondream
    "@cf/moondream/moondream3.1-9B-A2B",
]

# ==================== OpenAI-Compatible Schemas ====================

class Message(BaseModel):
    role: str
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None
    
    @field_validator('content', mode='before')
    @classmethod
    def normalize_content(cls, v):
        """Normalize content to string"""
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            if v.get("type") == "text" and "text" in v:
                return v["text"]
            if "text" in v:
                return v["text"]
            return str(v)
        if isinstance(v, list):
            texts = []
            for item in v:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        texts.append(item["text"])
                    elif "text" in item:
                        texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts) if texts else ""
        return str(v) if v is not None else ""

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(256, ge=1, le=8192)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = False
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
    """Prepare payload for Cloudflare API."""
    
    messages = []
    for msg in request.messages:
        content = msg.content if msg.content else ""
        if content and str(content).strip():
            messages.append({
                "role": msg.role,
                "content": str(content)
            })
    
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one non-empty message is required"
        )
    
    payload = {"messages": messages}
    
    if request.max_tokens:
        payload["max_tokens"] = min(request.max_tokens, 8192)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    
    return payload

def call_cloudflare_api(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Cloudflare AI API."""
    url = f"{BASE_URL}{model}"
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        
        if response.status_code != 200:
            try:
                error_json = response.json()
                if error_json.get("errors"):
                    error_msg = error_json["errors"][0].get("message", str(error_json))
                else:
                    error_msg = response.text
            except:
                error_msg = response.text
            
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
                    "content": response_text or ""
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": usage_data.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage_data.get("completion_tokens", 0) or 0,
            "total_tokens": usage_data.get("total_tokens", 0) or 0
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
    
    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not available. Available: {AVAILABLE_MODELS}"
        )
    
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required"
        )
    
    request_id = generate_id()
    created_at = get_timestamp()
    
    try:
        payload = prepare_payload(request)
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
