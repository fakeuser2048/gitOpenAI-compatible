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
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-3.1-8b-instruct-fast",
    "@cf/meta/llama-3.1-8b-instruct-fp8",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/meta/llama-guard-3-8b",
    "@cf/meta-llama/llama-2-7b-chat-hf-lora",
    "@cf/google/gemma-4-26b-a4b-it",
    "@cf/google/gemma-2b-it-lora",
    "@cf/google/gemma-7b-it-lora",
    "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/qwen/qwq-32b",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/mistral/mistral-7b-instruct-v0.2-lora",
    "@cf/ibm-granite/granite-4.0-h-micro",
    "@cf/moondream/moondream3.1-9B-A2B",
]

# ==================== Token Counter ====================
def count_tokens(text: str) -> int:
    """Simple token counter (approximate)"""
    # هر ۴ کاراکتر ≈ ۱ توکن
    return len(text) // 4

def estimate_total_tokens(messages: List[Dict[str, str]], max_tokens: int) -> int:
    """Estimate total tokens for a request"""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
    return total + max_tokens

# ==================== Message Compressor ====================
def compress_messages(messages: List[Dict[str, str]], max_context: int = 30000) -> List[Dict[str, str]]:
    """
    Compress messages to fit within context limit.
    Preserves system message and recent messages, summarizes older ones.
    """
    if not messages:
        return messages
    
    MAX_INPUT_TOKENS = max_context
    MAX_RECENT_MESSAGES = 10  # تعداد پیام‌های اخیر که نگه داشته می‌شوند
    
    # محاسبه توکن‌های فعلی
    current_tokens = 0
    for msg in messages:
        current_tokens += count_tokens(msg.get("content", ""))
    
    # اگر زیر محدودیت است، همان را برگردان
    if current_tokens < MAX_INPUT_TOKENS:
        return messages
    
    # جداسازی پیام سیستم (اگر وجود دارد)
    system_msg = None
    other_messages = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg
        else:
            other_messages.append(msg)
    
    # پیام‌های اخیر را نگه دار
    recent_messages = other_messages[-MAX_RECENT_MESSAGES:] if len(other_messages) > MAX_RECENT_MESSAGES else other_messages
    
    # پیام‌های قدیمی‌تر را خلاصه کن
    older_messages = other_messages[:-MAX_RECENT_MESSAGES] if len(other_messages) > MAX_RECENT_MESSAGES else []
    
    compressed = []
    
    # سیستم پیام را اضافه کن
    if system_msg:
        compressed.append(system_msg)
    
    # خلاصه پیام‌های قدیمی
    if older_messages:
        summary_text = "خلاصه مکالمات قبلی:\n"
        for msg in older_messages:
            role = "کاربر" if msg.get("role") == "user" else "دستیار"
            content = msg.get("content", "")[:200]
            summary_text += f"{role}: {content}\n"
        
        compressed.append({
            "role": "system",
            "content": f"[خلاصه مکالمات قبلی]\n{summary_text}"
        })
    
    # پیام‌های اخیر را اضافه کن
    compressed.extend(recent_messages)
    
    return compressed

# ==================== OpenAI-Compatible Schemas ====================

class Message(BaseModel):
    role: str
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None
    
    @field_validator('content', mode='before')
    @classmethod
    def normalize_content(cls, v):
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
    """Prepare payload for Cloudflare API with automatic compression."""
    
    # Convert messages to dict format
    raw_messages = []
    for msg in request.messages:
        content = msg.content if msg.content else ""
        if content and str(content).strip():
            raw_messages.append({
                "role": msg.role,
                "content": str(content)
            })
    
    if not raw_messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one non-empty message is required"
        )
    
    # Compress messages if needed
    MAX_CONTEXT = 30000  # 32768 - reserve for output
    compressed_messages = compress_messages(raw_messages, MAX_CONTEXT)
    
    payload = {"messages": compressed_messages}
    
    # محدود کردن max_tokens
    if request.max_tokens:
        payload["max_tokens"] = min(request.max_tokens, 1024)
    else:
        payload["max_tokens"] = 512
    
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
    description="OpenAI-compatible API for Cloudflare AI models with automatic context compression",
    version="1.0.0"
)

# ==================== Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "Cloudflare AI API (OpenAI-Compatible) - با قابلیت فشرده‌سازی خودکار",
        "models": AVAILABLE_MODELS,
        "features": {
            "auto_compression": "فشرده‌سازی خودکار پیام‌های طولانی",
            "context_limit": "32768 توکن",
            "max_messages": "نامحدود (با خلاصه‌سازی)"
        },
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
    """OpenAI-compatible chat completions endpoint with auto-compression."""
    
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
