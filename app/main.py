import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
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

# ==================== Available Models (Verified Working) ====================
AVAILABLE_MODELS = [
    # ===== Text Generation =====
    "@cf/meta/llama-3.2-1b-instruct",
    "@cf/meta/llama-3.2-3b-instruct",
    "@cf/meta/llama-3.2-11b-vision-instruct",
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
    "@cf/zai-org/glm-4.7-flash",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/qwen/qwq-32b",
    "@cf/qwen/qwen2.5-coder-32b-instruct",
    "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/mistral/mistral-7b-instruct-v0.2-lora",
    "@cf/ibm-granite/granite-4.0-h-micro",

    # ===== Text Embeddings =====
    "@cf/baai/bge-base-en-v1.5",
    "@cf/baai/bge-large-en-v1.5",
    "@cf/baai/bge-small-en-v1.5",
    "@cf/baai/bge-m3",
    "@cf/google/embeddinggemma-300m",
    "@cf/qwen/qwen3-embedding-0.6b",
    "@cf/pfnet/plamo-embedding-1b",

    # ===== Text Classification =====
    "@cf/baai/bge-reranker-base",
    "@cf/huggingface/distilbert-sst-2-int8",

    # ===== Automatic Speech Recognition =====
    "@cf/openai/whisper",
    "@cf/openai/whisper-large-v3-turbo",
    "@cf/openai/whisper-tiny-en",
    "@cf/deepgram/flux",
    "@cf/deepgram/nova-3",

    # ===== Text-to-Speech =====
    "@cf/myshell-ai/melotts",
    "@cf/deepgram/aura-1",
    "@cf/deepgram/aura-2-en",
    "@cf/deepgram/aura-2-es",

    # ===== Text-to-Image =====
    "@cf/black-forest-labs/flux-1-schnell",
    "@cf/black-forest-labs/flux-2-klein-9b",
    "@cf/black-forest-labs/flux-2-klein-4b",
    "@cf/black-forest-labs/flux-2-dev",
    "@cf/lykon/dreamshaper-8-lcm",
    "@cf/runwayml/stable-diffusion-v1-5-img2img",
    "@cf/runwayml/stable-diffusion-v1-5-inpainting",
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "@cf/bytedance/stable-diffusion-xl-lightning",
    "@cf/leonardo/phoenix-1.0",
    "@cf/leonardo/lucid-origin",

    # ===== Image-to-Text =====
    "@cf/llava-hf/llava-1.5-7b-hf",
    "@cf/moondream/moondream3.1-9B-A2B",

    # ===== Image Classification =====
    "@cf/microsoft/resnet-50",

    # ===== Object Detection =====
    "@cf/facebook/detr-resnet-50",

    # ===== Translation =====
    "@cf/meta/m2m100-1.2b",
    "@cf/ai4bharat/indictrans2-en-indic-1B",

    # ===== Voice Activity Detection =====
    "@cf/pipecat-ai/smart-turn-v2",
]

# ==================== OpenAI-Compatible Schemas ====================

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(256, ge=1, le=2048)
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
        if msg.content and msg.content.strip():
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
    
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one non-empty message is required"
        )
    
    payload = {"messages": messages}
    
    if request.max_tokens:
        payload["max_tokens"] = min(request.max_tokens, 2048)
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    
    return payload

def call_cloudflare_api(model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call Cloudflare AI API with better error handling."""
    url = f"{BASE_URL}{model}"
    
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        
        # Log for debugging
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
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
        
        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid JSON response from Cloudflare"
            )
        
        # Check if result is valid
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Empty response from Cloudflare"
            )
        
        return result
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Cloudflare API timeout"
        )
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
    """Format Cloudflare response to OpenAI-compatible format with safe checks."""
    
    # Check success
    if not cloudflare_response.get("success", False):
        errors = cloudflare_response.get("errors", [])
        error_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )
    
    # Get result with safe checks
    result = cloudflare_response.get("result")
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No result in Cloudflare response"
        )
    
    # Get response text with safe check
    response_text = result.get("response")
    if response_text is None:
        response_text = ""
    
    # Get usage with safe checks
    usage_data = result.get("usage", {})
    if usage_data is None:
        usage_data = {}
    
    prompt_tokens = usage_data.get("prompt_tokens")
    completion_tokens = usage_data.get("completion_tokens")
    total_tokens = usage_data.get("total_tokens")
    
    # Handle None values
    if prompt_tokens is None:
        prompt_tokens = 0
    if completion_tokens is None:
        completion_tokens = 0
    if total_tokens is None:
        total_tokens = 0
    
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
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
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
            detail=f"Model '{request.model}' not available. Available: {AVAILABLE_MODELS}"
        )
    
    # Validate messages
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required"
        )
    
    request_id = generate_id()
    created_at = get_timestamp()
    
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
