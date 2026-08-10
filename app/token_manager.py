# app/token_manager.py
import os
import json
import time
from pathlib import Path
import httpx
from typing import Optional, List

class TokenManager:
    def __init__(self):
        self.tokens_file = Path("tokens.json")
        self.tokens: List[dict] = []
        self.current_index = 0
        self.load_tokens()
    
    def load_tokens(self):
        """بارگذاری توکن‌ها از فایل یا environment"""
        # اول از environment
        env_token = os.getenv("GITHUB_COPILOT_TOKEN", "")
        if env_token:
            self.tokens = [{"token": env_token.strip(), "added": time.time()}]
        
        # بعد از فایل
        if self.tokens_file.exists():
            try:
                with open(self.tokens_file) as f:
                    file_tokens = json.load(f)
                    # اضافه کردن توکن‌های جدید بدون تکرار
                    existing = {t["token"] for t in self.tokens}
                    for t in file_tokens:
                        if t["token"] not in existing:
                            self.tokens.append(t)
            except:
                pass
    
    def save_tokens(self):
        """ذخیره توکن‌ها"""
        with open(self.tokens_file, 'w') as f:
            json.dump(self.tokens, f, indent=2)
    
    def get_token(self) -> str:
        """دریافت توکن معتبر"""
        if not self.tokens:
            return os.getenv("GITHUB_COPILOT_TOKEN", "")
        
        # تلاش برای پیدا کردن اولین توکن معتبر
        for i in range(len(self.tokens)):
            token_data = self.tokens[self.current_index]
            if self.is_token_valid(token_data["token"]):
                return token_data["token"]
            
            # رفتن به توکن بعدی
            self.current_index = (self.current_index + 1) % len(self.tokens)
        
        # اگر هیچکدوم معتبر نبود، آخرین توکن رو برگردون
        return self.tokens[-1]["token"]
    
    async def is_token_valid(self, token: str) -> bool:
        """بررسی اعتبار توکن"""
        if not token.startswith("GitHub-Bearer "):
            token = f"GitHub-Bearer {token}"
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    "https://api.individual.githubcopilot.com/github/chat/threads",
                    headers={
                        "Authorization": token,
                        "copilot-integration-id": "copilot-chat",
                        "x-github-api-version": "2025-05-01",
                    },
                    json={}
                )
                return resp.status_code == 200
        except:
            return False
    
    def add_token(self, new_token: str):
        """اضافه کردن توکن جدید"""
        if not new_token:
            return
        
        token_data = {"token": new_token.strip(), "added": time.time()}
        
        # جلوگیری از تکرار
        if not any(t["token"] == token_data["token"] for t in self.tokens):
            self.tokens.insert(0, token_data)  # اضافه به اول لیست
            self.save_tokens()
    
    def remove_invalid_tokens(self):
        """حذف توکن‌های نامعتبر"""
        valid_tokens = []
        for token_data in self.tokens:
            if time.time() - token_data["added"] < 7 * 24 * 3600:  # یک هفته
                valid_tokens.append(token_data)
        
        self.tokens = valid_tokens
        self.save_tokens()
