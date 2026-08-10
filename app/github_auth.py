# app/github_auth.py
import os
import json
import time
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
import httpx
from pathlib import Path

logger = logging.getLogger(__name__)

class GitHubTokenManager:
    def __init__(self):
        self.token: Optional[str] = None
        self.expires_at: Optional[datetime] = None
        self.token_file = Path("current_token.json")
        self.lock = asyncio.Lock()
        self.github_session = os.getenv("GITHUB_SESSION_COOKIE", "")
        self.github_session_same_site = os.getenv("GITHUB_SESSION_SAME_SITE", "")
        self.initial_token = os.getenv("GITHUB_COPILOT_TOKEN", "")
        self.refresh_interval = int(os.getenv("TOKEN_REFRESH_INTERVAL", "3600"))
        
    def _get_cookies(self) -> Dict[str, str]:
        cookies = {}
        if self.github_session:
            cookies["user_session"] = self.github_session
        if self.github_session_same_site:
            cookies["__Host-user_session_same_site"] = self.github_session_same_site
        return cookies
    
    async def _fetch_new_token(self) -> Optional[str]:
        cookies = self._get_cookies()
        if not cookies:
            logger.warning("No GitHub session cookies available")
            return None
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "GitHub-Copilot/1.0",
            "Origin": "https://github.com",
            "Referer": "https://github.com/copilot",
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    "https://api.github.com/copilot/token",
                    headers=headers,
                    cookies=cookies
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    token = data.get("token")
                    if token:
                        logger.info("Got new token from copilot/token endpoint")
                        return f"GitHub-Bearer {token}"
                
                logger.info("Trying alternative method...")
                
                resp = await client.post(
                    "https://api.individual.githubcopilot.com/github/chat/threads",
                    headers={
                        **headers,
                        "Authorization": f"GitHub-Bearer {self.initial_token}" if self.initial_token else "",
                        "copilot-integration-id": "copilot-chat",
                        "x-github-api-version": "2025-05-01",
                    },
                    json={}
                )
                
                if resp.status_code == 401:
                    refresh_resp = await client.get(
                        "https://api.individual.githubcopilot.com/github/chat/token",
                        headers=headers,
                        cookies=cookies
                    )
                    if refresh_resp.status_code == 200:
                        new_token = refresh_resp.json().get("token", "")
                        if new_token:
                            return f"GitHub-Bearer {new_token}"
                
                logger.error(f"Failed to get token. Status: {resp.status_code}")
                return None
                
            except Exception as e:
                logger.error(f"Error fetching token: {e}")
                return None
    
    async def get_valid_token(self) -> Optional[str]:
        async with self.lock:
            if self.token and self.expires_at and datetime.now() < self.expires_at:
                return self.token
            
            if self.initial_token and await self._validate_token(self.initial_token):
                self.token = self.initial_token.strip()
                self.expires_at = datetime.now() + timedelta(seconds=self.refresh_interval)
                return self.token
            
            logger.info("Fetching new token...")
            new_token = await self._fetch_new_token()
            
            if new_token:
                self.token = new_token
                self.expires_at = datetime.now() + timedelta(seconds=self.refresh_interval)
                self._save_token_to_file(new_token)
                logger.info(f"New token obtained. Expires: {self.expires_at}")
                return new_token
            
            if self.token:
                logger.warning("Could not refresh, using cached token")
                return self.token
            
            logger.error("No valid token available")
            return None
    
    async def _validate_token(self, token: str) -> bool:
        if not token:
            return False
        
        if not token.startswith("GitHub-Bearer "):
            token = f"GitHub-Bearer {token}"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
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
    
    def _save_token_to_file(self, token: str):
        try:
            with open(self.token_file, 'w') as f:
                json.dump({
                    "token": token,
                    "saved_at": time.time(),
                    "expires_at": self.expires_at.isoformat() if self.expires_at else None
                }, f)
        except:
            pass
    
    def load_saved_token(self) -> Optional[str]:
        if self.token_file.exists():
            try:
                with open(self.token_file) as f:
                    data = json.load(f)
                    saved_time = data.get("saved_at", 0)
                    if time.time() - saved_time < 86400:
                        token = data.get("token")
                        if token:
                            self.token = token
                            expires = data.get("expires_at")
                            if expires:
                                self.expires_at = datetime.fromisoformat(expires)
                            return token
            except:
                pass
        return None
    
    async def start_auto_refresh(self):
        logger.info(f"Starting auto-refresh every {self.refresh_interval} seconds")
        
        saved_token = self.load_saved_token()
        if saved_token and await self._validate_token(saved_token):
            self.token = saved_token
            logger.info("Loaded saved token")
        
        while True:
            await asyncio.sleep(self.refresh_interval)
            try:
                logger.info("Auto-refreshing token...")
                new_token = await self._fetch_new_token()
                if new_token:
                    async with self.lock:
                        self.token = new_token
                        self.expires_at = datetime.now() + timedelta(seconds=self.refresh_interval)
                    logger.info("Token auto-refreshed successfully")
                else:
                    logger.warning("Auto-refresh failed, keeping current token")
            except Exception as e:
                logger.error(f"Error in auto-refresh: {e}")
