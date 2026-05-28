"""
src/models/ollama_client.py
────────────────────────────
Unified LLM client that supports two backends:
  1. Ollama  (local)         — default for local dev
  2. HuggingFace Inference API (cloud) — for deployment

Set env var BACKEND=huggingface and HF_TOKEN=hf_xxx to use HF.
"""
from __future__ import annotations
import os, json, asyncio, logging, re
from typing import Optional
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL      = "qwen2.5:3b"
HF_API_URL         = "https://api-inference.huggingface.co/models"
HF_MODEL           = "Qwen/Qwen2.5-3B-Instruct"


class OllamaClient:
    """
    Drop-in client for both Ollama (local) and HF Inference API (cloud).
    The rest of the codebase calls .chat() and .generate() — backend is transparent.
    """
    def __init__(self, base_url=DEFAULT_OLLAMA_URL, model=DEFAULT_MODEL,
                 timeout=120, max_connections=10):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = aiohttp.ClientTimeout(total=timeout)
        self._connector = aiohttp.TCPConnector(limit=max_connections)
        self._session: Optional[aiohttp.ClientSession] = None

        # Detect backend
        self.backend  = os.getenv("BACKEND", "ollama").lower()   # "ollama" | "huggingface"
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.hf_model = os.getenv("HF_MODEL", HF_MODEL)

        if self.backend == "huggingface":
            logger.info(f"[LLMClient] Backend = HuggingFace  model = {self.hf_model}")
        else:
            logger.info(f"[LLMClient] Backend = Ollama  url = {self.base_url}  model = {self.model}")

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=self._connector, timeout=self.timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── HuggingFace backend ───────────────────────────────────────────────────
 # ── HuggingFace (Now supports Groq Proxy) ─────────────────────────────────
    
# ── HuggingFace (Now supports Groq Proxy) ─────────────────────────────────
    async def _hf_chat(self, messages: list[dict], system: Optional[str], max_tokens: int) -> str:
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        # Build clean payload
        payload = {
            "model": str(self.hf_model).strip(),  # .strip() removes any accidental spaces from Railway UI
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,                   # Swapped to 0.2 (safer default for Groq)
            "stream": False,
        }
        
        # Intercept and route to Groq if using Groq Key
        if str(self.hf_token).startswith("gsk_"):
            url = "https://api.groq.com/openai/v1/chat/completions"
        else:
            url = f"https://api-inference.huggingface.co/v1/chat/completions"
            
        async with session.post(url, headers=headers, json=payload) as resp:
            # If Groq returns an error, let's capture the exact text description in our logs to read it
            if resp.status != 200:
                err_text = await resp.text()
                logger.error(f"Groq API Error Response: {err_text}")
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    # ── Ollama backend ────────────────────────────────────────────────────────
    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)))
    async def _ollama_chat(self, messages, system, temperature, max_tokens) -> str:
        session = await self._get_session()
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        payload = {
            "model": self.model,
            "messages": full_messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("message", {}).get("content", "")

    # ── Public interface ──────────────────────────────────────────────────────
    async def chat(self, messages: list[dict], system: Optional[str] = None,
                   temperature: float = 0.1, max_tokens: int = 512) -> str:
        if self.backend == "huggingface":
            return await self._hf_chat(messages, system, max_tokens)
        return await self._ollama_chat(messages, system, temperature, max_tokens)

    async def generate(self, prompt: str, system: Optional[str] = None,
                       temperature: float = 0.1, max_tokens: int = 512) -> dict:
        text = await self.chat(
            [{"role": "user", "content": prompt}], system, temperature, max_tokens)
        return {"text": text, "model": self.model}

    async def health_check(self) -> dict:
        if self.backend == "huggingface":
            return {"status": "ok", "backend": "huggingface", "model": self.hf_model}
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as resp:
                resp.raise_for_status()
                data  = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return {
                    "status": "ok", "backend": "ollama",
                    "model": self.model,
                    "model_available": any(self.model.split(":")[0] in m for m in models),
                    "available_models": models,
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.close()
