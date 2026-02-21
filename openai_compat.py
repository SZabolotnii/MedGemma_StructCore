"""
Minimal OpenAI-compatible chat-completions client (DSPy-free).

Supports:
- POST /v1/chat/completions
- optional `response_format` (Structured Output / JSON schema) with graceful fallback

Intended for local inference backends (LM Studio, llama.cpp server, vLLM, etc.).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse


@dataclass
class ChatResult:
    text: str
    raw: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleChatClient:
    def __init__(self, *, url: str, model: str, timeout: int = 180, debug: bool = False):
        root_url, v1_url = self._normalize_urls(url or "")
        self.url = root_url
        self.v1_url = v1_url
        self.model = model
        # Local inference can be slow (esp. quantized + LoRA on CPU). Allow overriding
        # the request timeout without touching code.
        env_timeout = (os.getenv("OPENAI_COMPAT_TIMEOUT_S") or "").strip()
        if env_timeout:
            try:
                timeout = int(env_timeout)
            except Exception:
                pass
        self.timeout = int(timeout)
        self.debug = bool(debug)

    @staticmethod
    def _normalize_urls(url: str) -> tuple[str, str]:
        """Return (root_url, v1_url).

        Accepts either a root backend URL (e.g., http://127.0.0.1:1245) or a v1 URL
        (e.g., http://127.0.0.1:1245/v1). Also tolerates full endpoint URLs like
        http://127.0.0.1:1245/v1/chat/completions by trimming the path to the /v1 root.

        This normalization is intentionally permissive because local OpenAI-compatible
        servers are often configured behind reverse proxies or UI tools that expose a
        full path rather than a base URL.
        """
        u = (url or "").strip()
        if not u:
            return "", "/v1"
        if "://" not in u:
            u = f"http://{u}"
        p = urlparse(u)
        # Keep scheme/netloc/query/fragment; normalize only path.
        path = (p.path or "").rstrip("/")

        # If the user passed a full endpoint (e.g. /v1/chat/completions), trim to the root.
        # We only trim when "/v1" is a real segment boundary.
        v1_idx = path.find("/v1")
        if v1_idx != -1:
            after = path[v1_idx + len("/v1") :]
            if after == "" or after.startswith("/"):
                path = path[:v1_idx]
        root = urlunparse((p.scheme, p.netloc, path, "", p.query, p.fragment)).rstrip("/")
        v1 = f"{root}/v1"
        return root, v1

    def list_models(self) -> List[str]:
        req = urllib.request.Request(f"{self.v1_url}/models", headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GET {self.v1_url}/models failed: {e.code} {e.reason}") from e
        except Exception as e:
            raise RuntimeError(f"GET {self.v1_url}/models failed: {e}") from e

        obj = json.loads(raw)
        ids: List[str] = []
        for item in (obj.get("data") or []):
            mid = str((item or {}).get("id") or "").strip()
            if mid:
                ids.append(mid)
        return ids

    def assert_model_available(self) -> None:
        ready_timeout_s = int(os.getenv("OPENAI_COMPAT_MODEL_READY_TIMEOUT_S", "180"))
        deadline = time.time() + max(10, ready_timeout_s)
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                ids = self.list_models()
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                # During llama-server startup /v1/models may return 503 until the model is loaded.
                last_err = e
                time.sleep(0.5)

        if last_err is not None:
            raise SystemExit(f"Failed to verify model availability via /v1/models: {last_err}") from last_err
        if self.model in ids:
            return
        preview = ", ".join(ids[:12]) + (", ..." if len(ids) > 12 else "")
        raise SystemExit(
            "Requested model id is not available on the OpenAI-compatible backend.\n"
            f"- Requested: {self.model}\n"
            f"- URL: {self.url}\n"
            f"- v1 URL: {self.v1_url}\n"
            f"- Available (sample): {preview}\n"
            "Fix: restart the backend with the correct weights, or pass a correct --model.\n"
        )

    def chat(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 768,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(user_prompt)})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if repetition_penalty is not None:
            # llama.cpp historically used "repeat_penalty" (and may ignore "repetition_penalty").
            # Safe to send both: unknown fields should be ignored by the backend.
            rp = float(repetition_penalty)
            payload["repetition_penalty"] = rp
            payload["repeat_penalty"] = rp
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if min_p is not None:
            payload["min_p"] = float(min_p)
        if typical_p is not None:
            payload["typical_p"] = float(typical_p)
        if stop:
            payload["stop"] = [str(s) for s in stop if str(s)]
        if response_format is not None:
            payload["response_format"] = response_format

        def _rf_type(rf: Any) -> str:
            if not isinstance(rf, dict):
                return "none"
            t = str(rf.get("type") or "").strip()
            return t if t else "unknown"

        def _call(pl: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
            if self.debug:
                dbg = {
                    "temperature": pl.get("temperature"),
                    "max_tokens": pl.get("max_tokens"),
                    "repetition_penalty": pl.get("repetition_penalty"),
                    "repeat_penalty": pl.get("repeat_penalty"),
                    "top_p": pl.get("top_p"),
                    "min_p": pl.get("min_p"),
                    "typical_p": pl.get("typical_p"),
                    "stop": pl.get("stop"),
                    "response_format": (pl.get("response_format") or {}).get("type") if isinstance(pl.get("response_format"), dict) else None,
                }
                dbg = {k: v for k, v in dbg.items() if v is not None}
                print(
                    f"[openai_compat] POST {self.v1_url}/chat/completions model={self.model} "
                    f"params={json.dumps(dbg, ensure_ascii=False)}"
                )
            req = urllib.request.Request(
                f"{self.v1_url}/chat/completions",
                data=json.dumps(pl).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, dict) and obj.get("error"):
                raise RuntimeError(f"OpenAI-compatible error: {obj.get('error')}")
            text = str(obj["choices"][0]["message"]["content"] or "")
            return text, obj

        attempts: List[str] = []
        try:
            attempts.append(_rf_type(payload.get("response_format")))
            text, obj = _call(payload)
            return ChatResult(
                text=text,
                raw=obj,
                meta={
                    "response_format_attempts": attempts,
                    "response_format_final": attempts[-1] if attempts else "none",
                    "used_response_format_fallback": bool(len(attempts) > 1),
                },
            )
        except Exception as e:
            # Some servers reject response_format. Retry with widely-supported fallback(s).
            if "response_format" not in payload:
                raise
            try:
                payload["response_format"] = {"type": "json_object"}
                attempts.append(_rf_type(payload.get("response_format")))
                text, obj = _call(payload)
                return ChatResult(
                    text=text,
                    raw=obj,
                    meta={
                        "response_format_attempts": attempts,
                        "response_format_final": attempts[-1] if attempts else "none",
                        "used_response_format_fallback": bool(len(attempts) > 1),
                        "fallback_reason": str(e),
                    },
                )
            except Exception:
                payload.pop("response_format", None)
                attempts.append(_rf_type(payload.get("response_format")))
                text, obj = _call(payload)
                return ChatResult(
                    text=text,
                    raw=obj,
                    meta={
                        "response_format_attempts": attempts,
                        "response_format_final": attempts[-1] if attempts else "none",
                        "used_response_format_fallback": bool(len(attempts) > 1),
                        "fallback_reason": str(e),
                    },
                )

    def complete(
        self,
        *,
        prompt: str,
        max_tokens: int = 768,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        """Raw text completion via /completion endpoint (no chat template).

        Use this when the model was fine-tuned on raw text (not chat format),
        e.g. the hard200 Stage2 LoRA.
        """
        payload: Dict[str, Any] = {
            "prompt": str(prompt),
            "temperature": float(temperature),
            "n_predict": int(max_tokens),
            "stream": False,
        }
        if repetition_penalty is not None:
            payload["repeat_penalty"] = float(repetition_penalty)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if min_p is not None:
            payload["min_p"] = float(min_p)
        if typical_p is not None:
            payload["typical_p"] = float(typical_p)
        if stop:
            payload["stop"] = [str(s) for s in stop if str(s)]

        if self.debug:
            print(f"[openai_compat] POST {self.url}/completion prompt_len={len(prompt)} max_tokens={max_tokens}")
        req = urllib.request.Request(
            f"{self.url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        obj = json.loads(raw.decode("utf-8"))
        text = str(obj.get("content") or "")
        return ChatResult(
            text=text,
            raw=obj,
            meta={"endpoint": "completion", "prompt_len": len(prompt)},
        )
