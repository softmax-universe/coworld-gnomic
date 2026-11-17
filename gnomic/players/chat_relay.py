"""Hosted OpenAI Chat Completions relay player for Gnomic (preserves canonical prompt + parser)."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from gnomic.llm_transport import current_window
from gnomic.protocol import parse_reply_text

from .posttrain_prompt import messages_for
from .sdk import GameView, Policy, main_for

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _extract_chat_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("Chat completion response missing choices[0]")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        return "".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    if isinstance(content, str):
        return content
    raise ValueError("Chat completion message content is empty")


class ChatRelayPolicy(Policy):
    async def _reply(self, view: GameView) -> dict[str, Any]:
        window = current_window.get()
        endpoint = os.environ.get("COWORLD_LLM_ENDPOINT", "").strip().rstrip("/")
        if not endpoint:
            raise RuntimeError("COWORLD_LLM_ENDPOINT is not configured")
        model = os.environ.get("COWORLD_LLM_MODEL", "").strip()
        if not model:
            raise RuntimeError("COWORLD_LLM_MODEL is not configured")
        messages = messages_for(view)
        max_tokens = int(os.environ.get("GNOMIC_LEARNER_MAX_TOKENS", "1536"))
        remaining = max(1.0, window.deadline - time.monotonic())
        req_body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "X-Coworld-Player-Slot": str(window.seat),
        }
        async with httpx.AsyncClient(trust_env=False, timeout=remaining) as client:
            resp = await client.post(
                f"{endpoint}/v1/chat/completions",
                json=req_body,
                headers=headers,
            )
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"Chat completion failed ({resp.status_code}): {resp.text[:300]}")
        text = _extract_chat_text(resp.json()).strip()
        result = parse_reply_text(text)
        if result.kind == "invalid":
            m = _FENCE_RE.match(text)
            if m:
                result = parse_reply_text(m.group(1).strip())
        if result.kind == "invalid":
            raise ValueError(f"Model reply is invalid: {result.error}")
        reply = result.value
        assert reply is not None
        if reply.get("rid") != view.request["rid"]:
            raise ValueError("Model reply has the wrong request id")
        return reply

    async def introduce(self, view: GameView) -> str:
        name = (await self._reply(view))["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model gnome name must be nonempty text")
        return name

    async def action(self, view: GameView) -> str:
        action = (await self._reply(view))["action"]
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Model action must be nonempty text")
        return action

    async def propose(self, view: GameView) -> dict:
        proposal = (await self._reply(view))["proposal"]
        if not isinstance(proposal, dict):
            raise TypeError("Model proposal must be an object")
        return proposal

    async def debate(self, view: GameView) -> dict:
        reply = await self._reply(view)
        if not isinstance(reply["text"], str) or reply["vote_intent"] not in {
            "aye",
            "nay",
        }:
            raise ValueError("Model debate reply has invalid fields")
        return {"text": reply["text"], "vote_intent": reply["vote_intent"]}

    async def vote(self, view: GameView) -> str:
        vote = (await self._reply(view))["vote"]
        if vote not in {"aye", "nay"}:
            raise ValueError("Model vote must be aye or nay")
        return vote


if __name__ == "__main__":
    main_for(ChatRelayPolicy)
