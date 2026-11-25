"""Saved-model native player with the same prompt and parser as teacher export."""

from __future__ import annotations

import os
import time
from typing import Any

from gnomic.llm_transport import complete_native, current_window
from gnomic.protocol import parse_reply_text

from .posttrain_prompt import messages_for
from .sdk import GameView, Policy, main_for


class NativePolicy(Policy):
    async def reply(self, view: GameView) -> dict[str, Any]:
        window = current_window.get()
        messages = messages_for(view)
        text = (
            await complete_native(
                {
                    "system": messages[0]["content"],
                    "messages": messages[1:],
                    "max_tokens": int(
                        os.environ.get("GNOMIC_LEARNER_MAX_TOKENS", "1024")
                    ),
                },
                os.environ.get("COWORLD_LLM_MODEL", "anthropic/claude-opus-4.7"),
                purpose="learner",
                slot=window.seat,
                timeout=window.deadline - time.monotonic(),
            )
        ).text
        result = parse_reply_text(text)
        if result.kind == "invalid":
            raise ValueError(f"Model reply is invalid: {result.error}")
        reply = result.value
        assert reply is not None
        return reply


if __name__ == "__main__":
    main_for(NativePolicy)
