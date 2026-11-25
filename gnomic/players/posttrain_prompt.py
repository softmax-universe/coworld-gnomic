"""One player-visible prompt for Gnomic post-training and model inference."""

from __future__ import annotations

import json

from .sdk import GameView

REPLIES = {
    "introduce_request": '{"name":"your gnome name"}',
    "action_request": '{"action":"your natural-language move"}',
    "action_repair_request": '{"action":"your corrected natural-language move"}',
    "proposal_request": '{"proposal":{"kind":"enact","text":"new rule"}}. For amend, include rule_id and text. For repeal or transmute, include rule_id. You may add rationale.',
    "debate_request": '{"text":"your statement","vote_intent":"aye"}. Choose aye or nay.',
    "vote_request": '{"vote":"aye"}. Choose aye or nay.',
}


def messages_for(view: GameView) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": view.prompt_selection.rules(
                "You are a Gnomic player. Choose your move using the current player-visible view.",
                (
                    "Play Gnomic using the information visible to your seat.",
                    "Choose the move for the current request.",
                ),
                (
                    "Choose your Gnomic move based on what your seat can see.",
                    "Answer this turn’s request.",
                ),
            )
            + "\nReply with one JSON object. "
            + REPLIES[view.request["type"]],
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "seat": view.seat,
                    "turn": view.turn,
                    "proposer": view.proposer,
                    "host_constraints": view.host_constraints,
                    "limits": view.limits,
                    "rules": view.rules,
                    "state": view.state,
                    "proposal": view.proposal,
                    "debates": view.debates,
                    "recent_turns": view.history[-2:],
                    "request": {
                        k: v
                        for k, v in view.request.items()
                        if k not in {"rid", "timeout_s"}
                    },
                },
                ensure_ascii=False,
            ),
        },
    ]
