"""Haiku baseline: minimal LLM player through Coworld native Messages.

Reads the learner model override from ``COWORLD_LLM_MODEL``.
A failed model call is surfaced to the game.
"""

from __future__ import annotations

import json
import os

from gnomic.llm_transport import complete_native

from .scribe import ScribePolicy
from .sdk import GameView, Policy, main_for

SYSTEM = """\
You are seat {seat} in a game of Gnomic with {n} players. Players take turns \
proposing rule changes; after a debate, everyone votes; a Judge LLM enacts passed \
proposals and applies all rules each turn. You win by reaching the victory \
threshold in points (see the common state key 'victory_points') or having the \
most points when the game ends. Be strategic: propose rules that favor you but \
can attract a majority; vote your interest.\
"""


class LlmClient:
    def __init__(self) -> None:
        self.model_id = os.environ.get(
            "COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5"
        )

    async def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        body = {
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        payload = await complete_native(
            body, self.model_id, purpose="learner", timeout=120
        )
        text = payload.text.strip()
        if not text:
            raise ValueError("model returned empty text")
        return text


def _context(view: GameView) -> str:
    return json.dumps(
        {
            "turn": view.turn,
            "proposer": view.proposer,
            "your_seat": view.seat,
            "rules": view.rules,
            "state": view.state,
            "current_proposal": view.proposal,
            "debate_so_far": view.debates,
            "recent_turns": view.history[-4:],
        },
        ensure_ascii=False,
    )


class HaikuPolicy(Policy):
    """Haiku writes proposals, debate statements and votes."""

    def __init__(self) -> None:
        self.client = LlmClient()
        self.baseline = ScribePolicy()

    def introduce(self, view: GameView) -> str:
        return self.baseline.introduce(view)

    def action(self, view: GameView) -> str:
        return self.baseline.action(view)

    def _system(self, view: GameView) -> str:
        canonical = f"You are seat {view.seat} in a game of Gnomic with {view.num_players} players."
        system = SYSTEM.format(seat=view.seat, n=view.num_players)
        return (
            view.prompt_selection.introduction(
                canonical,
                f"seat {view.seat} in a game of Gnomic with {view.num_players} players",
            )
            + ("" if view.prompt_selection.selected == "canonical" else "\n\n")
            + view.prompt_selection.rules(
                system[len(canonical) :],
                (
                    " Players propose rule changes in turn. Debate precedes a vote by everyone; a Judge LLM enacts passed proposals and applies every rule each turn.",
                    "Reach the point victory threshold in common state 'victory_points', or hold the most points when the game ends, to win.",
                    "Be strategic: propose rules that favor you but can attract a majority; vote your interest.",
                ),
                (
                    " Rule changes are proposed on players' turns, debated, then voted on by all players. Passed changes are enacted by a Judge LLM, which applies all rules each turn.",
                    "Victory comes from reaching the points threshold recorded as 'victory_points' in common state, or from having the most points at game end.",
                    "Be strategic: propose rules that favor you but can attract a majority; vote your interest.",
                ),
            )
        )

    async def propose(self, view: GameView) -> dict:
        out = await self.client.complete(
            self._system(view),
            "It is your turn to propose one rule change. Reply with ONLY the proposal text "
            "(one or two sentences, imperative, unambiguous).\n\nGame context:\n"
            + _context(view),
            max_tokens=150,
        )
        return {
            "kind": "enact",
            "text": out,
            "rationale": "Haiku baseline proposal.",
        }

    async def debate(self, view: GameView) -> dict:
        out = await self.client.complete(
            self._system(view),
            "Debate the current proposal in at most two sentences (you speak once). "
            "Reply with ONLY your statement.\n\nGame context:\n" + _context(view),
            max_tokens=120,
        )
        support = self.baseline._supports(view)
        return {"text": out, "vote_intent": "aye" if support else "nay"}

    async def vote(self, view: GameView) -> str:
        out = await self.client.complete(
            self._system(view),
            "Vote on the current proposal. Reply with exactly one word: aye or nay.\n\n"
            "Game context:\n" + _context(view),
            max_tokens=8,
        )
        vote = out.lower()
        if vote not in {"aye", "nay"}:
            raise ValueError("vote must be aye or nay")
        return vote


if __name__ == "__main__":
    main_for(HaikuPolicy)
