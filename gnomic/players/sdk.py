"""Player SDK: implement five callbacks and the harness ensures legal play.

    class MyPolicy(Policy):
        def introduce(self, view) -> str: ...
        def action(self, view) -> str: ...
        def propose(self, view) -> dict: ...
        def debate(self, view) -> dict: ...
        def vote(self, view) -> str:  # "aye" | "nay"

The harness owns transport, request IDs and deadlines. A failed callback sends
an empty response for its request ID so the host can adjudicate the failure.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import sys
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any

from gnomic.lifecycle import OwnershipUnsettled, bounded, main_owned, player_loop
from gnomic.llm_transport import AttemptPublisher, LearnerWindow, current_window

from ..prompts import PromptSelection
from ..protocol import make_reply

REPLY_MARGIN_S = 0.35


def stable_rng_int(*parts: object, mod: int) -> int:
    """Deterministic pseudo-random int (builtin hash() is per-process salted)."""
    basis = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(basis.encode()).hexdigest(), 16) % mod


# --- view -------------------------------------------------------------------


@dataclass
class GameView:
    """Folded game state: every prompt is answerable from the view alone."""

    prompt_selection: PromptSelection = field(default_factory=PromptSelection)
    seat: int = -1
    session: dict = field(default_factory=dict)
    host_constraints: list[str] = field(default_factory=list)
    rules: list[dict] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    turn: int = 0
    proposer: int = -1
    proposal: dict = field(default_factory=dict)
    debates: list[dict] = field(default_factory=list)
    votes: list[dict] = field(default_factory=list)
    last_ruling: dict = field(default_factory=dict)
    game_over: dict = field(default_factory=dict)
    request: dict = field(default_factory=dict)

    @property
    def num_players(self) -> int:
        return len(self.session.get("seats", [])) or 3

    @property
    def limits(self) -> dict:
        return self.session.get("limits", {})

    def my_points(self) -> int:
        players = self.state.get("players", [])
        if 0 <= self.seat < len(players):
            try:
                return int(players[self.seat].get("points", 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def fold(self, msg: dict) -> None:
        mtype = msg.get("type")
        if isinstance(mtype, str) and mtype.endswith("_request"):
            self.request = msg
        if mtype == "lobby":
            self.seat = msg["seat"]
            self.prompt_selection = PromptSelection.model_validate(
                msg.get("prompt_selection", {})
            )
        elif mtype == "game_start":
            self.session = msg.get("session", {})
            self.host_constraints = msg.get("host_constraints", [])
            self.seat = msg.get("you", {}).get("seat", self.seat)
            self.rules = msg.get("rules", [])
            self.state = msg.get("state", {})
            self.history = msg.get("history", [])
        elif mtype == "turn_start":
            self.turn = msg.get("turn", self.turn)
            self.proposer = msg.get("proposer", -1)
            self.rules = msg.get("rules", self.rules)
            self.state = msg.get("state", self.state)
            self.proposal = {}
            self.debates = []
            self.votes = []
            self.request = {}
        elif mtype == "proposal_made":
            self.proposal = msg.get("proposal", {})
        elif mtype == "action_ruling":
            self.state = msg.get("state", self.state)
        elif mtype == "debate_made":
            self.debates = list(msg.get("statements", []))
        elif mtype == "debate_request":
            self.proposal = msg.get("proposal", self.proposal)
        elif mtype == "vote_request":
            self.proposal = msg.get("proposal", self.proposal)
            self.debates = msg.get("debates", self.debates)
        elif mtype == "vote_reveal":
            self.votes = msg.get("votes", [])
        elif mtype == "judge_ruling":
            self.rules = msg.get("rules", self.rules)
            self.state = msg.get("state", self.state)
            self.last_ruling = msg
            self.history.append(
                {
                    "turn": msg.get("turn"),
                    "proposer": self.proposer,
                    "proposal": self.proposal,
                    "votes": self.votes,
                    "passed_vote": msg.get("passed_vote"),
                    "ruling": {
                        "summary": msg.get("summary", ""),
                        "source": msg.get("source", ""),
                    },
                }
            )
        elif mtype == "game_over":
            self.game_over = msg


# --- policy -----------------------------------------------------------------


PHASES = {
    "introduce_request": ("introduce", "name"),
    "action_request": ("action", "action"),
    "action_repair_request": ("action", "action"),
    "proposal_request": ("propose", "proposal"),
    "debate_request": ("debate", None),
    "vote_request": ("vote", "vote"),
}


class Policy:
    """Override the five decision callbacks."""

    def introduce(self, view: GameView) -> str | Coroutine[Any, Any, str]:
        raise NotImplementedError

    def action(self, view: GameView) -> str | Coroutine[Any, Any, str]:
        raise NotImplementedError

    def propose(self, view: GameView) -> dict | Coroutine[Any, Any, dict]:
        raise NotImplementedError

    def debate(self, view: GameView) -> dict | Coroutine[Any, Any, dict]:
        raise NotImplementedError

    def vote(self, view: GameView) -> str | Coroutine[Any, Any, str]:
        raise NotImplementedError

    async def reply(self, view: GameView) -> dict:
        phase, field = PHASES[view.request["type"]]
        answer = getattr(self, phase)(view)
        if inspect.isawaitable(answer):
            answer = await answer
        return {field: answer} if field else answer

    def on_message(self, view: GameView, raw: dict) -> None:
        """Optional observer for every inbound message."""


@dataclass
class PlayerFailure:
    turn: int
    phase: str
    reason: str


# --- transports ---------------------------------------------------------------


class Transport:
    async def recv(self) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError

    async def send(self, message: dict) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def close(self) -> None:
        pass


class InProcessTransport(Transport):
    """Player side of a server InProcessChannel (tests + conformance gate)."""

    def __init__(self, channel: Any) -> None:
        self.channel = channel

    async def recv(self) -> dict | None:
        return await self.channel.player_recv()

    async def send(self, message: dict) -> None:
        await self.channel.player_send(message)


class WebSocketTransport(Transport):
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None

    async def connect(self) -> None:
        import websockets

        self.ws = await websockets.connect(self.url, max_size=16 * 1024 * 1024)

    async def recv(self) -> dict | None:
        assert self.ws is not None
        raw = await anext(self.ws.__aiter__(), None)
        return None if raw is None else json.loads(raw)

    async def send(self, message: dict) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps(message))

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()


# --- session (the harness) -----------------------------------------------------


class PlayerSession:
    def __init__(self, policy: Policy, transport: Transport) -> None:
        self.policy = policy
        self.transport = transport
        self.view = GameView()
        self.failures: list[PlayerFailure] = []
        self.final: dict | None = None
        self.stopped = False
        self.cleanup_deadline = float("inf")

    async def _callback(self, timeout_s: float) -> dict:
        """Await native policy work under its owned absolute decision deadline."""
        budget = max(0.1, timeout_s - REPLY_MARGIN_S)
        view = copy.deepcopy(self.view)
        publish = AttemptPublisher(view.request["rid"], self.transport.send)
        window = LearnerWindow(
            view.seat, type(self.policy).__name__, time.monotonic() + budget, publish
        )
        token = current_window.set(window)
        try:
            return await bounded(
                self.policy.reply(view), asyncio.get_running_loop().time() + budget
            )
        finally:
            current_window.reset(token)

    async def handle(self, msg: dict) -> None:
        self.view.fold(msg)
        try:
            self.policy.on_message(self.view, msg)
        except Exception:  # noqa: BLE001 - observer hooks never crash the loop
            pass
        mtype = msg.get("type")
        if mtype == "final":
            self.final = msg
            return
        if mtype not in PHASES:
            return
        try:
            reply = await self._callback(msg["timeout_s"])
            if not isinstance(reply, dict) or "rid" in reply:
                raise ValueError(
                    "return one action object without a transport request ID"
                )
        except OwnershipUnsettled:
            raise
        except Exception as error:
            self.failures.append(
                PlayerFailure(self.view.turn, mtype, type(error).__name__)
            )
            reply = {}
        await self.transport.send(make_reply(msg["rid"], reply))

    async def run(self) -> None:
        """Consume messages until `final`. Treat any close after `final` as clean."""
        exit_state = await player_loop(
            self.transport.recv, self.transport.send, self.handle
        )
        self.stopped = exit_state.kind == "stopped"
        self.cleanup_deadline = exit_state.cleanup_deadline


async def run_ws_player(
    policy: Policy, url: str | None = None, *, max_attempts: int = 8
) -> None:
    """Connect to the game (reconnecting with backoff) and play until `final`."""
    url = url or os.environ["COWORLD_PLAYER_WS_URL"]
    session = PlayerSession(policy, WebSocketTransport(url))
    attempt = 0
    while session.final is None and not session.stopped:
        transport = WebSocketTransport(url)
        try:
            await transport.connect()
            attempt = 0
            session.transport = transport
            await session.run()
        except Exception as e:
            if isinstance(e, OwnershipUnsettled):
                raise
            attempt += 1
            if attempt >= max_attempts:
                print(
                    f"[sdk] giving up after {attempt} connection failures: {type(e).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            await asyncio.sleep(min(5.0, 0.5 * attempt))
        finally:
            await bounded(transport.close(), asyncio.get_running_loop().time() + 1)
    for event in session.failures:
        print(
            f"[sdk] failed turn={event.turn} phase={event.phase}: {event.reason}",
            file=sys.stderr,
            flush=True,
        )


def main_for(policy_factory) -> None:
    """Entrypoint helper for player containers."""
    main_owned(run_ws_player(policy_factory()))
