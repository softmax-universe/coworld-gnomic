"""A model decision uses the same prompt and ordinary player SDK as the exporter."""

from __future__ import annotations

import asyncio
import json

import pytest

from gnomic.engine import HOST_CONSTRAINTS, Board
from gnomic.llm_transport import Response
from gnomic.players.native import NativePolicy
from gnomic.players.posttrain_prompt import REPLIES, messages_for
from gnomic.players.sdk import GameView, InProcessTransport, PlayerSession
from gnomic.server.channel import InProcessChannel
from gnomic.server.config import GameConfig
from gnomic.server.episode import Episode, PlayerMoveError


def _phase_from_prompt(content: str) -> str:
    for phase, example in REPLIES.items():
        if content.endswith(example):
            return phase
    raise AssertionError(f"Prompt did not end with any known phase shape: {content[-120:]!r}")


def scripted_generation(
    messages: list[dict[str, str]], _deadline: float, *, fenced: bool = True
) -> str:
    content = messages[1]["content"]
    for forbidden in ("rid", "timeout_s", "action_window_s", "created_turn"):
        assert forbidden not in content
    phase = _phase_from_prompt(content)
    reply: dict = {}
    if phase == "introduce_request":
        reply["name"] = "Model Gnome"
    elif phase in {"action_request", "action_repair_request"}:
        reply["action"] = "pass"
    elif phase == "proposal_request":
        reply["proposal"] = {
            "kind": "enact",
            "text": "Every player gains 1 point after each turn.",
            "rationale": "A common rule.",
        }
    elif phase == "debate_request":
        reply.update({"text": "I support this common rule.", "vote_intent": "aye"})
    elif phase == "vote_request":
        reply["vote"] = "aye"
    raw = json.dumps(reply)
    return f"```json\n{raw}\n```" if fenced else raw


def test_prompt_is_plain_text_and_omits_protocol_metadata() -> None:
    board = Board.initial(seed=42)
    cfg = GameConfig.model_validate(
        {
            "tokens": ["a", "b", "c"],
            "players": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "judge_mode": "deterministic",
            "turns_max": 3,
        }
    )
    view = GameView(seat=0)
    view.fold({"type": "lobby", "seat": 0})
    view.fold({"type": "introduce_request", "rid": 1, "timeout_s": 15})
    intro = messages_for(view)
    assert intro[1]["content"].endswith(REPLIES["introduce_request"])
    assert "host_constraints" not in intro[1]["content"]
    assert "action_request" not in intro[1]["content"]

    view.fold(
        {
            "type": "game_start",
            "session": {
                "id": "gnomic-0000002a",
                "seats": [{"seat": i, "policy": f"G{i}"} for i in range(3)],
                "limits": cfg.limits_payload(),
            },
            "you": {"seat": 0},
            "host_constraints": list(HOST_CONSTRAINTS),
            "rules": board.rules_dict(include_history=False),
            "state": board.state.as_dict(),
            "history": [],
        }
    )
    view.fold(
        {
            "type": "turn_start",
            "turn": 1,
            "proposer": 0,
            "votes_required": 3,
            "rules": board.rules_dict(include_history=False),
            "state": board.state.as_dict(),
        }
    )
    view.fold({"type": "action_request", "turn": 1, "rid": 2, "timeout_s": 45, "attempt": 1})
    action_msgs = messages_for(view)
    user_text = action_msgs[1]["content"]
    assert user_text.endswith(REPLIES["action_request"])
    for noise in (
        "action_window_s",
        "proposal_window_s",
        "debate_window_s",
        "vote_window_s",
        "judge_window_s",
        "created_turn",
        "host_constraints",
        "proposal_request",
        "vote_request",
    ):
        assert noise not in user_text
    assert "Rule 101 (protected):" in user_text
    assert "Rule 201 (mutable):" in user_text


@pytest.mark.asyncio
@pytest.mark.parametrize("fenced", [False, True])
async def test_model_policy_completes_normal_three_seat_game(
    monkeypatch, fenced: bool
) -> None:
    async def complete(body, model, **kwargs):
        text = scripted_generation(
            [{"role": "system", "content": body["system"]}, *body["messages"]],
            0,
            fenced=fenced,
        )
        return Response(
            model=model,
            content=[{"type": "text", "text": text}],
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr("gnomic.players.native.complete_native", complete)
    channels = [InProcessChannel(seat) for seat in range(3)]
    sessions = [
        PlayerSession(NativePolicy(), InProcessTransport(channel))
        for channel in channels
    ]
    tasks = [asyncio.create_task(session.run()) for session in sessions]
    config = GameConfig.model_validate(
        {
            "tokens": ["a", "b", "c"],
            "players": [{"name": name} for name in ("Alpha", "Beta", "Gamma")],
            "judge_mode": "deterministic",
            "turns_max": 3,
            "action_window_s": 2,
            "proposal_window_s": 2,
            "debate_window_s": 2,
            "vote_window_s": 2,
            "judge_window_s": 2,
        }
    )
    results, replay = await Episode(config, channels, seed=42).run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)
    assert results["turns_played"] == 3
    assert all(not session.failures for session in sessions)
    assert all(
        event["action"] == {"text": "pass", "default": False}
        for event in replay["events"]
        if event["type"] == "action_made"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_text",
    [
        '{"action":"pass"} {"action":"different move"}',
        'I will analyze this turn carefully.\n\n{"action":"pass"}',
        '{"action":"pass", "extra": true}',
    ],
)
async def test_model_policy_rejects_invalid_or_ambiguous_reply_unscored(
    monkeypatch, bad_text: str
) -> None:
    async def complete(body, model, **kwargs):
        phase = _phase_from_prompt(body["messages"][0]["content"])
        if phase == "introduce_request":
            text = '{"name": "Model Gnome"}'
        else:
            text = bad_text
        return Response(
            model=model,
            content=[{"type": "text", "text": text}],
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr("gnomic.players.native.complete_native", complete)
    channels = [InProcessChannel(seat) for seat in range(3)]
    sessions = [
        PlayerSession(NativePolicy(), InProcessTransport(channel))
        for channel in channels
    ]
    tasks = [asyncio.create_task(session.run()) for session in sessions]
    config = GameConfig.model_validate(
        {
            "tokens": ["a", "b", "c"],
            "players": [{"name": name} for name in ("Alpha", "Beta", "Gamma")],
            "judge_mode": "deterministic",
            "turns_max": 1,
            "action_window_s": 2,
            "proposal_window_s": 2,
            "debate_window_s": 2,
            "vote_window_s": 2,
            "judge_window_s": 2,
        }
    )
    episode = Episode(config, channels, seed=42)
    with pytest.raises(PlayerMoveError) as exc_info:
        await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": None})
    await asyncio.gather(*tasks)
    assert exc_info.value.phase == "action"
    assert episode.results is None
    assert episode.move_failure is not None


@pytest.mark.asyncio
async def test_model_policy_provider_failure_terminates_unscored(monkeypatch) -> None:
    async def complete(body, model, **kwargs):
        phase = _phase_from_prompt(body["messages"][0]["content"])
        if phase == "introduce_request":
            return Response(
                model=model,
                content=[{"type": "text", "text": '{"name": "Model Gnome"}'}],
                usage={"input_tokens": 1, "output_tokens": 1},
            )
        raise ValueError("native Messages failed with status 503")

    monkeypatch.setattr("gnomic.players.native.complete_native", complete)
    channels = [InProcessChannel(seat) for seat in range(3)]
    sessions = [
        PlayerSession(NativePolicy(), InProcessTransport(channel))
        for channel in channels
    ]
    tasks = [asyncio.create_task(session.run()) for session in sessions]
    config = GameConfig.model_validate(
        {
            "tokens": ["a", "b", "c"],
            "players": [{"name": name} for name in ("Alpha", "Beta", "Gamma")],
            "judge_mode": "deterministic",
            "turns_max": 1,
            "action_window_s": 2,
            "proposal_window_s": 2,
            "debate_window_s": 2,
            "vote_window_s": 2,
            "judge_window_s": 2,
        }
    )
    episode = Episode(config, channels, seed=42)
    with pytest.raises(PlayerMoveError):
        await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": None})
    await asyncio.gather(*tasks)
    assert episode.results is None
    assert episode.move_failure is not None
