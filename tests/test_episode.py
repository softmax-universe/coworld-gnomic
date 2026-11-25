from __future__ import annotations

import asyncio

import pytest

from gnomic.players.baseline import BaselinePolicy
from gnomic.judge import ActionRuling, DeterministicJudge
from gnomic.server.channel import InProcessChannel
from gnomic.server.config import GameConfig
from gnomic.server.episode import Episode, PlayerMoveError


def config(**overrides) -> GameConfig:
    values = {
        "tokens": ["a", "b", "c"],
        "players": [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}],
        "judge_mode": "deterministic",
        "proposal_window_s": 0.5,
        "debate_window_s": 0.5,
        "vote_window_s": 0.5,
        "judge_window_s": 2,
        "introduce_window_s": 0.2,
        "episode_timeout_seconds": 60,
    }
    values.update(overrides)
    return GameConfig.model_validate(values)


async def drive(channel: InProcessChannel) -> None:
    policy = BaselinePolicy()
    while True:
        message = await channel.player_recv()
        if message.get("type") == "final":
            return
        reply = await policy.respond(message)
        if reply is not None:
            await channel.player_send(reply)


@pytest.mark.asyncio
async def test_full_certification_episode_reaches_circuit_victory() -> None:
    channels = [InProcessChannel(i) for i in range(3)]
    tasks = [asyncio.create_task(drive(channel)) for channel in channels]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(config(), channels, seed=42, broadcast=broadcast)
    results, replay = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)

    assert results["termination"] == "constitution_victory"
    assert results["turns_played"] % 3 == 0
    assert results["turns_played"] <= 45
    assert all(results["game_points"][seat] >= 100 for seat in results["winner_slots"])
    assert sum(results["scores"]) == pytest.approx(1)
    assert replay["format"] == "gnomic-replay-v1"
    assert replay["judge_usage"]["calls"] == 0
    assert sum(event["type"] == "action_made" for event in emitted) == results["turns_played"]
    assert sum(event["type"] == "debate_made" for event in emitted) == results["turns_played"]


@pytest.mark.asyncio
async def test_cap_produces_three_way_co_win_when_every_vote_fails() -> None:
    channels = [InProcessChannel(i) for i in range(3)]

    async def nay_driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            reply = await policy.respond(message)
            if reply is not None:
                if message["type"] == "vote_request":
                    reply["vote"] = "nay"
                await channel.player_send(reply)

    tasks = [asyncio.create_task(nay_driver(channel)) for channel in channels]
    results, _ = await Episode(config(turns_max=2), channels, seed=1).run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)
    points = results["game_points"]
    assert results["winner_slots"] == [seat for seat, value in enumerate(points) if value == max(points)]
    assert sum(results["scores"]) == pytest.approx(1)
    assert results["termination"] == "turn_cap"


@pytest.mark.asyncio
async def test_rejected_action_gets_one_player_repair_attempt() -> None:
    channels = [InProcessChannel(i) for i in range(3)]

    async def action_driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            reply = await policy.respond(message)
            if message.get("type") == "action_request":
                reply = {"rid": message["rid"], "action": "I steal every point without a rule."}
            if reply is not None:
                await channel.player_send(reply)

    class RepairJudge(DeterministicJudge):
        async def act(self, board, action_record, turn, turns_max):  # type: ignore[override]
            return (
                ActionRuling(
                    valid=False,
                    summary="No active rule authorizes stealing points.",
                    state_ops=[],
                    winner_slots=[],
                ),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0},
            )

    tasks = [asyncio.create_task(action_driver(channel)) for channel in channels]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(config(turns_max=1), channels, seed=3, broadcast=broadcast)
    episode.judge = RepairJudge()
    results, _ = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)

    actions = [event for event in emitted if event["type"] == "action_made"]
    rulings = [event for event in emitted if event["type"] == "action_ruling"]
    assert [event["attempt"] for event in actions] == [1, 2]
    assert actions[1]["action"]["text"] == "pass"
    assert [event["valid"] for event in rulings] == [False, True]
    assert any(event["type"] == "proposal_made" for event in emitted)


@pytest.mark.asyncio
async def test_introductions_set_gnome_names_with_owner_attribution_for_spectators() -> None:
    channels = [InProcessChannel(i) for i in range(3)]
    names = ["Yura", "Yura", ""]  # duplicate claim + one silent seat

    async def introducer(channel: InProcessChannel, name: str) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            if message.get("type") == "introduce_request":
                if name:
                    await channel.player_send({"rid": message["rid"], "name": name})
                continue
            reply = await policy.respond(message)
            if reply is not None:
                await channel.player_send(reply)

    tasks = [
        asyncio.create_task(introducer(channel, names[seat]))
        for seat, channel in enumerate(channels)
    ]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(config(turns_max=3), channels, seed=7, broadcast=broadcast)
    results, replay = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)

    # Duplicates are suffixed; the silent seat 2 falls back to its house default
    # (also Yura), taking the third suffix.
    assert episode.seat_names == ["Yura", "Yura (2)", "Yura (3)"]

    # Replay and the spectator game_start carry the owning player; agents never see it.
    assert all("player" in row for row in replay["players"])
    spectator_start = next(e for e in emitted if e["type"] == "game_start")
    assert all("player" in row for row in spectator_start["session"]["seats"])


@pytest.mark.asyncio
async def test_player_facing_game_start_never_reveals_seat_ownership() -> None:
    channels = [InProcessChannel(i) for i in range(3)]
    player_starts: list[dict] = []

    async def driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            if message.get("type") == "game_start":
                player_starts.append(message)
            reply = await policy.respond(message)
            if reply is not None:
                await channel.player_send(reply)

    tasks = [asyncio.create_task(driver(channel)) for channel in channels]
    episode = Episode(config(turns_max=3), channels, seed=9)
    results, _ = await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": results["scores"]})
    await asyncio.gather(*tasks)

    assert len(player_starts) == 3
    for start in player_starts:
        seats = start["session"]["seats"]
        assert [row["policy"] for row in seats] == ["Ivan", "Anton", "Yura"]
        assert all("player" not in row for row in seats)
        assert "Alpha" not in str(seats)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,bad_reply",
    [
        ("action", {"text": "pass"}),
        ("action", {"action": {"text": "pass"}}),
        ("action", {"action": "   "}),
        ("proposal", {"kind": "enact", "text": "Unwrapped alias."}),
        ("proposal", {"proposal": {"kind": "amend", "rule_id": True, "text": "Bool id."}}),
        ("proposal", {"proposal": {"kind": "enact", "text": "Extra.", "extra": 1}}),
        ("debate", {"text": "Bad intent.", "vote_intent": "maybe"}),
        ("debate", {"text": "   ", "vote_intent": "aye"}),
        ("vote", {"vote": "abstain"}),
        ("vote", {"vote": "aye", "extra": 1}),
    ],
)
async def test_invalid_or_alias_move_fails_explicitly_without_default(
    phase: str, bad_reply: dict
) -> None:
    channels = [InProcessChannel(i) for i in range(3)]
    target_request = f"{phase}_request"
    failing_seat = 1 if phase == "debate" else 0

    async def driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            if message.get("type") == target_request and channel.seat == failing_seat:
                await channel.player_send({"rid": message["rid"], **bad_reply})
                continue
            reply = await policy.respond(message)
            if reply is not None:
                await channel.player_send(reply)

    tasks = [asyncio.create_task(driver(channel)) for channel in channels]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(config(turns_max=3), channels, seed=11, broadcast=broadcast)
    with pytest.raises(PlayerMoveError) as exc_info:
        await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": None})
    await asyncio.gather(*tasks)

    err = exc_info.value
    assert episode.results is None
    assert err.phase == phase
    assert err.turn == 1
    assert err.failed_seats == [failing_seat]
    assert err.failures[0]["reason"] == "invalid_reply"
    assert episode.move_failure == {
        "turn": 1,
        "phase": phase,
        "failed_seats": [failing_seat],
        "failures": err.failures,
    }
    assert episode.operator_log()["move_failure"] == episode.move_failure
    forbidden_event = {
        "action": "action_made",
        "proposal": "proposal_made",
        "debate": "debate_made",
        "vote": "vote_reveal",
    }[phase]
    assert not any(event["type"] == forbidden_event for event in emitted)
    assert not any(event["type"] == "game_over" for event in emitted)


@pytest.mark.asyncio
async def test_action_repair_timeout_fails_after_preserving_attempt_one() -> None:
    channels = [InProcessChannel(i) for i in range(3)]

    async def driver(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            if message.get("type") == "action_request":
                await channel.player_send(
                    {"rid": message["rid"], "action": "I take 50 points."}
                )
                continue
            if message.get("type") == "action_repair_request":
                continue
            reply = await policy.respond(message)
            if reply is not None:
                await channel.player_send(reply)

    class RejectJudge(DeterministicJudge):
        async def act(self, board, action_record, turn, turns_max):  # type: ignore[override]
            return (
                ActionRuling(
                    valid=False,
                    summary="Unauthorized action.",
                    state_ops=[],
                    winner_slots=[],
                ),
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0},
            )

    tasks = [asyncio.create_task(driver(channel)) for channel in channels]
    emitted: list[dict] = []

    async def broadcast(message: dict) -> None:
        emitted.append(message)

    episode = Episode(
        config(turns_max=1, action_window_s=0.05), channels, seed=5, broadcast=broadcast
    )
    episode.judge = RejectJudge()
    with pytest.raises(PlayerMoveError) as exc_info:
        await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": None})
    await asyncio.gather(*tasks)

    assert episode.results is None
    assert exc_info.value.phase == "action_repair"
    assert exc_info.value.failed_seats == [0]
    assert exc_info.value.failures[0]["reason"] == "timeout"
    actions = [event for event in emitted if event["type"] == "action_made"]
    assert [event["attempt"] for event in actions] == [1]
    assert not any(event["type"] == "proposal_made" for event in emitted)
    assert not any(event["type"] == "game_over" for event in emitted)


@pytest.mark.asyncio
async def test_simultaneous_debate_and_vote_failures_record_all_failing_seats_without_scores() -> None:
    channels = [InProcessChannel(i) for i in range(3)]

    async def bad_debaters(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            if message.get("type") == "debate_request":
                continue
            reply = await policy.respond(message)
            if reply is not None:
                await channel.player_send(reply)

    tasks = [asyncio.create_task(bad_debaters(channel)) for channel in channels]
    episode = Episode(config(turns_max=1, debate_window_s=0.05), channels, seed=13)
    with pytest.raises(PlayerMoveError) as exc_info:
        await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": None})
    await asyncio.gather(*tasks)

    assert episode.results is None
    assert exc_info.value.phase == "debate"
    assert exc_info.value.failed_seats == [1, 2]
    assert [f["reason"] for f in exc_info.value.failures] == ["timeout", "timeout"]

    channels = [InProcessChannel(i) for i in range(3)]

    async def bad_voters(channel: InProcessChannel) -> None:
        policy = BaselinePolicy()
        while True:
            message = await channel.player_recv()
            if message.get("type") == "final":
                return
            if message.get("type") == "vote_request":
                await channel.player_send({"rid": message["rid"], "vote": "invalid"})
                continue
            reply = await policy.respond(message)
            if reply is not None:
                await channel.player_send(reply)

    tasks = [asyncio.create_task(bad_voters(channel)) for channel in channels]
    episode = Episode(config(turns_max=1, vote_window_s=0.05), channels, seed=14)
    with pytest.raises(PlayerMoveError) as exc_info:
        await episode.run()
    for channel in channels:
        await channel.send({"type": "final", "scores": None})
    await asyncio.gather(*tasks)

    assert episode.results is None
    assert exc_info.value.phase == "vote"
    assert exc_info.value.failed_seats == [0, 1, 2]
    assert [f["reason"] for f in exc_info.value.failures] == [
        "invalid_reply",
        "invalid_reply",
        "invalid_reply",
    ]


@pytest.mark.asyncio
async def test_disconnected_seat_fails_without_inventing_scores() -> None:
    from gnomic.server.channel import NullSeatChannel

    channels = [NullSeatChannel(0), InProcessChannel(1), NullSeatChannel(2)]
    task = asyncio.create_task(drive(channels[1]))  # type: ignore[arg-type]
    episode = Episode(config(turns_max=1), channels, seed=15)
    with pytest.raises(PlayerMoveError) as exc_info:
        await episode.run()
    await channels[1].send({"type": "final", "scores": None})
    await task

    assert episode.results is None
    assert exc_info.value.phase == "action"
    assert exc_info.value.failed_seats == [0]
    assert exc_info.value.failures[0]["reason"] == "disconnected"


def test_player_move_error_failure_payload_distinguishes_player_and_provider_causes() -> None:
    player_err = PlayerMoveError(
        1,
        "debate",
        [
            {
                "seat": 2,
                "rid": 5,
                "reason": "invalid_model_reply",
                "detail": "Model returned a response that failed player validation.",
            }
        ],
    )
    payload = player_err.player_failure_payload()
    assert set(payload) == {"failed_policy_index", "message", "kind"}
    assert payload["failed_policy_index"] == 2
    assert payload["kind"] == "player_error"
    assert 1 <= len(payload["message"]) <= 2000

    provider_err = PlayerMoveError(
        1,
        "vote",
        [
            {
                "seat": 1,
                "rid": 7,
                "reason": "provider_error",
                "detail": "Model provider returned HTTP 503.",
            }
        ],
    )
    inconclusive = provider_err.player_failure_payload()
    assert set(inconclusive) == {"failed_policy_index", "message", "kind"}
    assert inconclusive["failed_policy_index"] == 1
    assert inconclusive["kind"] == "episode_inconclusive"
    assert "HTTP 503" in inconclusive["message"]
