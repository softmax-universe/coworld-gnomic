from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gnomic.server.app import GameServer, build_app


def write_config(tmp_path) -> str:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "players": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                "judge_mode": "deterministic",
                "player_connect_timeout_seconds": 600,
            }
        )
    )
    return str(path)


def test_live_http_and_global_websocket_contract(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGAME_CONFIG_URI", write_config(tmp_path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    game = GameServer()
    with TestClient(build_app(game)) as client:
        assert client.get("/healthz").json() == {"status": "ok", "mode": "live"}
        for route in ("/client/global", "/client/player?slot=0&token=a", "/client/replay"):
            response = client.get(route)
            assert response.status_code == 200
            assert "GNOMIC MOOT" in response.text
            assert 'aria-label="Replay controls"' in response.text
        with client.websocket_connect("/global") as websocket:
            assert websocket.receive_json()["type"] == "hello"


def test_player_auth_rejects_wrong_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGAME_CONFIG_URI", write_config(tmp_path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    with TestClient(build_app(GameServer())) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/player?slot=0&token=wrong") as websocket:
                websocket.receive_json()


def test_replay_contract(tmp_path, monkeypatch) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"format": "gnomic-replay-v1", "events": []}))
    monkeypatch.setenv("COGAME_LOAD_REPLAY_URI", str(replay))
    monkeypatch.delenv("COGAME_CONFIG_URI", raising=False)
    with TestClient(build_app(GameServer())) as client:
        assert client.get("/healthz").json()["mode"] == "replay"
        with client.websocket_connect("/replay") as websocket:
            message = websocket.receive_json()
            assert message["type"] == "replay"
            assert message["data"]["format"] == "gnomic-replay-v1"


async def test_certification_probe_survives_fast_episode(tmp_path, monkeypatch) -> None:
    """The hosted certifier connects to /global before the episode starts but
    pings only later; a fast episode must not tear the socket down at done.
    Reproduces the 0.2.2 certification failure (cow_b12a3042, "no close frame
    received or sent"): with the old close-at-done behavior the ping below
    fails; with the spectator drain the pong arrives and the server still
    exits once the client leaves."""
    import socket

    from gnomic.players.baseline import BaselinePolicy
    from gnomic.players.client import run_policy

    import uvicorn
    import websockets

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "players": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                "judge_mode": "deterministic",
                "turns_max": 1,
                "action_window_s": 0.05,
                "proposal_window_s": 0.05,
                "debate_window_s": 0.05,
                "vote_window_s": 0.05,
                "judge_window_s": 1,
                "player_connect_timeout_seconds": 10,
                "introduce_window_s": 0.05,
                "episode_timeout_seconds": 60,
            }
        )
    )
    monkeypatch.setenv("COGAME_CONFIG_URI", str(path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_RESULTS_URI", raising=False)
    monkeypatch.delenv("COGAME_SAVE_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_LOG_URI", raising=False)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    game = GameServer()
    server = uvicorn.Server(uvicorn.Config(build_app(game), host="127.0.0.1", port=port, log_level="error"))
    game.server = server
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    async with websockets.connect(f"ws://127.0.0.1:{port}/global") as websocket:
        hello = json.loads(await asyncio.wait_for(websocket.recv(), 5))
        assert hello["type"] == "hello"
        await asyncio.gather(*(
            run_policy(BaselinePolicy(), f"ws://127.0.0.1:{port}/player?slot={slot}&token={token}")
            for slot, token in enumerate(("a", "b", "c"))
        ))
        assert game.fatal_error is None
        assert game.results is not None and sum(game.results["scores"]) == pytest.approx(1)
        while not game.done:
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.5)  # past the old 1s grace in which the server used to exit
        pong_waiter = await websocket.ping(b"coworld-certification-ping")
        await asyncio.wait_for(pong_waiter, 2)

    await asyncio.wait_for(serve_task, 10)


def test_fatal_operator_log_preserves_root_cause(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGAME_CONFIG_URI", write_config(tmp_path))
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    log_path = tmp_path / "operator.json"
    monkeypatch.setenv("COGAME_LOG_URI", str(log_path))
    game = GameServer()
    game.episode = SimpleNamespace(operator_log=lambda: {"turn": 7, "phase": "judge"})  # type: ignore[assignment]
    game.fatal_error = "JudgeError: model returned no text"

    game._write_operator_log()

    assert json.loads(log_path.read_text()) == {
        "turn": 7,
        "phase": "judge",
        "fatal_error": "JudgeError: model returned no text",
    }


@pytest.mark.parametrize(
    "mode,expected_kind,expected_seat",
    [
        ("accepted", None, None),
        ("malformed", "player_error", 0),
        ("unavailable", "episode_inconclusive", 0),
        ("malformed-seat-1", "player_error", 1),
    ],
)
async def test_native_player_failure_artifact_contract(
    tmp_path, monkeypatch, mode: str, expected_kind: str | None, expected_seat: int | None
) -> None:
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    import uvicorn

    from gnomic.players.baseline import BaselinePolicy
    from gnomic.players.client import run_policy
    from gnomic.players.native import NativePolicy
    from gnomic.players.sdk import run_ws_player

    def legal_reply(prompt: str, slot: str) -> str:
        if '{"name":' in prompt:
            return json.dumps({"name": f"Gnome-{slot}"})
        if '{"action":' in prompt:
            return json.dumps({"action": "pass"})
        if '{"proposal":' in prompt:
            return json.dumps(
                {"proposal": {"kind": "enact", "text": "Ceremonial rule for testing."}}
            )
        if '"vote_intent":' in prompt:
            return json.dumps({"text": "Coherent proposal.", "vote_intent": "aye"})
        return json.dumps({"vote": "aye"})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            slot = self.headers.get("X-Coworld-Player-Slot", "0")
            prompt = payload["messages"][-1]["content"]
            if mode == "unavailable" and slot == "0":
                raw = json.dumps({"error": "upstream unavailable"}).encode()
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if (mode == "malformed" and slot == "0") or (
                mode == "malformed-seat-1" and slot == "1" and '{"name":' not in prompt
            ):
                text = "not valid json"
            else:
                text = legal_reply(prompt, slot)
            raw = json.dumps(
                {
                    "model": payload["model"],
                    "content": [{"type": "text", "text": text}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args) -> None:
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=provider.serve_forever, daemon=True)
    thread.start()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tokens": ["a", "b", "c"],
                "players": [{"name": "native-0"}, {"name": "seat-1"}, {"name": "seat-2"}],
                "judge_mode": "deterministic",
                "turns_max": 1,
                "action_window_s": 5,
                "proposal_window_s": 5,
                "debate_window_s": 5,
                "vote_window_s": 5,
                "judge_window_s": 5,
                "introduce_window_s": 5,
                "player_connect_timeout_seconds": 10,
                "episode_timeout_seconds": 60,
            }
        )
    )
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.json"
    failure_path = tmp_path / "player_failure.json"
    log_path = tmp_path / "operator.json"

    monkeypatch.setenv("COGAME_CONFIG_URI", str(config_path))
    monkeypatch.setenv("COGAME_RESULTS_URI", f"file://{results_path}")
    monkeypatch.setenv("COGAME_SAVE_REPLAY_URI", f"file://{replay_path}")
    monkeypatch.setenv("COGAME_PLAYER_FAILURE_URI", f"file://{failure_path}")
    monkeypatch.setenv("COGAME_LOG_URI", f"file://{log_path}")
    monkeypatch.delenv("COGAME_LOAD_REPLAY_URI", raising=False)
    monkeypatch.delenv("COGAME_SAVE_TRAJECTORY_URI", raising=False)
    monkeypatch.setenv("COWORLD_LLM_ENDPOINT", f"http://127.0.0.1:{provider.server_port}")
    monkeypatch.setenv("COWORLD_LLM_MODEL", "fixture/native")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    game = GameServer()
    server = uvicorn.Server(
        uvicorn.Config(build_app(game), host="127.0.0.1", port=port, log_level="error")
    )
    game.server = server
    serve_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.05)

        if mode == "malformed-seat-1":
            seat_tasks = [
                run_ws_player(NativePolicy(), f"ws://127.0.0.1:{port}/player?slot=0&token=a"),
                run_ws_player(NativePolicy(), f"ws://127.0.0.1:{port}/player?slot=1&token=b"),
                run_policy(BaselinePolicy(), f"ws://127.0.0.1:{port}/player?slot=2&token=c"),
            ]
        else:
            seat_tasks = [
                run_ws_player(NativePolicy(), f"ws://127.0.0.1:{port}/player?slot=0&token=a"),
                run_policy(BaselinePolicy(), f"ws://127.0.0.1:{port}/player?slot=1&token=b"),
                run_policy(BaselinePolicy(), f"ws://127.0.0.1:{port}/player?slot=2&token=c"),
            ]
        await asyncio.gather(*seat_tasks)
        await asyncio.wait_for(serve_task, 10)
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)

    if mode == "accepted":
        assert game.fatal_error is None
        assert results_path.exists()
        assert replay_path.exists()
        assert not failure_path.exists()
        assert sum(json.loads(results_path.read_text())["scores"]) == pytest.approx(1.0)
    else:
        assert game.fatal_error == "PlayerMoveError"
        assert not results_path.exists()
        assert not replay_path.exists()
        assert failure_path.exists()
        emitted = json.loads(failure_path.read_text())
        assert set(emitted) == {"failed_policy_index", "message", "kind"}
        assert emitted["failed_policy_index"] == expected_seat
        assert emitted["kind"] == expected_kind
        assert 1 <= len(emitted["message"]) <= 2000
        if mode == "unavailable":
            assert "503" in emitted["message"]
        operator = json.loads(log_path.read_text())
        assert operator["move_failure"]["failed_seats"] == [expected_seat]
