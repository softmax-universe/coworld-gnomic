"""Regression test: legacy WebSocket players without stop ACKs settle cleanly under private capture + connected spectator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import uvicorn
import websockets

import gnomic.server.app as app_mod
from gnomic.server.app import GameServer, build_app


@pytest.mark.asyncio
async def test_legacy_websocket_players_settle_with_spectator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = {
        "tokens": ["t0", "t1", "t2"],
        "players": [{"name": "p0"}, {"name": "p1"}, {"name": "p2"}],
        "turns_max": 1,
        "judge_mode": "deterministic",
        "player_connect_timeout_seconds": 30.0,
        "introduce_window_s": 0.2,
        "action_window_s": 0.2,
        "proposal_window_s": 0.2,
        "debate_window_s": 0.2,
        "vote_window_s": 0.2,
    }
    cfg_path = tmp_path / "cfg.json"
    res_path = tmp_path / "results.json"
    traj_path = tmp_path / "traj.json"
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv("COGAME_CONFIG_URI", f"file://{cfg_path}")
    monkeypatch.setenv("COGAME_RESULTS_URI", f"file://{res_path}")
    monkeypatch.setenv("COGAME_SAVE_TRAJECTORY_URI", f"file://{traj_path}")
    monkeypatch.setenv("COWORLD_EPISODE_ID", "ep-legacy-ws")
    monkeypatch.setenv("COWORLD_GAME_VERSION", "0.1.9")
    monkeypatch.setenv("COWORLD_SOURCE_REVISION", "a" * 40)
    monkeypatch.setattr(app_mod, "SPECTATOR_DRAIN_SECONDS", 0.3)

    gs = GameServer()
    app = build_app(gs)
    u_cfg = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    u_cfg.timeout_graceful_shutdown = 1
    srv = uvicorn.Server(u_cfg)
    gs.server = srv
    srv_task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.02)
    port = srv.servers[0].sockets[0].getsockname()[1]

    async def legacy_player(slot: int, tok: str) -> None:
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/player?slot={slot}&token={tok}"
        ) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type", "")
                rid = msg.get("rid", 1)
                if mtype == "introduce_request":
                    await ws.send(json.dumps({"rid": rid, "name": f"P{slot}"}))
                elif mtype == "action_request":
                    await ws.send(json.dumps({"rid": rid, "action": "pass"}))
                elif mtype == "proposal_request":
                    await ws.send(
                        json.dumps(
                            {
                                "rid": rid,
                                "proposal": {"kind": "enact", "text": "Harmless rule."},
                            }
                        )
                    )
                elif mtype == "debate_request":
                    await ws.send(
                        json.dumps({"rid": rid, "text": "Ok.", "vote_intent": "aye"})
                    )
                elif mtype == "vote_request":
                    await ws.send(json.dumps({"rid": rid, "vote": "aye"}))
                elif mtype == "final":
                    break

    spec_ws = await websockets.connect(f"ws://127.0.0.1:{port}/global")
    try:
        players = [asyncio.create_task(legacy_player(i, f"t{i}")) for i in range(3)]
        await asyncio.gather(*players)
        await srv_task
        await gs.stop()
    finally:
        await spec_ws.close()

    assert gs.ownership_joined is True
    assert gs.fatal_error is None
    assert res_path.is_file()
    traj = json.loads(traj_path.read_text())
    assert traj["episode"]["status"] == "completed"
