"""One player-visible prompt for Gnomic post-training and model inference."""

from __future__ import annotations

import json
from typing import Any

from .sdk import GameView

REPLIES = {
    "introduce_request": '{"name": "your gnome name"}',
    "action_request": '{"action": "your natural-language move"}',
    "action_repair_request": '{"action": "your corrected natural-language move"}',
    "proposal_request": '{"proposal": {"kind": "enact", "text": "your new rule"}}',
    "debate_request": '{"text": "your debate statement", "vote_intent": "aye"}',
    "vote_request": '{"vote": "aye"}',
}

PHASE_INSTRUCTIONS = {
    "introduce_request": "Choose a short gnome name for this game.",
    "action_request": 'State one natural-language game action for this turn (or "pass").',
    "action_repair_request": (
        'Your previous action was rejected. State one corrected natural-language game action (or "pass").'
    ),
    "proposal_request": (
        "Propose one rule change: enact (text), amend (rule_id and text), "
        "repeal (rule_id), or transmute a protected rule (rule_id)."
    ),
    "debate_request": 'Give a short debate statement on the proposal and your vote_intent ("aye" or "nay").',
    "vote_request": 'Cast your secret vote ("aye" or "nay") on the proposal.',
}


def observation_for(view: GameView) -> dict[str, Any]:
    """Structured player-visible observation for private capture."""
    return {
        "seat": view.seat,
        "turn": view.turn,
        "proposer": view.proposer,
        "rules": view.rules,
        "state": view.state,
        "proposal": view.proposal,
        "debates": view.debates,
        "recent_turns": view.history[-2:],
        "request": {
            k: v for k, v in view.request.items() if k not in {"rid", "timeout_s"}
        },
    }


def _fmt_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_user_prompt(view: GameView) -> str:
    kind = view.request["type"]
    lines: list[str] = []

    if kind == "introduce_request":
        lines.append(f"Seat: {view.seat}")
    else:
        turns_max = view.limits.get("turns_max")
        turn_label = f"{view.turn}/{turns_max}" if turns_max is not None else str(view.turn)
        lines.append(f"Seat: {view.seat} | Turn: {turn_label} | Proposer: seat {view.proposer}")

        players = view.state.get("players", [])
        if players:
            lines.append("")
            lines.append("Players:")
            for seat_idx, pdata in enumerate(players):
                if isinstance(pdata, dict):
                    fields = ", ".join(f"{k}={_fmt_value(v)}" for k, v in pdata.items())
                else:
                    fields = _fmt_value(pdata)
                lines.append(f"- seat {seat_idx}: {fields}")

        common = view.state.get("common", {})
        if isinstance(common, dict) and common:
            lines.append("")
            lines.append("Public state:")
            for k, v in common.items():
                lines.append(f"- {k}: {_fmt_value(v)}")

        active_rules = [r for r in view.rules if r.get("active", True)]
        if active_rules:
            lines.append("")
            lines.append("Active rules:")
            for rule in active_rules:
                status = "mutable" if rule.get("mutable", True) else "protected"
                lines.append(f"- Rule {rule['id']} ({status}): {rule['text']}")

        recent = view.history[-2:]
        if recent:
            lines.append("")
            lines.append("Recent turns:")
            for item in recent:
                prop = item.get("proposal") or {}
                pkind = prop.get("kind", "")
                ptext = prop.get("text", "")
                prid = prop.get("rule_id")
                desc = pkind
                if prid is not None:
                    desc += f" rule {prid}"
                if ptext:
                    desc += f": {ptext}"
                summary = (item.get("ruling") or {}).get("summary", "")
                outcome = "passed" if item.get("passed_vote") else "failed"
                lines.append(
                    f"- Turn {item.get('turn')} (proposer seat {item.get('proposer')}): "
                    f"{desc} -> {outcome}" + (f" ({summary})" if summary else "")
                )

        if kind == "action_repair_request":
            lines.append("")
            original = view.request.get("original_action")
            if isinstance(original, dict) and "text" in original:
                lines.append(f"Previous action: {original['text']}")
            elif original is not None:
                lines.append(f"Previous action: {_fmt_value(original)}")
            reason = view.request.get("rejection_reason")
            if reason:
                lines.append(f"Rejection reason: {reason}")

        if kind in {"debate_request", "vote_request"} and view.proposal:
            lines.append("")
            lines.append(f"Proposal by seat {view.proposer}:")
            if "kind" in view.proposal:
                lines.append(f"- kind: {view.proposal['kind']}")
            if "rule_id" in view.proposal:
                lines.append(f"- rule_id: {view.proposal['rule_id']}")
            if view.proposal.get("text"):
                lines.append(f"- text: {view.proposal['text']}")
            if view.proposal.get("rationale"):
                lines.append(f"- rationale: {view.proposal['rationale']}")
            votes_req = view.request.get("votes_required")
            if votes_req is not None:
                lines.append(f"- votes_required: {votes_req}")

        if kind == "vote_request" and view.debates:
            lines.append("")
            lines.append("Debate:")
            for stmt in view.debates:
                lines.append(f"- seat {stmt.get('seat')}: {stmt.get('text', '')}")

    lines.append("")
    lines.append(PHASE_INSTRUCTIONS[kind])
    lines.append("Reply with ONLY one JSON object and no other text:")
    lines.append(REPLIES[kind])
    return "\n".join(lines)


def messages_for(view: GameView) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": view.prompt_selection.rules(
                "You are a Gnomic player. Reply with ONLY one JSON object and no other text.",
                (
                    "Play Gnomic using the information visible to your seat.",
                    "Reply with ONLY one JSON object and no other text.",
                ),
                (
                    "Choose your Gnomic move based on what your seat can see.",
                    "Output only one JSON object with no commentary.",
                ),
            ),
        },
        {
            "role": "user",
            "content": _format_user_prompt(view),
        },
    ]
