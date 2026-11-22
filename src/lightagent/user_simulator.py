"""A deterministic user state machine, not a source of tasks or training data."""
from __future__ import annotations

from .models import Final


class DeterministicUser:
    def __init__(self, missing_slots: set[str], answers: dict[str, str]):
        if not missing_slots <= answers.keys():
            raise ValueError("每个缺失参数必须有预先配置的用户回答")
        self._remaining = set(missing_slots)
        self._answers = dict(answers)

    def respond(self, final: Final) -> str | None:
        requested = set(final.requested_slots)
        if final.status != "clarification" or not requested or not requested <= self._remaining:
            return None
        self._remaining -= requested
        return "；".join(self._answers[slot] for slot in sorted(requested))
