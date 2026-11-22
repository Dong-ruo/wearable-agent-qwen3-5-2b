"""TRL environment_factory contract adapter; no trainer, data loading, or reward fitting.

This module intentionally does not import TRL. Test tool execution on the lightweight
runtime first. A real GRPO integration still needs version checks, a reward function,
chat-template verification and a training-host smoke test.
"""
from __future__ import annotations

import json
from typing import Callable

from ..environment import Environment
from ..models import Scenario
from ..goals import EpisodeSpec, GoalSpec
from ..rewards import RewardConfig
from ..user_simulator import DeterministicUser
from ..policy import SYSTEM


class TaskToolEnv:
    def __init__(self, scenario: Scenario | None = None, request: str = "", goal: GoalSpec | None = None,
                 reward_config: RewardConfig = RewardConfig()):
        self._scenario = scenario.model_copy(deep=True) if scenario is not None else None
        self._request = request
        self._env = Environment(self._scenario) if self._scenario is not None else None
        self._goal = goal
        self._reward_config = reward_config
        self._last_reward = None

    def reset(self, **kwargs) -> str | None:
        """Reset one rollout; never reuse writes from another rollout."""
        if "episode" in kwargs or "episode_json" in kwargs:
            episode = EpisodeSpec.model_validate_json(kwargs["episode_json"]) if "episode_json" in kwargs else EpisodeSpec.model_validate(kwargs["episode"])
            self._scenario, self._request, self._goal = episode.scenario, episode.request, episode.goal
            self._env = Environment(self._scenario)
            self._env._user = DeterministicUser(episode.missing_slots, episode.user_answers)
        elif self._scenario is not None:
            self._env = Environment(self._scenario)
        else:
            raise ValueError("必须提供外部 episode；当前不生成或自采样任务数据")
        self._last_reward = None
        # Only the user request is visible, never goals, answers, faults or test tags.
        return None if "episode_json" in kwargs else "\n用户任务：" + self._request

    def _execute(self, name: str, arguments: dict) -> str:
        return json.dumps(self._env.execute(name, arguments), ensure_ascii=False)

    def clock_now(self) -> str:
        """Read the simulated current date, local time and timezone."""
        return self._execute("clock_now", {})

    def time_resolve(self, expression: str) -> str:
        """Resolve a Chinese time expression using the episode clock.

        Args:
            expression: Explicit time such as 明天下午六点 or 2026-11-17 18:00.
        """
        return self._execute("time_resolve", {"expression": expression})

    def calendar_list(self, start: str, end: str) -> str:
        """Query events overlapping a half-open time interval.

        Args:
            start: Interval start in ISO 8601 with a UTC offset.
            end: Interval end in ISO 8601 with a UTC offset.
        """
        return self._execute("calendar_list", {"start": start, "end": end})

    def weather_get(self, city: str, date: str) -> str:
        """Query local simulated weather without assuming missing observations are sunny.

        Args:
            city: City explicitly provided by the user.
            date: Local date in YYYY-MM-DD format.
        """
        return self._execute("weather_get", {"city": city, "date": date})

    def calendar_find_slot(self, start: str, end: str, duration_minutes: int) -> str:
        """Find a free slot without creating an event or changing user consent.

        Args:
            start: Earliest allowed start in ISO 8601 with a UTC offset.
            end: End of the search window in ISO 8601 with a UTC offset.
            duration_minutes: Required activity duration in minutes.
        """
        return self._execute("calendar_find_slot", {"start": start, "end": end, "duration_minutes": duration_minutes})

    def calendar_create(self, title: str, activity: str, city: str, start: str,
                        duration_minutes: int, idempotency_key: str) -> str:
        """Create an event after checking conditions and obtaining any needed consent.

        Args:
            title: Human-readable event title.
            activity: Either outdoor_run or indoor_stretch.
            city: User-provided city.
            start: Activity start in ISO 8601 with a UTC offset.
            duration_minutes: Activity duration in minutes.
            idempotency_key: Stable key reused only with identical arguments.
        """
        return self._execute("calendar_create", {"title": title, "activity": activity, "city": city,
                             "start": start, "duration_minutes": duration_minutes, "idempotency_key": idempotency_key})

    def calendar_get(self, event_id: str) -> str:
        """Read the persisted event to verify a write.

        Args:
            event_id: ID returned by calendar_create.
        """
        return self._execute("calendar_get", {"event_id": event_id})

    def reminder_create(self, event_id: str, minutes_before: int, idempotency_key: str) -> str:
        """Create a simulated reminder for an existing event.

        Args:
            event_id: ID of an existing event.
            minutes_before: Nonnegative minutes before the event.
            idempotency_key: Stable key reused only with identical arguments.
        """
        return self._execute("reminder_create", {"event_id": event_id, "minutes_before": minutes_before,
                             "idempotency_key": idempotency_key})

    def reminder_get(self, reminder_id: str) -> str:
        """Read the persisted reminder to verify its time and linked event.

        Args:
            reminder_id: ID returned by reminder_create.
        """
        return self._execute("reminder_get", {"reminder_id": reminder_id})

    def calendar_update(self, event_id: str, idempotency_key: str, title: str | None = None,
                        activity: str | None = None, city: str | None = None, start: str | None = None,
                        duration_minutes: int | None = None, expected_revision: int | None = None) -> str:
        """Update a selected event atomically and move any linked reminder.

        Args:
            event_id: Event explicitly selected by the user.
            idempotency_key: Stable operation key.
            title: Replacement title, if requested.
            activity: Replacement activity, if requested.
            city: Replacement city, if requested.
            start: New ISO 8601 start with a UTC offset.
            duration_minutes: New duration in minutes.
            expected_revision: Revision observed by calendar_get, if supplied.
        """
        return self._execute("calendar_update", {"event_id": event_id, "idempotency_key": idempotency_key,
            "title": title, "activity": activity, "city": city, "start": start,
            "duration_minutes": duration_minutes, "expected_revision": expected_revision})

    def calendar_delete(self, event_id: str, idempotency_key: str, expected_revision: int | None = None) -> str:
        """Delete a user-selected event and its reminders.

        Args:
            event_id: Event the user explicitly asked to delete.
            idempotency_key: Stable operation key.
            expected_revision: Previously observed revision, if supplied.
        """
        return self._execute("calendar_delete", {"event_id": event_id, "idempotency_key": idempotency_key, "expected_revision": expected_revision})

    def activity_search(self, indoor: bool | None = None, max_duration_minutes: int = 240) -> str:
        """Filter the activity catalog without selecting a plan.

        Args:
            indoor: Indoor or outdoor filter; omit for both.
            max_duration_minutes: Maximum acceptable catalog duration.
        """
        return self._execute("activity_search", {"indoor": indoor, "max_duration_minutes": max_duration_minutes})

    def preferences_get(self) -> str:
        """Read configured preferences without guessing missing values."""
        return self._execute("preferences_get", {})

    def user_ask(self, question: str, slots: list[str]) -> str:
        """Ask the deterministic user only for necessary missing information.

        Args:
            question: Clear clarification question.
            slots: Names of the missing fields actually asked about.
        """
        return self._execute("user_ask", {"question": question, "slots": slots})


def make_environment_factory(scenario: Scenario, request: str) -> Callable[[], TaskToolEnv]:
    """Capture config in a zero-argument factory compatible with TRL's discovery model."""
    frozen = scenario.model_copy(deep=True)

    def factory() -> TaskToolEnv:
        return TaskToolEnv(frozen, request)

    return factory
