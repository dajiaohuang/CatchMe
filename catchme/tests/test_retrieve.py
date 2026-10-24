"""Tests for catchme.pipelines.retrieve helpers."""

from __future__ import annotations

from datetime import datetime

import pytest

import catchme.pipelines.retrieve as retrieve_pipeline
from catchme.pipelines.retrieve import _resolve_time_range, _sessions_in_range


def _make_day(sessions: list[dict]) -> dict:
    return {"node_id": "day_1", "kind": "day", "children": sessions}


def _sess(start_str: str, end_str: str) -> dict:
    """Build a minimal session dict from 'YYYY-MM-DD HH:MM' strings."""
    fmt = "%Y-%m-%d %H:%M"
    return {
        "node_id": f"s_{start_str}",
        "kind": "session",
        "start": datetime.strptime(start_str, fmt).timestamp(),
        "end": datetime.strptime(end_str, fmt).timestamp(),
    }


class TestTimeRangeResolution:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("start_hour", -1),
            ("start_hour", 24),
            ("start_hour", 25),
            ("end_hour", -1),
            ("end_hour", 25),
            ("start_hour", True),
            ("end_hour", float("inf")),
        ],
    )
    def test_invalid_llm_hours_are_ignored(self, monkeypatch, field, value):
        response = {
            "has_time": True,
            "dates": None,
            "reasoning": "synthetic invalid hour",
            "start_hour": 6,
            "end_hour": 18,
        }
        response[field] = value
        monkeypatch.setattr(retrieve_pipeline, "_load_retrieve_config", lambda: {})
        monkeypatch.setattr(retrieve_pipeline, "_get_prompt", lambda _key: "{query}")
        monkeypatch.setattr(
            retrieve_pipeline,
            "_llm_json",
            lambda *_args, **_kwargs: response.copy(),
        )

        resolved = _resolve_time_range("synthetic query", object())
        day = _make_day([_sess("2026-03-23 08:00", "2026-03-23 09:00")])
        _sessions_in_range(day, resolved["start_hour"], resolved["end_hour"])

        assert resolved[field] is None

    def test_end_hour_24_is_preserved(self, monkeypatch):
        response = {
            "has_time": True,
            "dates": None,
            "start_hour": 18,
            "end_hour": 24,
        }
        monkeypatch.setattr(retrieve_pipeline, "_load_retrieve_config", lambda: {})
        monkeypatch.setattr(retrieve_pipeline, "_get_prompt", lambda _key: "{query}")
        monkeypatch.setattr(
            retrieve_pipeline,
            "_llm_json",
            lambda *_args, **_kwargs: response.copy(),
        )

        resolved = _resolve_time_range("synthetic query", object())

        assert resolved["end_hour"] == 24


class TestSessionsInRange:
    """Unit tests for _sessions_in_range, including cross-midnight windows."""

    def test_no_filter_returns_all(self):
        day = _make_day([_sess("2026-03-23 10:00", "2026-03-23 12:00")])
        assert len(_sessions_in_range(day, None, None)) == 1

    def test_normal_range(self):
        s1 = _sess("2026-03-23 09:00", "2026-03-23 10:00")
        s2 = _sess("2026-03-23 14:00", "2026-03-23 16:00")
        s3 = _sess("2026-03-23 20:00", "2026-03-23 22:00")
        day = _make_day([s1, s2, s3])

        result = _sessions_in_range(day, 12, 18)
        assert len(result) == 1
        assert result[0]["node_id"] == s2["node_id"]

    def test_normal_range_boundary(self):
        """Session ending exactly at window start should NOT match."""
        s = _sess("2026-03-23 10:00", "2026-03-23 12:00")
        day = _make_day([s])
        assert len(_sessions_in_range(day, 12, 18)) == 0

    def test_cross_midnight_late_evening(self):
        """22-06 window should match a session at 23:00."""
        s = _sess("2026-03-23 23:00", "2026-03-23 23:45")
        day = _make_day([s])
        result = _sessions_in_range(day, 22, 6)
        assert len(result) == 1

    def test_cross_midnight_early_morning(self):
        """22-06 window should match a session at 01:00 (same calendar day)."""
        s = _sess("2026-03-24 01:00", "2026-03-24 02:00")
        day = _make_day([s])
        result = _sessions_in_range(day, 22, 6)
        assert len(result) == 1

    def test_cross_midnight_excludes_midday(self):
        """22-06 window should NOT match a session at 14:00."""
        s = _sess("2026-03-23 14:00", "2026-03-23 16:00")
        day = _make_day([s])
        result = _sessions_in_range(day, 22, 6)
        assert len(result) == 0

    def test_cross_midnight_22_02(self):
        """The exact scenario from the bug report: 22-02 window."""
        s_late = _sess("2026-03-23 22:30", "2026-03-23 23:30")
        s_early = _sess("2026-03-24 00:30", "2026-03-24 01:30")
        s_mid = _sess("2026-03-23 15:00", "2026-03-23 17:00")

        day_23 = _make_day([s_late, s_mid])
        result_23 = _sessions_in_range(day_23, 22, 2)
        assert len(result_23) == 1
        assert result_23[0]["node_id"] == s_late["node_id"]

        day_24 = _make_day([s_early])
        result_24 = _sessions_in_range(day_24, 22, 2)
        assert len(result_24) == 1
        assert result_24[0]["node_id"] == s_early["node_id"]

    def test_end_hour_24(self):
        """end_hour=24 means up to midnight — should work like before."""
        s = _sess("2026-03-23 23:00", "2026-03-23 23:59")
        day = _make_day([s])
        result = _sessions_in_range(day, 18, 24)
        assert len(result) == 1

    def test_same_start_end_returns_nothing(self):
        """sh == eh means zero-width window — nothing should match."""
        s = _sess("2026-03-23 10:00", "2026-03-23 12:00")
        day = _make_day([s])
        assert len(_sessions_in_range(day, 10, 10)) == 0
