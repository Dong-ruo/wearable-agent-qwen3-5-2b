from datetime import datetime

import pytest

from lightagent.timeparse import TimeResolutionError, resolve


NOW = datetime.fromisoformat("2026-11-16T10:00:00+08:00")


@pytest.mark.parametrize("expression,expected", [
    ("明天下午六点", "2026-11-17T18:00:00+08:00"),
    ("明天18:30", "2026-11-17T18:30:00+08:00"),
    ("后天上午九点半", "2026-11-18T09:30:00+08:00"),
    ("2026-12-01 00:05", "2026-12-01T00:05:00+08:00"),
    ("明天下午六点三十分", "2026-11-17T18:30:00+08:00"),
])
def test_explicit_time(expression, expected):
    assert resolve(expression, NOW, "Asia/Shanghai").isoformat() == expected


@pytest.mark.parametrize("expression", ["明天六点", "明天晚上十二点", "明天上午十二点", "明天25:00", "明天18:99", "昨天18:00", "今天09:00", "明天18:00或后天18:00"])
def test_ambiguous_invalid_or_past_time_asks_instead_of_guessing(expression):
    with pytest.raises(TimeResolutionError):
        resolve(expression, NOW, "Asia/Shanghai")


def test_local_date_and_year_rollover():
    now = datetime.fromisoformat("2026-12-31T16:30:00+00:00")
    assert resolve("明天18:00", now, "Asia/Shanghai").isoformat() == "2027-01-02T18:00:00+08:00"


@pytest.mark.parametrize("expression,now", [
    ("2027-03-14 02:30", "2027-03-01T00:00:00-05:00"),
    ("2027-11-07 01:30", "2027-11-01T00:00:00-04:00"),
])
def test_dst_nonexistent_and_repeated_wall_times(expression, now):
    with pytest.raises(TimeResolutionError):
        resolve(expression, datetime.fromisoformat(now), "America/New_York")

