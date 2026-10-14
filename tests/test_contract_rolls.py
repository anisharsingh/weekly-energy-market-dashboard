"""Tests for contract_rolls.py. Run from the repo root:  pytest tests"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_rolls import BDAY, brent_expiry, roll_dates_by_week, wti_expiry  # noqa: E402


def d(s):
    return pd.Timestamp(s)


def test_wti_expiry_known_dates():
    assert wti_expiry(2026, 10) == d("2026-09-22")   # 25th is a Friday: 3 business days back
    assert wti_expiry(2026, 11) == d("2026-10-20")   # 25th is a Sunday: start from Friday the 23rd
    assert wti_expiry(2025, 9) == d("2025-08-20")
    assert wti_expiry(2025, 11) == d("2025-10-21")   # 25th is a Saturday
    assert wti_expiry(2025, 12) == d("2025-11-20")


def test_brent_expiry_known_dates():
    assert brent_expiry(2026, 11) == d("2026-09-30")  # last business day of September
    assert brent_expiry(2026, 12) == d("2026-10-30")  # Oct 31 2026 is a Saturday
    assert brent_expiry(2025, 10) == d("2025-08-29")  # Aug 31 2025 is a Sunday


def test_expiries_are_always_business_days():
    for year in (2025, 2026, 2027):
        for month in range(1, 13):
            assert BDAY.is_on_offset(wti_expiry(year, month))
            assert BDAY.is_on_offset(brent_expiry(year, month))


def test_roll_weeks_match_the_reported_rolls():
    weeks = pd.DatetimeIndex(["2026-09-18", "2026-09-25", "2026-10-02", "2026-10-09"])
    rolls = roll_dates_by_week(weeks)
    # WTI October contract expired Sep 22 -> week ending Sep 25
    assert rolls["wti_roll_date"] == {d("2026-09-25"): "2026-09-22"}
    # Brent November contract expired Sep 30 -> week ending Oct 2
    assert rolls["brent_roll_date"] == {d("2026-10-02"): "2026-09-30"}


def test_a_week_with_no_expiry_has_no_flag():
    rolls = roll_dates_by_week(pd.DatetimeIndex(["2026-10-09"]))
    assert rolls["wti_roll_date"] == {} and rolls["brent_roll_date"] == {}
