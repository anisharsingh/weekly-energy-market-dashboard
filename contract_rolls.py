"""Futures contract expiry dates and roll-week flags for WTI (NYMEX CL) and Brent (ICE).

Why this exists: Yahoo's continuous series (CL=F, BZ=F) switches from the expiring contract to the
next one around expiry. If the new contract trades at a different price, the weekly price change and
the Brent-WTI spread jump even though nothing happened in the market. Knowing which weeks contain an
expiry lets the notebook mark them.

Rules used (these follow the exchange contract specifications):
  WTI   Trading ends 3 business days before the 25th calendar day of the month BEFORE the delivery
        month. If the 25th is not a business day, count 3 business days back from the business day
        before the 25th.
  Brent Trading ends on the last business day of the SECOND month before the contract month
        (the November contract ends on the last business day of September).

Limits: business days use an approximate US exchange holiday calendar (Brent really follows the ICE
Futures Europe calendar, which has UK bank holidays), and Yahoo can switch contracts a day or two
away from the expiry date. A flag therefore means "a roll happened around this week", not an exact
timestamp.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay


class ExchangeHolidays(AbstractHolidayCalendar):
    """Approximate exchange closures: US federal holidays except Columbus Day and Veterans Day,
    plus Good Friday."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


BDAY = CustomBusinessDay(calendar=ExchangeHolidays())


def _is_business_day(d: pd.Timestamp) -> bool:
    return bool(BDAY.is_on_offset(d))


def wti_expiry(year: int, month: int) -> pd.Timestamp:
    """Last trading day of the WTI contract for delivery month (year, month)."""
    delivery = pd.Timestamp(year=year, month=month, day=1)
    prior = delivery - pd.DateOffset(months=1)
    the_25th = pd.Timestamp(year=prior.year, month=prior.month, day=25)
    reference = the_25th if _is_business_day(the_25th) else the_25th - BDAY
    return reference - 3 * BDAY


def brent_expiry(year: int, month: int) -> pd.Timestamp:
    """Last trading day of the Brent contract for contract month (year, month)."""
    two_before = pd.Timestamp(year=year, month=month, day=1) - pd.DateOffset(months=2)
    month_end = two_before + pd.offsets.MonthEnd(0)
    return month_end if _is_business_day(month_end) else month_end - BDAY


def roll_dates_by_week(weeks) -> dict[str, dict[pd.Timestamp, str]]:
    """For each week (labelled by its Friday; the week runs Saturday to Friday) return the expiry date
    of any WTI or Brent contract that expired inside that week, as 'YYYY-MM-DD'.

    Result: {"wti_roll_date": {friday: date_string}, "brent_roll_date": {...}}. Weeks without an
    expiry are simply absent from the inner dictionaries.
    """
    weeks = pd.DatetimeIndex(weeks)
    first, last = weeks.min() - pd.DateOffset(months=1), weeks.max() + pd.DateOffset(months=5)
    out = {"wti_roll_date": {}, "brent_roll_date": {}}
    for delivery in pd.date_range(first.replace(day=1), last, freq="MS"):
        for key, fn in (("wti_roll_date", wti_expiry), ("brent_roll_date", brent_expiry)):
            expiry = fn(delivery.year, delivery.month)
            for friday in weeks:
                if friday - pd.Timedelta(days=6) <= expiry <= friday:
                    out[key][friday] = expiry.strftime("%Y-%m-%d")
    return out
