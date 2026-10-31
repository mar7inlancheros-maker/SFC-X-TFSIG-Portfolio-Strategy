"""Fechas de sesion de bolsa para las fichas: NYSE y Toronto con sus festivos.

La regla cuenta sesiones (`simulate_trades`: el dia k es la k-esima sesion
despues de la entrada). Con dias habiles a secas, una ventana de 63 sesiones
que cruza Thanksgiving, Navidad, Ano Nuevo y MLK terminaba cuatro sesiones
antes. Los cierres extraordinarios (lutos nacionales) no se conocen de
antemano y no estan.

Sin calendario para la bolsa (Londres, Paris...), se cuentan dias habiles y
la fecha puede quedar algun festivo corta.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from pandas.tseries.holiday import (
    MO,
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    next_monday,
    next_monday_or_tuesday,
    sunday_to_monday,
    weekend_to_monday,
)
from pandas.tseries.offsets import BDay, CustomBusinessDay


class NYSECalendar(AbstractHolidayCalendar):
    # Ano Nuevo en sabado no cierra el viernes anterior (31-dic-2021 abrio).
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


class TSXCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=next_monday),
        Holiday("Family Day", month=2, day=1, offset=pd.DateOffset(weekday=MO(3)), start_date="2008-01-01"),
        GoodFriday,
        # Lunes anterior al 25 de mayo.
        Holiday("Victoria Day", month=5, day=24, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday("Canada Day", month=7, day=1, observance=next_monday),
        Holiday("Civic Holiday", month=8, day=1, offset=pd.DateOffset(weekday=MO(1))),
        USLaborDay,
        Holiday("Thanksgiving", month=10, day=1, offset=pd.DateOffset(weekday=MO(2))),
        Holiday("Christmas", month=12, day=25, observance=weekend_to_monday),
        Holiday("Boxing Day", month=12, day=26, observance=next_monday_or_tuesday),
    ]


CANADA = (".TO", ".V", ".NE", ".CN")


@lru_cache(maxsize=None)
def _offset(market: str):
    if market == "US":
        return CustomBusinessDay(calendar=NYSECalendar())
    if market == "CA":
        return CustomBusinessDay(calendar=TSXCalendar())
    return BDay()


def market_of(ticker: str) -> str:
    """US sin sufijo (BRK-B), CA para Toronto y sus hermanas, OTHER el resto."""
    t = ticker.upper()
    if "." not in t:
        return "US"
    return "CA" if t.endswith(CANADA) else "OTHER"


def session_date(ticker: str, levels_date: pd.Timestamp, n: float) -> pd.Timestamp:
    """La sesion `n` despues de la de entrada; la de entrada (n = 0) es la
    siguiente a la de los niveles, que salen del ultimo cierre."""
    off = _offset(market_of(ticker))
    entry = pd.Timestamp(levels_date).normalize() + off
    return entry + int(round(n)) * off
