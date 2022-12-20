"""Contains functions to load data regarding the average irradiance of facilities."""

from __future__ import annotations

import calendar
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def load(filepath: str | Path) -> Dict[str, pd.DataFrame]:
    with open(filepath, "r") as file_reader:
        data_dict = json.load(file_reader)
    hourly_irr = data_dict["outputs"]["hourly"]

    irr_tot = dict()
    datetime_format = "%Y%m%d:%H%M"
    for month_num, month_name in enumerate(calendar.month_abbr[1:], start=1):
        # Extracts all entries which refer to current month.
        irr_current_month_all_hours = list(filter(
            lambda x: datetime.strptime(x["time"], datetime_format).month == month_num,
            hourly_irr
        ))
        irr_current_month = dict()
        # Iterate on even hour of current month and puts in 'irr_current_month' the
        # organized data for create a DataFrame.
        for irr_hour in irr_current_month_all_hours:
            irr_datetime = datetime.strptime(irr_hour["time"], datetime_format)
            if irr_datetime.day not in irr_current_month:
                irr_current_month[irr_datetime.day] = dict()
            irr_current_month[irr_datetime.day][irr_datetime.hour] = irr_hour["G(i)"]
        irr_tot[month_name] = pd.DataFrame(data=irr_current_month)

    return irr_tot


def load_months_avg(filepath: str | Path) -> pd.DataFrame:
    irr = load(filepath)
    irr_avg = {month: irr[month].mean(axis=1) for month in calendar.month_abbr[1:]}
    return pd.DataFrame(data=irr_avg)
