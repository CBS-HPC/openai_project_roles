from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from openai_project_roles.app import tab_usage


def test_missing_day_ranges_with_coverage() -> None:
    start_d = date(2026, 1, 1)
    end_d = date(2026, 1, 5)

    cached = pd.DataFrame(
        {
            "project_id": ["__coverage__"] * 3,
            "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 4)],
            "cost_usd": [0.0, 0.0, 0.0],
        }
    )

    ranges = tab_usage._missing_day_ranges(
        cached_df=cached,
        start_d=start_d,
        end_d=end_d,
        project_id_filter=None,
    )

    assert ranges == [(date(2026, 1, 3), date(2026, 1, 3)), (date(2026, 1, 5), date(2026, 1, 5))]


def test_add_coverage_rows_appends_days() -> None:
    base = pd.DataFrame(
        {
            "project_id": ["p1"],
            "day": [date(2026, 1, 1)],
            "cost_usd": [1.25],
        }
    )

    out = tab_usage._add_coverage_rows(base, date(2026, 1, 1), date(2026, 1, 3))

    cov = out[out["project_id"] == "__coverage__"]
    assert len(cov) == 3
    assert set(cov["day"].tolist()) == {date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)}
    assert (cov["cost_usd"] == 0.0).all()


def test_df_to_daily_costs_dict_drops_zero() -> None:
    df = pd.DataFrame(
        {
            "project_id": ["p1", "p1", "p2"],
            "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 1)],
            "cost_usd": [0.0, 2.5, 0.0],
        }
    )

    out = tab_usage._df_to_daily_costs_dict(df)
    assert out == {"p1": {"2026-01-02": 2.5}}


def test_daily_costs_dict_to_df_drops_invalid_dates() -> None:
    daily = {"p1": {"2026-01-01": 1.0, "bad-date": 2.0}}
    df = tab_usage._daily_costs_dict_to_df(daily)
    assert len(df) == 1
    assert df.iloc[0]["day"] == date(2026, 1, 1)


def test_safe_concat_daily_empty_result() -> None:
    out = tab_usage._safe_concat_daily()
    assert list(out.columns) == ["project_id", "day", "cost_usd"]
    assert out.empty


def test_load_usage_cache_csv_drops_last_day(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "project_id": ["p1", "p1"],
            "day": ["2026-01-01", "2026-01-02"],
            "cost_usd": [1.0, 2.0],
        }
    )
    path = tmp_path / "usage.csv"
    df.to_csv(path, index=False)

    loaded = tab_usage._load_usage_cache_csv(path)
    assert loaded["day"].max() == date(2026, 1, 1)
    assert len(loaded) == 1
