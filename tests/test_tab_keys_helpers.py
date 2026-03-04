from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from openai_project_roles.app import tab_keys


def test_missing_day_ranges_usage_with_coverage() -> None:
    cached = pd.DataFrame(
        {
            "project_id": ["__coverage__"] * 2,
            "api_key_id": ["__coverage__"] * 2,
            "day": [date(2026, 1, 1), date(2026, 1, 3)],
            "metric": ["__coverage__"] * 2,
            "value": [0.0, 0.0],
        }
    )

    out = tab_keys._missing_day_ranges_usage(
        cached_df=cached,
        start_d=date(2026, 1, 1),
        end_d=date(2026, 1, 4),
    )

    assert out == [(date(2026, 1, 2), date(2026, 1, 2)), (date(2026, 1, 4), date(2026, 1, 4))]


def test_add_coverage_rows_usage() -> None:
    df = pd.DataFrame(columns=["project_id", "api_key_id", "day", "metric", "value"])
    out = tab_keys._add_coverage_rows_usage(df, date(2026, 1, 1), date(2026, 1, 2))
    assert len(out) == 2
    assert set(out["day"].tolist()) == {date(2026, 1, 1), date(2026, 1, 2)}


def test_usage_totals_dict_to_tidy_df() -> None:
    totals = {"key1": {"requests": 10, "tokens": 5}}
    df = tab_keys._usage_totals_dict_to_tidy_df("proj1", date(2026, 1, 1), totals)
    assert set(df.columns) == {"project_id", "api_key_id", "day", "metric", "value"}
    assert len(df) == 2
    assert set(df["metric"].tolist()) == {"requests", "tokens"}


def test_usage_totals_dict_to_tidy_df_drops_zero_values() -> None:
    totals = {"key1": {"requests": 0, "tokens": 5}}
    df = tab_keys._usage_totals_dict_to_tidy_df("proj1", date(2026, 1, 1), totals)
    assert len(df) == 1
    assert df.iloc[0]["metric"] == "tokens"


def test_safe_concat_usage_empty() -> None:
    out = tab_keys._safe_concat_usage()
    assert list(out.columns) == ["project_id", "api_key_id", "day", "metric", "value"]
    assert out.empty


def test_load_usage_by_key_cache_csv(tmp_path: Path) -> None:
    path = tmp_path / "usage_by_key.csv"
    path.write_text(
        "project_id,api_key_id,day,metric,value\np1,k1,2026-01-01,requests,2\n", encoding="utf-8"
    )
    df = tab_keys._load_usage_by_key_cache_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["project_id"] == "p1"


def test_api_key_names_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "api_key_names.csv"
    to_save = pd.DataFrame(
        {
            "api_key_id": ["key_a", "key_b"],
            "key_name": ["Finance ETL", "Support Bot"],
        }
    )
    tab_keys._save_api_key_names_csv(path, to_save)
    loaded = tab_keys._load_api_key_names_csv(path)
    assert len(loaded) == 2
    assert dict(zip(loaded["api_key_id"], loaded["key_name"]))["key_a"] == "Finance ETL"
